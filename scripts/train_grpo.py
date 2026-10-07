"""手写 GRPO：从 LoRA SFT 采样代码，以隔离执行的测试通过率训练。"""

import argparse
import json
import math
import shutil
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from code_reward import score_batch, execution_info
from qwen_posttrain.artifacts import ROOT, sha256
from train_common import locked_jsonl, new_run, save_meta, sft_adapter
from qwen_posttrain.policy import action_logprobs, prompt_order

GROUP_SIZE = 4
MAX_PROMPT_TOKENS = 512
MAX_COMPLETION_TOKENS = 512
CLIP = 0.2
UPDATES_PER_GROUP = 2
TEMPERATURE = 0.8


def group_advantages(scores: list[float]) -> torch.Tensor:
    """把四个测试通过率变成均值为零的优势；同分组返回全零，不更新。

    例如 [1, 0, 0, 0] 的优势约为 [1.732, -0.577, -0.577, -0.577]。
    这里只比较同一道题的四份答案，不将不同题目的难度混在一起标准化。
    """
    values = torch.tensor(scores, dtype=torch.float32)
    if len(values) != GROUP_SIZE or not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('GRPO 必须收到四个 [0,1] 内的有限执行分')
    deviation = values - values.mean()
    scale = deviation.square().mean().sqrt()
    return deviation / scale if scale > 1e-8 else torch.zeros_like(values)


def grpo_loss(new_logp: torch.Tensor, old_logp: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    """计算一个答案的 token 级裁剪损失，并对答案长度取平均。

    new/old 都是生成 token 的 log probability，形状 [答案 token 数]；
    old 来自生成时的固定策略，绝不能在第二遍更新前重新计算。裁剪的是
    surrogate objective，不是直接限制参数或保证真实 KL 小于某个阈值。
    beta=0 的本实现不使用 reference KL，也不加载价值网络。
    """
    if new_logp.shape != old_logp.shape or new_logp.ndim != 1 or new_logp.numel() == 0:
        raise ValueError('新旧概率必须是相同长度的非空一维向量')
    ratio = (new_logp - old_logp).clamp(-20, 20).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - CLIP, 1 + CLIP) * advantage).mean()


def reward_environment() -> str:
    """在加载大模型前验证 Docker 与奖励链路；只在容器中执行小型探针。

    镜像 ID 写进运行元数据。不同 CPU 架构可能有不同镜像 ID，因此服务器
    另记录实际 ID，而不把 Mac 镜像 ID 当成跨平台相等的必要条件。
    """
    image = execution_info()['docker_image_id']
    tests = ['assert f(2) == 3', 'assert f(5) == 6']
    scores = score_batch(['def f(x): return x + 1', 'def f(x): return 0'], [tests, tests])
    if scores != [1.0, 0.0]:
        raise ValueError(f'Docker 奖励探针异常：{scores}')
    return image


def load_grpo_data(tokenizer) -> tuple[list[dict], list[dict], dict]:
    """加载专用冻结数据并逐行验证，不在训练时截断或临时改变数据池。

    prompt 只能来自已校验的 GRPO 数据，train/valid 不能混题。参考答案仅存在
    provenance，不被本函数读取；模型采样时只看到 prompt，奖励只读取 tests。
    """
    train_file, valid_file, lock = locked_jsonl(ROOT / 'data/grpo', 'grpo')
    if lock['profile'] != 'grpo-execution-audited-v1' or lock['quality_sha256'] != sha256(ROOT/'data/grpo-quality.json'):
        raise ValueError('GRPO 补审未完成或质量报告发生变化')
    if lock.get('semantic_review_sha256') != sha256(ROOT/'data/grpo-semantic-review.json'):
        raise ValueError('GRPO 语义复核尚未完成或版本不一致')
    if lock['worker_sha256'] != sha256(ROOT / 'scripts/code_reward_worker.py'):
        raise ValueError('执行 worker 与数据复核时不同，需重新审核')
    if lock['tokenizer_sha256'] != sha256(ROOT / 'models/base/tokenizer.json'):
        raise ValueError('Base tokenizer 与数据冻结时不同')
    if lock['max_prompt_tokens'] != MAX_PROMPT_TOKENS or lock['max_completion_tokens'] != MAX_COMPLETION_TOKENS:
        raise ValueError('数据长度协议与训练不一致')
    # tokenizer 保存时 JSON 的默认字段可能被补全，文件哈希不同并不等于分词不同。
    # 用冻结 Base 的词表、特殊 token 和每道实际题干的 token ID 检查语义兼容性。
    base_tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/base', local_files_only=True)
    if tokenizer.get_vocab() != base_tokenizer.get_vocab() or tokenizer.special_tokens_map != base_tokenizer.special_tokens_map:
        raise ValueError('SFT 与冻结 Base 的词表或特殊 token 不一致')
    all_ids, partitions = set(), []
    for path in (train_file, valid_file):
        rows = [json.loads(line) for line in path.open()]
        if not rows:
            raise ValueError('GRPO train/valid 不得为空')
        for row in rows:
            if row['source_id'] in all_ids:
                raise ValueError('数据 source_id 重复或跨划分重叠')
            all_ids.add(row['source_id'])
            ids = tokenizer.encode(row['prompt'], add_special_tokens=False)
            if ids != base_tokenizer.encode(row['prompt'], add_special_tokens=False):
                raise ValueError('SFT tokenizer 改变了冻结题干的 token ID')
            length = len(ids)
            if not 0 < length <= MAX_PROMPT_TOKENS:
                raise ValueError('题干为空或超长；拒绝静默截断')
            if not row['tests'] or not all(isinstance(t, str) and t.strip() for t in row['tests']):
                raise ValueError('奖励测试为空或格式错误')
        partitions.append(rows)
    return partitions[0], partitions[1], lock


def generate_group(policy, tokenizer, row: dict, *, greedy: bool = False):
    """采样并保留原始输出、旧概率、是否达到长度上限；不自动清洗代码。

    输入 ids 是 [1, 题干长度]；完整返回序列是 [1, 题干+答案长度]。
    四份随机答案共用同一未更新策略。开发集则只做一次贪心生成。
    达到上限且没有 EOS 记为 hit_length_limit，不武断认定一定语法错误。
    """
    device = next(policy.parameters()).device
    prompt_ids = tokenizer(row['prompt'], return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    length = prompt_ids.shape[1]
    if not 0 < length <= MAX_PROMPT_TOKENS:
        raise ValueError('题干为空或超长')
    policy.eval()
    rollouts, outputs = [], []
    with torch.no_grad():
        for _ in range(1 if greedy else GROUP_SIZE):
            options = {} if greedy else {'temperature': TEMPERATURE, 'top_k': 0, 'top_p': 1.0}
            ids = policy.generate(input_ids=prompt_ids, do_sample=not greedy, use_cache=True,
                                  max_new_tokens=MAX_COMPLETION_TOKENS, pad_token_id=tokenizer.eos_token_id,
                                  **options)
            answer = ids[0, length:]
            if answer.numel() == 0:
                raise ValueError('模型未生成任何 token')
            outputs.append({'completion': tokenizer.decode(answer, skip_special_tokens=True),
                            'response_tokens': answer.numel(),
                            'hit_length_limit': answer.numel() == MAX_COMPLETION_TOKENS and answer[-1].item() != tokenizer.eos_token_id})
            if not greedy:
                old_logp = action_logprobs(policy(ids).logits, ids, length, TEMPERATURE).detach()
                rollouts.append((ids, old_logp))
    return length, rollouts, outputs


def update_group(policy, optimizer, rollouts, prompt_length: int, advantages: torch.Tensor) -> dict:
    """对有效组更新两遍；每遍累积四份答案梯度后才 optimizer.step。

    policy 只包含可训练 LoRA 参数；reference 不存在。old_logp 固定，new_logp
    每遍重算。记录损失和裁剪比例以诊断更新；同分组不调用 optimizer。
    """
    if len(rollouts) != GROUP_SIZE or len(advantages) != GROUP_SIZE:
        raise ValueError('一组必须有四份候选与四个优势')
    if not torch.count_nonzero(advantages):
        return {'updates': 0, 'loss': None, 'clip_fraction': None}
    policy.train()
    parameters = [p for p in policy.parameters() if p.requires_grad]
    losses, clipped = [], []
    for _ in range(UPDATES_PER_GROUP):
        optimizer.zero_grad(set_to_none=True)
        for (ids, old_logp), advantage in zip(rollouts, advantages, strict=True):
            new_logp = action_logprobs(policy(ids).logits, ids, prompt_length, TEMPERATURE)
            loss = grpo_loss(new_logp, old_logp, advantage) / GROUP_SIZE
            if not torch.isfinite(loss):
                raise FloatingPointError('GRPO 损失不是有限数')
            loss.backward()
            losses.append(loss.detach().item() * GROUP_SIZE)
            ratio = (new_logp.detach() - old_logp).exp()
            clipped.append(((ratio < 1 - CLIP) | (ratio > 1 + CLIP)).float().mean().item())
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
    return {'updates': UPDATES_PER_GROUP, 'loss': sum(losses)/len(losses), 'clip_fraction': sum(clipped)/len(clipped)}


def evaluate_dev(policy, tokenizer, rows: list[dict], path: Path) -> dict:
    """在专用 valid 上贪心生成并执行训练来源测试，用于选 checkpoint。

    只读 valid 的题干和测试，不访问 EvalPlus；按全部测试通过的题数选择模型。
    保存每题完整输出，便于解释分数变化；相同分数不覆盖更早的最佳 checkpoint。
    """
    passed, score_sum = 0, 0.0
    with path.open('w') as stream:
        for row in rows:
            _, _, output = generate_group(policy, tokenizer, row, greedy=True)
            score = score_batch([output[0]['completion']], [row['tests']])[0]
            passed += score == 1.0; score_sum += score
            stream.write(json.dumps({'source_id': row['source_id'], **output[0], 'reward': score}, ensure_ascii=False)+'\n')
            stream.flush()
    return {'passed': passed, 'total': len(rows), 'pass_rate': passed/len(rows), 'mean_reward': score_sum/len(rows)}


def save_adapter(policy, tokenizer, folder: Path) -> None:
    """适配器与 tokenizer 一起保存，使中间 checkpoint 也能直接用于统一评测。"""
    policy.save_pretrained(folder)
    tokenizer.save_pretrained(folder)


def main() -> None:
    """默认单卡正式训练；--preflight-only 仅采样判分，可在本地 MPS 运行。

    预检不会创建优化器或修改权重，也不称为训练完成。正式训练记录真实候选，
    按开发集选择 best，并分别保存 last_model，防止将末次权重误当最佳权重。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft-run', required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--max-steps', type=int, default=200, help='采样组数，不是更新次数')
    parser.add_argument('--max-hours', type=float, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-every', type=int, default=50, help='每多少组用冻结 valid 评估')
    parser.add_argument('--preflight-only', action='store_true', help='仅采样/判分，不更新；支持本地 MPS')
    args = parser.parse_args()
    if min(args.max_steps, args.eval_every) < 1 or not math.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error('组数、验证间隔、时间必须为有限正数')
    if not args.preflight_only and (torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        parser.error('正式训练要求一张支持 bf16 的 CUDA 显卡；本地只可 --preflight-only')
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    adapter = sft_adapter(ROOT / args.sft_run)
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    train_rows, valid_rows, lock = load_grpo_data(tokenizer)
    image = reward_environment()
    # 只有数据、起点和 Docker 都可用才创建独立目录，避免假冒“已经开始训练”。
    run = new_run('grpo_preflight' if args.preflight_only else 'grpo', args.run_id)
    torch.manual_seed(args.seed)
    # CUDA 正式训练明确使用 bf16；本地预检沿用权重精度，避免假设 MPS 等同 CUDA。
    base = AutoModelForCausalLM.from_pretrained(ROOT/'models/base', dtype=torch.bfloat16 if device == 'cuda' else 'auto', attn_implementation='sdpa', local_files_only=True)
    policy = PeftModel.from_pretrained(base, adapter, is_trainable=not args.preflight_only).to(device)
    # 前后两次概率比较不能混入 dropout 随机差异；本实现仅让 LoRA 参与优化。
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.0
    policy.config.use_cache = False
    if not args.preflight_only:
        policy.gradient_checkpointing_enable(); policy.enable_input_require_grads()
    optimizer = None if args.preflight_only else torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=5e-6)
    # 固定种子打乱题目；遍历一轮前不会重复取题，不按生成正确率挑“容易题”。
    order = prompt_order(len(train_rows), args.seed)
    started = time.monotonic()
    completed, updates, zero_groups, truncated = 0, 0, 0, 0
    best_passed, best_group = -1, None
    validation = []
    best_path = run/'checkpoints/best'
    if not args.preflight_only:
        # 起始 SFT 也参与开发集比较。后续没有超过它时，best 保留起点，last
        # 仍保存训练末态；因此分析提升必须查看 best_group 和 validation。
        metric = evaluate_dev(policy, tokenizer, valid_rows, run/'dev-group-0.jsonl')
        validation.append({'group': 0, **metric}); best_passed, best_group = metric['passed'], 0
        save_adapter(policy, tokenizer, best_path)
    with (run/'train_log.jsonl').open('w') as log, (run/'rollouts.jsonl').open('w') as evidence:
        for step in range(1, args.max_steps + 1):
            if time.monotonic()-started >= args.max_hours*3600: break
            row = train_rows[next(order)]
            # 先生成四份答案再执行原测试；参考答案不会传入模型或作为候选补位。
            length, rollouts, output = generate_group(policy, tokenizer, row)
            scores = score_batch([x['completion'] for x in output], [row['tests']]*GROUP_SIZE)
            advantages = group_advantages(scores).to(device)
            zero_groups += not bool(torch.count_nonzero(advantages))
            truncated += sum(x['hit_length_limit'] for x in output)
            evidence.write(json.dumps({'group': step, 'source_id': row['source_id'], 'prompt': row['prompt'],
                                       'candidates': [{**x, 'reward': score} for x,score in zip(output,scores,strict=True)]}, ensure_ascii=False)+'\n'); evidence.flush()
            outcome = {'updates': 0, 'loss': None, 'clip_fraction': None} if args.preflight_only else update_group(policy,optimizer,rollouts,length,advantages)
            # 组数与更新数分别累计：一组可能更新两次，也可能因奖励同分零更新。
            updates += outcome['updates']; completed = step
            record = {'group': step, 'source_id': row['source_id'], 'rewards': scores, 'advantages': advantages.tolist(),
                      'optimizer_steps': updates, 'zero_variance_groups': zero_groups, 'effective_groups': step-zero_groups,
                      'length_limit_candidates': truncated, **outcome, 'elapsed_seconds': time.monotonic()-started,
                      'cuda_peak_bytes': torch.cuda.max_memory_allocated() if device=='cuda' else None}
            if not args.preflight_only and (step % args.eval_every == 0 or step == args.max_steps):
                # 开发集按同一贪心协议复测。训练期间不读取公开 EvalPlus 的测试。
                metric = evaluate_dev(policy,tokenizer,valid_rows,run/f'dev-group-{step}.jsonl')
                validation.append({'group': step, **metric}); record['dev'] = metric
                save_adapter(policy,tokenizer,run/'checkpoints'/f'group-{step}')
                if metric['passed'] > best_passed:
                    best_passed,best_group = metric['passed'],step
                    save_adapter(policy,tokenizer,best_path)
            log.write(json.dumps(record)+'\n'); log.flush(); print(record,flush=True)
    if not args.preflight_only:
        # 若预算在两次周期验证之间用完，补测实际末态，再分别保存 last 和 best。
        if completed and validation[-1]['group'] != completed:
            metric = evaluate_dev(policy,tokenizer,valid_rows,run/f'dev-group-{completed}.jsonl')
            validation.append({'group': completed, **metric})
            if metric['passed'] > best_passed:
                best_passed,best_group = metric['passed'],completed; save_adapter(policy,tokenizer,best_path)
        save_adapter(policy,tokenizer,run/'last_model')
        shutil.copytree(best_path,run/'final_model')
    status = ('signal_observed' if completed-zero_groups else 'no_group_contrast') if args.preflight_only else (
        'no_parameter_updates' if not updates else 'completed' if completed == args.max_steps else 'time_budget_reached')
    save_meta(run, stage='grpo_preflight' if args.preflight_only else 'grpo', implementation='manual', status=status,
              sft_run=str(adapter.parent), sft_adapter_sha256=sha256(adapter/'adapter_model.safetensors'),
              sft_adapter_config_sha256=sha256(adapter/'adapter_config.json'), grpo_lock_sha256=sha256(ROOT/'data/grpo.lock.json'),
              sft_tokenizer_sha256=sha256(adapter/'tokenizer.json'),
              train_sha256=lock['train_sha256'],valid_sha256=lock['valid_sha256'],eval_sha256=lock['eval_sha256'],
              worker_sha256=lock['worker_sha256'],docker_image_id=image,device=device,
              reward_backend='ssh_tunnel_mac_docker' if os.environ.get('QWEN_REWARD_URL') else 'local_docker',
              completed_groups=completed,requested_groups=args.max_steps,optimizer_steps=updates,zero_variance_groups=zero_groups,
              length_limit_candidates=truncated,best_group=best_group,validation=validation,
              train_runtime_seconds=time.monotonic()-started,max_hours=args.max_hours,
              settings={'group_size':GROUP_SIZE,'max_prompt_tokens':MAX_PROMPT_TOKENS,'max_completion_tokens':MAX_COMPLETION_TOKENS,
                        'clip':CLIP,'updates_per_group':UPDATES_PER_GROUP,'temperature':TEMPERATURE,'top_k':0,'top_p':1.0,
                        'learning_rate':5e-6,'beta':0.0,'seed':args.seed,'eval_every':args.eval_every})
    print(f'GRPO {status}：{run}',flush=True)


if __name__ == '__main__':
    main()
