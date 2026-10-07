"""为 GRPO 冻结函数题：不截断题干，先在 Docker 复核参考答案与训练测试。"""

import argparse
import ast
import hashlib
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from audit_preference_examples import EXAMPLES, example_test
from code_reward import score_batch
from qwen_posttrain.artifacts import ROOT, sha256, verified_tasks
from prepare_preference import function_pair_reason, overlaps_eval_function, python_code
from prepare_sft import eval_ngrams, overlaps_eval, words
from train_common import locked_jsonl

MAX_PROMPT_TOKENS = 512
MAX_COMPLETION_TOKENS = 512


def implementation_key(code: str) -> str:
    """忽略注释与 docstring 后计算代码指纹，只用于去重，不修改训练题或答案。"""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def checked_tests(tests: list[str]) -> list[str]:
    """仅接受单条 assert 测试，并按 AST 去重；空测试或无断言代码不能贡献奖励。

    这里仅解析语法，不运行代码。至少五条不同断言才能进入候选池；真正的执行
    留给 Docker。断言通过仍不证明题干所有约束正确，报告中必须保留这一限制。
    """
    unique = {}
    if not isinstance(tests, list):
        return []
    for test in tests:
        if not isinstance(test, str):
            return []
        try:
            tree = ast.parse(test)
        except SyntaxError:
            return []
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
            return []
        unique.setdefault(ast.dump(tree, include_attributes=False), test.strip())
    return list(unique.values()) if 5 <= len(unique) <= 20 else []


def null_answer(code: str) -> str:
    """构造接口相同、全部返回 None 的探针，检查测试是否连明显空实现都放行。

    探针仅用于拒绝无区分力的测试，不是偏好负例，也不会写入 GRPO 模型输入。
    """
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.body = [ast.Return(value=ast.Constant(value=None))]
    return ast.unparse(ast.fix_missing_locations(tree))


def main() -> None:
    """筛选、执行复核并冻结数据；开发集只来自原 valid，不移动原始划分。

    训练文件只存题干和测试；参考答案另存 provenance，用于追溯筛选原因。
    先固定静态筛选，再按种子随机取题、复核测试；不按模型 rollout 表现挑题。
    不改已有 SFT、RL、PPO 或偏好数据。中断时留下的工作目录拒绝自动覆盖。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-count', type=int, default=2048)
    parser.add_argument('--valid-count', type=int, default=128)
    args = parser.parse_args()
    if min(args.train_count, args.valid_count) < 1:
        parser.error('数量必须为正数')
    folder = ROOT / 'data/grpo'
    lock_path = ROOT / 'data/grpo.lock.json'
    if folder.exists() or lock_path.exists():
        raise FileExistsError('GRPO 数据已存在，不覆盖')
    rl_train, rl_valid, rl_lock = locked_jsonl(ROOT / 'data/rl', 'rl')
    sft_train, sft_valid, _ = locked_jsonl(ROOT / 'data/sft', 'sft')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/base', local_files_only=True)
    tasks = verified_tasks()
    ngrams = eval_ngrams(tasks)
    names = {re.sub(r'[^a-z0-9]', '', t['entry_point'].casefold()) for t in tasks}
    image = subprocess.check_output(['docker', 'image', 'inspect', 'qwen-code-eval:0.3.1', '--format', '{{.Id}}'], text=True).strip()
    assert score_batch(['def f(x): return x + 1', 'def f(x): return 0'], [['assert f(2) == 3']] * 2) == [1.0, 0.0]
    pools, counts = {}, {}
    raw_train_prompts, raw_train_codes = set(), set()
    for split, rl_file, sft_file in [('train', rl_train, sft_train), ('valid', rl_valid, sft_valid)]:
        sft = {x['source_id']: x for x in map(json.loads, sft_file.open())}
        if split == 'train':
            raw_train_prompts = {' '.join(words(x['prompt'])) for x in sft.values()}
            raw_train_codes = {implementation_key(x['completion']) for x in sft.values()}
        candidates, counter = [], Counter()
        for row in map(json.loads, rl_file.open()):
            counter['input_rows'] += 1
            ref = sft[row['source_id']]
            if row['prompt'] != ref['prompt']:
                raise ValueError('RL/SFT 题干不一致')
            code = python_code(ref['completion'])
            if code is None:
                counter['invalid_reference'] += 1; continue
            tokens = len(tokenizer.encode(row['prompt'], add_special_tokens=False))
            answer_tokens = len(tokenizer.encode(code, add_special_tokens=False)) + 1
            if tokens > MAX_PROMPT_TOKENS or tokens == 0:
                counter['prompt_over_limit'] += 1; continue
            if answer_tokens > MAX_COMPLETION_TOKENS:
                counter['reference_over_limit'] += 1; continue
            probe = null_answer(code)
            question = re.sub(r'```[\s\S]*?```', '', row['prompt'])
            question = re.sub(r'sample input|input format', '', question, flags=re.I)
            reason = function_pair_reason(question, code, probe)
            if reason:
                counter[reason] += 1; continue
            if overlaps_eval(row['prompt'], ngrams) or overlaps_eval_function(code, probe, names):
                counter['eval_overlap'] += 1; continue
            tests = checked_tests(row['tests'])
            if not tests:
                counter['invalid_or_insufficient_tests'] += 1; continue
            key = implementation_key(code)
            prompt_key = ' '.join(words(row['prompt']))
            if split == 'valid' and (key in raw_train_codes or prompt_key in raw_train_prompts):
                counter['overlap_with_sft_train'] += 1; continue
            examples = [t for a, b in EXAMPLES.findall(row['prompt']) if (t := example_test({'input': row['prompt'], 'chosen': code}, a, b))]
            entry = next(n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
            candidates.append({**row, 'tests': tests, 'reference': code, 'probe': probe,
                               'implementation': key, 'prompt_key': prompt_key, 'entry_point': entry,
                               'prompt_tokens': tokens, 'reference_tokens': answer_tokens, 'examples': examples})
        random.Random(42).shuffle(candidates)
        counter['static_candidates'] = len(candidates)
        pools[split], counts[split] = candidates, counter
        print(split, dict(counter), flush=True)
    folder.mkdir()
    selected, seen_ids, seen_codes, seen_prompts = {}, set(), set(), set()
    dev_names = set()
    evidence = []
    # 先选开发集，再限制训练池：两边不能出现相同代码、规范题干或函数入口。
    with (folder / 'audit.jsonl').open('w') as audit:
        for split, target in [('valid', args.valid_count), ('train', args.train_count)]:
            accepted, entry_counts = [], Counter()
            for candidate in pools[split]:
                if len(accepted) >= target:
                    break
                if candidate['source_id'] in seen_ids or candidate['implementation'] in seen_codes or candidate['prompt_key'] in seen_prompts:
                    counts[split]['duplicate'] += 1; continue
                entry = candidate['entry_point']
                if (split == 'train' and entry in dev_names) or entry_counts[entry] >= 5:
                    counts[split]['entry_cap_or_dev_overlap'] += 1; continue
                reference_score, probe_score = score_batch([candidate['reference'], candidate['probe']], [candidate['tests']] * 2)
                example_score = score_batch([candidate['reference']], [candidate['examples']])[0] if candidate['examples'] else None
                decision = reference_score == 1.0 and probe_score < 1.0 and example_score in (None, 1.0)
                record = {'source_id': candidate['source_id'], 'split': split, 'reference_score': reference_score,
                          'null_probe_score': probe_score, 'example_score': example_score, 'accepted': decision}
                audit.write(json.dumps(record) + '\n'); audit.flush()
                if not decision:
                    counts[split]['execution_rejected'] += 1; continue
                accepted.append({k: candidate[k] for k in ('source_id', 'prompt', 'tests', 'entry_point')})
                evidence.append({**record, **{k: candidate[k] for k in ('reference', 'implementation', 'prompt_tokens', 'reference_tokens', 'examples')}})
                seen_ids.add(candidate['source_id']); seen_codes.add(candidate['implementation']); seen_prompts.add(candidate['prompt_key'])
                entry_counts[entry] += 1
                if split == 'valid': dev_names.add(entry)
                if len(accepted) % 50 == 0: print(split, len(accepted), '/', target, flush=True)
            if len(accepted) < target:
                raise ValueError(f'{split} 只有 {len(accepted)} 条通过，不降低门槛凑数；审核记录已保留')
            selected[split] = accepted
    for split, rows in selected.items():
        (folder / f'{split}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    (folder / 'provenance.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in evidence))
    assert image == subprocess.check_output(['docker','image','inspect','qwen-code-eval:0.3.1','--format','{{.Id}}'], text=True).strip()
    lock = {'profile': 'grpo-execution-checked-v1', 'counts': {k: len(v) for k,v in selected.items()},
            'filter_counts': {k: dict(v) for k,v in counts.items()}, 'eval_sha256': rl_lock['eval_sha256'],
            'rl_lock_sha256': sha256(ROOT/'data/rl.lock.json'), 'sft_lock_sha256': sha256(ROOT/'data/sft.lock.json'),
            'train_sha256': sha256(folder/'train.jsonl'), 'valid_sha256': sha256(folder/'valid.jsonl'),
            'provenance_sha256': sha256(folder/'provenance.jsonl'), 'audit_sha256': sha256(folder/'audit.jsonl'),
            'worker_sha256': sha256(ROOT/'scripts/code_reward_worker.py'), 'builder_sha256': sha256(Path(__file__)),
            'tokenizer_sha256': sha256(ROOT/'models/base/tokenizer.json'), 'docker_image_id': image,
            'max_prompt_tokens': MAX_PROMPT_TOKENS, 'max_completion_tokens': MAX_COMPLETION_TOKENS,
            'seed': 42, 'limitation': '训练测试和部分可解析题干示例的执行复核；不证明所有输入正确。valid 曾用于 SFT 验证，不是未触碰测试集。'}
    lock_path.write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
    print('GRPO 数据冻结完成', lock['counts'], flush=True)


if __name__ == '__main__':
    main()
