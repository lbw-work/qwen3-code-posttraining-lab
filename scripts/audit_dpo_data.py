"""只读复核 DPO 数据及核心数学；不修改数据锁、模型或训练参数。"""
import ast
import hashlib
import json
import random
import re
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from code_reward import score_batch
from train_dpo import answer_input, dpo_loss
from qwen_posttrain.policy import action_logprobs
from transformers import AutoTokenizer

from qwen_posttrain.artifacts import ROOT
OUT = ROOT / 'results/analysis/dpo-preference-audit-v1'
COMMIT = 'f8b8c0f49dc92a430bae41585f9d467d3618fe2f'


def digest(path):
    """读取文件内容计算 SHA256；用于确认复核对应训练时的同一份数据。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    """逐行解析 JSONL；空行跳过，格式错误直接报错，避免静默漏掉样本。"""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompt_key(prompt):
    """沿用数据准备阶段的小写词序列，忽略外层引号等格式差异后定位来源。"""
    return hashlib.sha256(' '.join(re.findall(r'[a-z0-9_]+', prompt.casefold())).encode()).hexdigest()


def reference_check():
    """对照作者固定版本的两个函数，检查损失、答案掩码和梯度。

    只提取两个数学函数，不导入作者仓库的训练器或依赖。随机 logits 使用
    CPU float32，包含不同题干长度、单 token 答案、EOS 和右侧 padding。
    这可以发现符号或错位错误；不代表已验证 CUDA 实际模型和优化器行为。
    """
    url = f'https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/{COMMIT}/trainers.py'
    source = urllib.request.urlopen(url, timeout=30).read()
    (OUT / 'reference-trainers.py.txt').write_bytes(source)
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in ('preference_loss', '_get_batch_logps')]
    assert len(selected) == 2
    namespace = {'torch': torch, 'F': F, 'Tuple': Tuple}
    exec(compile(ast.Module(body=selected, type_ignores=[]), url, 'exec'), namespace)
    torch.manual_seed(42)
    differences = Counter()
    for case in range(40):
        prompt_length, answer_length = 1 + case % 7, 1 + case % 11
        length = prompt_length + answer_length
        ids = torch.randint(0, 19, (1, length))
        local = torch.randn(1, length, 19, requires_grad=True)
        # 在右侧添加 3 个 pad；其标签是 -100，因此不应参与参考答案概率。
        padded = torch.cat([local.detach(), torch.randn(1, 3, 19)], dim=1).requires_grad_()
        labels = torch.cat([ids, torch.full((1, 3), -100)], dim=1)
        labels[:, :prompt_length] = -100
        ours = action_logprobs(local, ids, prompt_length).sum()
        theirs = namespace['_get_batch_logps'](padded, labels).sum()
        ours.backward(); theirs.backward()
        probability_error = abs((ours - theirs).item())
        gradient_error = (local.grad - padded.grad[:, :length]).abs().max().item()
        differences['logprob_max_error'] = max(differences['logprob_max_error'], probability_error)
        differences['logprob_gradient_max_error'] = max(differences['logprob_gradient_max_error'], gradient_error)
        assert torch.allclose(ours, theirs, atol=1e-5, rtol=1e-6)
        assert gradient_error < 1e-6 and padded.grad[:, length:].abs().max().item() == 0
        values = torch.randn(4, requires_grad=True)
        copied = values.detach().clone().requires_grad_()
        ours_loss = dpo_loss(*values.unbind())
        theirs_loss = namespace['preference_loss'](*copied.unbind(), beta=0.1)[0]
        ours_loss.backward(); theirs_loss.backward()
        differences['loss_max_error'] = max(differences['loss_max_error'], abs((ours_loss - theirs_loss).item()))
        differences['loss_gradient_max_error'] = max(differences['loss_gradient_max_error'], (values.grad - copied.grad).abs().max().item())
        assert torch.allclose(ours_loss, theirs_loss, atol=1e-7)
        assert torch.allclose(values.grad, copied.grad, atol=1e-7)
    return {'reference_commit': COMMIT, 'reference_url': url,
            'reference_sha256': hashlib.sha256(source).hexdigest(), 'cases': 40,
            'passed': True, 'differences': dict(differences),
            'scope': 'CPU float32 synthetic logits; no actual model forward or optimizer check'}


def main():
    """检查全量结构后，固定抽取 100 对，在 Docker 中执行双方已有测试。

    统计基于全量数据；执行结论仅覆盖随机样本及原有测试，不能证明正例在
    所有边界输入上正确。保存样本和逐对分数，便于独立查看和复跑。
    """
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / 'data/preference' / f'{split}.jsonl' for split in ('train', 'valid')]
    lock = json.loads((ROOT / 'data/preference.lock.json').read_text())
    rows = []
    for split, path in zip(('train', 'valid'), paths):
        assert digest(path) == lock[f'{split}_sha256']
        rows.extend(dict(row, split=split, row_number=i + 1) for i, row in enumerate(read_rows(path)))
    provenance_path = ROOT / 'data/preference/provenance.jsonl'
    assert digest(provenance_path) == lock['provenance_sha256']
    provenance = {row['prompt_sha256']: row for row in read_rows(provenance_path)}
    sft = {row['source_id']: row for row in read_rows(ROOT / 'data/sft/train.jsonl')}
    stats = Counter(); mutation_counts = Counter(); negative_scores = Counter()
    ast_keys = []; prompt_keys = []; splits = {'train': set(), 'valid': set()}
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/base', local_files_only=True)
    for row in rows:
        key = prompt_key(row['prompt']); meta = provenance[key]
        assert row['split'] == meta['split']
        assert meta['chosen_score'] == 1 and 0 < meta['rejected_score'] < 1
        mutation_counts[f"{meta['mutation']['old']} -> {meta['mutation']['new']}"] += 1
        negative_scores[str(meta['rejected_score'])] += 1
        chosen_tree, rejected_tree = ast.parse(row['chosen']), ast.parse(row['rejected'])
        assert ast.dump(chosen_tree) != ast.dump(rejected_tree)
        # 按来源记录的行列还原唯一改动，确认 rejected 确实只改了一个 token。
        mutation = meta['mutation']
        lines = row['chosen'].splitlines(keepends=True)
        offset = sum(map(len, lines[:mutation['line'] - 1])) + mutation['column']
        assert row['chosen'][offset:offset + len(mutation['old'])] == mutation['old']
        assert row['rejected'] == (row['chosen'][:offset] + mutation['new']
                                   + row['chosen'][offset + len(mutation['old']):])
        names = [n.name for n in chosen_tree.body if isinstance(n, ast.FunctionDef)]
        stats['prompt_mentions_any_function_name'] += int(any(re.search(r'\b' + re.escape(name) + r'\b', row['prompt']) for name in names))
        stats['chosen_rejected_same_function_names'] += int(names == [n.name for n in rejected_tree.body if isinstance(n, ast.FunctionDef)])
        original = sft.get(meta['source_id'])
        stats['source_seen_in_sft_train'] += int(original is not None)
        stats['chosen_exactly_sft_completion'] += int(original is not None and original['completion'] == row['chosen'])
        stats['valid_source_seen_in_sft_train'] += int(row['split'] == 'valid' and original is not None)
        # 沿用准备阶段忽略 docstring 的 AST 去重规则，独立复算来源键。
        for node in ast.walk(chosen_tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
        assert hashlib.sha256(ast.dump(chosen_tree, include_attributes=False).encode()).hexdigest() == meta['chosen_ast_sha256']
        for answer in (row['chosen'], row['rejected']):
            ids, prefix_length = answer_input(tokenizer, row['prompt'], answer, torch.device('cpu'))
            combined = tokenizer.encode(row['prompt'] + answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            stats['token_boundary_matches_concat'] += int(ids[0].tolist() == combined)
            assert prefix_length > 0 and ids.shape[1] <= 1024
        prompt_keys.append(key); ast_keys.append(meta['chosen_ast_sha256']); splits[row['split']].add(key)
        row['provenance'] = meta
    stats['duplicate_prompts'] = len(prompt_keys) - len(set(prompt_keys))
    stats['duplicate_chosen_ast'] = len(ast_keys) - len(set(ast_keys))
    stats['train_valid_prompt_overlap'] = len(splits['train'] & splits['valid'])
    sample = random.Random(20261004).sample(rows, 100)
    (OUT / 'sample.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in sample))
    image = subprocess.check_output(['docker', 'image', 'inspect', 'qwen-code-eval:0.3.1', '--format', '{{.Id}}'], text=True).strip()
    checks = []
    for start in range(0, len(sample), 10):
        batch = sample[start:start + 10]
        scores = score_batch([row[field] for row in batch for field in ('chosen', 'rejected')],
                             [row['provenance']['tests'] for row in batch for _ in range(2)])
        for i, row in enumerate(batch):
            meta = row['provenance']; chosen, rejected = scores[2*i:2*i+2]
            checks.append({'source_id': meta['source_id'], 'split': row['split'], 'row_number': row['row_number'],
                           'chosen_score': chosen, 'rejected_score': rejected,
                           'matched': chosen == meta['chosen_score'] and rejected == meta['rejected_score']})
        print(f'Docker checked {start + len(batch)}/100 pairs', flush=True)
    assert image == subprocess.check_output(['docker', 'image', 'inspect', 'qwen-code-eval:0.3.1', '--format', '{{.Id}}'], text=True).strip()
    result = {'pairs': len(rows), 'counts': dict(stats), 'mutations': dict(mutation_counts),
              'rejected_scores': dict(negative_scores), 'sample_seed': 20261004,
              'sample_checks': checks, 'sample_matched': sum(row['matched'] for row in checks),
              'docker_image_id': image, 'docker_image_matches_export': image == lock['execution_export']['docker_image_id'],
              'hashes': {str(path.relative_to(ROOT)): digest(path) for path in paths + [provenance_path, ROOT/'data/preference.lock.json', ROOT/'eval.jsonl', Path(__file__)]},
              'reference_check': reference_check()}
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('sample_checks', 'hashes')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
