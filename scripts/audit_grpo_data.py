"""补审 GRPO 的题干示例和测试区分力；执行全部留在 Docker 中。"""

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from audit_preference_examples import example_test
from code_reward import score_batch
from export_verified_preference import mutations
from qwen_posttrain.artifacts import ROOT, sha256
from train_common import locked_jsonl


# 兼容原来源的 Sample Input 1 / Sample Output 1，编号必须对应。
# 仅解析明确的围栏字面量；不猜测自然语言期望，不在主机执行输入代码。
EXAMPLES = re.compile(
    r'\*\*Sample Input\s*(\d*):\*\*\s*```(?:python)?\s*\n(.*?)\n```\s*'
    r'\*\*Sample Output\s*\1:\*\*\s*```(?:python)?\s*\n(.*?)\n```', re.S | re.I)


def expanded_examples(prompt: str, reference: str) -> tuple[list[str], int]:
    """返回无歧义示例断言及匹配示例数量；未解析的数量单独记录。

    example_test 用 AST 字面量建立函数调用，不能处理的多解/自然语言示例保持
    未检查状态。解析成功的断言会追加到训练测试，让题干示例也约束奖励。
    """
    matches = EXAMPLES.findall(prompt)
    tests = [test for _, sample_input, sample_output in matches
             if (test := example_test({'input': prompt, 'chosen': reference}, sample_input, sample_output))]
    return list(dict.fromkeys(tests)), len(matches)


def zero_probe(reference: str) -> str:
    """保留函数接口，把实现替换成 return 0，检查恒定输出是否能骗过测试。

    这不是模型候选，也不是偏好数据。只有用于数据审查的坏实现，不能算作
    模型通过率或被加入训练答案。实际执行仍在受限容器里进行。
    """
    tree = ast.parse(reference)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.body = [ast.Return(value=ast.Constant(value=0))]
    return ast.unparse(ast.fix_missing_locations(tree))


def main() -> None:
    """先保存逐题证据，再可选择冻结审查后的版本，原版本归档不删除。

    每题检查参考答案、恒零探针及最多三份逻辑变异。变异全过并不证明测试错，
    可能存在等价变异；这里保守排除以降低弱测试风险，报告明确这一取舍。
    finalize 不运行模型，也不根据模型采样成功率挑题；train/valid 原划分保留。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize', action='store_true', help='用已完成报告冻结通过审查的题')
    args = parser.parse_args()
    folder = ROOT/'data/grpo'
    report_path = ROOT/'data/grpo-quality.json'
    audit_path = ROOT/'data/raw/grpo-preparation/quality-audit.jsonl'
    train_file, valid_file, lock = locked_jsonl(folder, 'grpo')
    partitions = {split: [json.loads(line) for line in path.open()]
                  for split, path in [('train',train_file),('valid',valid_file)]}
    provenance = {row['source_id']: row for row in map(json.loads, (folder/'provenance.jsonl').open())}
    if sha256(folder/'provenance.jsonl') != lock['provenance_sha256'] or sha256(ROOT/'scripts/code_reward_worker.py') != lock['worker_sha256']:
        raise ValueError('溯源记录或 worker 发生变化')
    if args.finalize:
        report = json.loads(report_path.read_text())
        archive = audit_path.parent/'initial-data'
        # 补审得分绑定最初输入。重新执行 finalize 时从已归档的同一输入重建，
        # 不重复运行执行审查，也不把第一次输出误当新的原始来源。
        if lock['profile'] == 'grpo-execution-audited-v1':
            if report['input_lock_sha256'] != sha256(archive/'grpo.lock.json'):
                raise ValueError('归档初始版本不匹配')
            original_lock = json.loads((archive/'grpo.lock.json').read_text())
            for split in ('train','valid'):
                if sha256(archive/f'{split}.jsonl') != original_lock[f'{split}_sha256']:
                    raise ValueError('归档划分哈希不匹配')
            if sha256(archive/'provenance.jsonl') != original_lock['provenance_sha256']:
                raise ValueError('归档溯源哈希不匹配')
            partitions = {split:list(map(json.loads,(archive/f'{split}.jsonl').open())) for split in ('train','valid')}
            provenance = {row['source_id']:row for row in map(json.loads,(archive/'provenance.jsonl').open())}
        elif report['input_lock_sha256'] != sha256(ROOT/'data/grpo.lock.json'):
            raise ValueError('审查初始输入发生变化')
        if report['audit_sha256'] != sha256(audit_path):
            raise ValueError('审查输入或逐题证据发生变化，拒绝冻结')
        audit = {row['source_id']:row for row in map(json.loads,audit_path.open())}
        accepted = {split:[row for row in rows if audit[row['source_id']]['accepted']] for split,rows in partitions.items()}
        # 顶层第一个 def 可能只是 helper，真实入口必须由测试实际调用的函数确定。
        # 只保留单一公共入口的题，并用真实入口重新检查跨划分重合及数量上限。
        entry_counts = Counter()
        entry_exclusions,entry_corrections = [],[]
        for split,rows in accepted.items():
            checked = []
            for row in rows:
                definitions = {n.name for n in ast.parse(provenance[row['source_id']]['reference']).body if isinstance(n,ast.FunctionDef)}
                called = {n.func.id for test in row['tests'] for n in ast.walk(ast.parse(test))
                          if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in definitions}
                if len(called) != 1:
                    entry_counts[f'{split}_multiple_public_entries'] += 1
                    entry_exclusions.append({'source_id':row['source_id'],'split':split,'reason':'multiple_public_entries'})
                    continue
                entry = next(iter(called))
                entry_counts['corrected_helper_entry'] += entry != row['entry_point']
                if entry != row['entry_point']:
                    entry_corrections.append({'source_id':row['source_id'],'old':row['entry_point'],'actual':entry})
                row['entry_point'] = entry
                checked.append(row)
            accepted[split] = checked
        dev_names = {row['entry_point'] for row in accepted['valid']}
        train_counts,filtered = Counter(),[]
        for row in accepted['train']:
            entry = row['entry_point']
            if entry in dev_names:
                entry_counts['train_actual_entry_in_valid'] += 1
                entry_exclusions.append({'source_id':row['source_id'],'split':'train','reason':'actual_entry_in_valid'})
            elif train_counts[entry] >= 5:
                entry_counts['train_actual_entry_cap'] += 1
                entry_exclusions.append({'source_id':row['source_id'],'split':'train','reason':'actual_entry_cap'})
            else:
                filtered.append(row); train_counts[entry] += 1
        accepted['train'] = filtered
        # 执行全过不能证明题意一致；应用单独保存、可审阅的语义错配排除记录。
        # 记录绑定初筛版本及执行审查，不能把旧人工审核套到别的数据版本上。
        semantic_path = ROOT/'data/grpo-semantic-review.json'
        semantic = json.loads(semantic_path.read_text())
        if semantic['source_initial_lock_sha256'] != report['input_lock_sha256'] or semantic['quality_audit_sha256'] != report['audit_sha256']:
            raise ValueError('语义复核记录不属于当前数据版本')
        excluded_ids = {item['source_id'] for item in semantic['exclusions']}
        accepted = {split:[row for row in rows if row['source_id'] not in excluded_ids] for split,rows in accepted.items()}
        if len(accepted['train']) < 1500 or len(accepted['valid']) < 100:
            raise ValueError('审查后数量不足，应从原来源继续严格筛选；不降低质量门槛')
        if not archive.exists():
            shutil.copytree(folder,archive)
            shutil.copy2(ROOT/'data/grpo.lock.json',archive/'grpo.lock.json')
        evidence = []
        for split,rows in accepted.items():
            for row in rows:
                details = audit[row['source_id']]
                row['tests'] = list(dict.fromkeys(row['tests'] + details['example_tests']))
                evidence.append({**provenance[row['source_id']], 'quality':details})
            (folder/f'{split}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
        (folder/'provenance.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in evidence))
        report['finalization'] = {'counts':{split:len(rows) for split,rows in accepted.items()},
                                  'entry_checks':dict(entry_counts),'exclusions':entry_exclusions,
                                  'entry_corrections':entry_corrections,'semantic_exclusions':semantic['exclusions'],
                                  'finalizer_sha256':sha256(Path(__file__))}
        # 执行审查与入口修正可能由两个代码版本完成，分别保留真实代码哈希。
        execution_script = Path(__file__) if sha256(Path(__file__)) == report['script_sha256'] else audit_path.parent/'quality-script-v1.py'
        if sha256(execution_script) != report['script_sha256']:
            raise ValueError('执行审查脚本快照与原记录不一致')
        report['execution_script_file'] = str(execution_script.relative_to(ROOT))
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        lock.update(profile='grpo-execution-audited-v1', counts=report['finalization']['counts'],
                    initial_lock_sha256=report['input_lock_sha256'], quality_sha256=sha256(report_path),
                    quality_audit_sha256=sha256(audit_path), quality_script_sha256=report['script_sha256'],
                    quality_script_file=report['execution_script_file'],
                    semantic_review_sha256=sha256(semantic_path),
                    finalizer_sha256=sha256(Path(__file__)), entry_checks=dict(entry_counts),
                    sft_tokenizer_sha256=report['sft_tokenizer_sha256'],
                    train_sha256=sha256(train_file), valid_sha256=sha256(valid_file),
                    provenance_sha256=sha256(folder/'provenance.jsonl'))
        (ROOT/'data/grpo.lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
        print('GRPO 补审后冻结',lock['counts'],flush=True)
        return

    if report_path.exists() or audit_path.exists():
        raise FileExistsError('审查记录已存在，拒绝覆盖')
    audit_path.parent.mkdir(parents=True,exist_ok=True)
    tokenizers = [AutoTokenizer.from_pretrained(path,local_files_only=True)
                  for path in (ROOT/'models/base',ROOT/'runs/lora_sft/lora-v1/final_model')]
    if tokenizers[0].get_vocab() != tokenizers[1].get_vocab() or tokenizers[0].special_tokens_map != tokenizers[1].special_tokens_map:
        raise ValueError('Base/SFT 词表或特殊 token 不同')
    image = subprocess.check_output(['docker','image','inspect','qwen-code-eval:0.3.1','--format','{{.Id}}'],text=True).strip()
    counts = Counter()
    flattened = [(split,row) for split,rows in partitions.items() for row in rows]
    with audit_path.open('w') as stream:
        # 同一容器执行一批，减少启动成本；每份代码仍由 worker 独立子进程运行。
        for start in range(0,len(flattened),20):
            planned, codes, cases = [], [], []
            for split,row in flattened[start:start+20]:
                reference = provenance[row['source_id']]['reference']
                for text in (row['prompt'],reference):
                    ids = [t.encode(text,add_special_tokens=False) for t in tokenizers]
                    if ids[0] != ids[1] or tokenizers[0].decode(ids[0]) != tokenizers[1].decode(ids[1]):
                        raise ValueError(f"分词或解码变化：{row['source_id']}")
                examples, matched = expanded_examples(row['prompt'],reference)
                tests = list(dict.fromkeys(row['tests']+examples))
                seed = int(hashlib.sha256(row['source_id'].encode()).hexdigest()[:8],16)
                variants = mutations(reference,seed,3)
                batch = [reference,zero_probe(reference)] + [code for code,_ in variants]
                calls = {node.name for node in ast.parse(reference).body if isinstance(node,ast.FunctionDef)}
                tests_call_function = all(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                                             and node.func.id in calls for node in ast.walk(ast.parse(test))) for test in tests)
                planned.append((split,row,examples,matched,variants,len(batch),tests_call_function))
                codes.extend(batch); cases.extend([tests]*len(batch))
            scores = score_batch(codes,cases)
            offset = 0
            for split,row,examples,matched,variants,size,tests_call_function in planned:
                reference_score,constant_score,*mutant_scores = scores[offset:offset+size]; offset += size
                reasons = []
                if reference_score != 1.0: reasons.append('reference_or_example_failed')
                if constant_score == 1.0: reasons.append('constant_output_passes_all')
                if not tests_call_function: reasons.append('assertion_does_not_call_function')
                if len(mutant_scores) >= 2 and all(score == 1.0 for score in mutant_scores): reasons.append('all_checked_mutants_survive')
                record = {'split':split,'source_id':row['source_id'],'accepted':not reasons,'reasons':reasons,
                          'reference_score':reference_score,'constant_zero_score':constant_score,
                          'mutations':[{'change':change,'score':score} for (_,change),score in zip(variants,mutant_scores,strict=True)],
                          'matched_examples':matched,'parsed_examples':len(examples),'example_tests':examples}
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                counts[f'{split}_checked'] += 1
                counts[f'{split}_accepted'] += not reasons
                counts['matched_examples'] += matched; counts['parsed_examples'] += len(examples)
                counts['mutants_executed'] += len(mutant_scores)
                counts['mutants_failed_tests'] += sum(score<1 for score in mutant_scores)
                counts['questions_with_no_mutation'] += not mutant_scores
                for reason in reasons: counts[reason] += 1
            stream.flush(); print(dict(counts),flush=True)
    if image != subprocess.check_output(['docker','image','inspect','qwen-code-eval:0.3.1','--format','{{.Id}}'],text=True).strip():
        raise ValueError('审查时镜像变化')
    report = {'profile':'grpo-quality-audit-v1','input_lock_sha256':sha256(ROOT/'data/grpo.lock.json'),
              'audit_sha256':sha256(audit_path),'script_sha256':sha256(Path(__file__)),
              'sft_tokenizer_sha256':sha256(ROOT/'runs/lora_sft/lora-v1/final_model/tokenizer.json'),
              'docker_image_id':image,'counts':dict(counts),
              'limitations':['测试通过不证明所有输入正确；无法解析的示例未验证。',
                             '未通过变异不一定表示错误；保守排除两个以上变异均全过的题会排除一些等价变异。',
                             '验证题来自原 SFT valid；只用于本阶段模型选择，不能冒充独立最终测试。']}
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
