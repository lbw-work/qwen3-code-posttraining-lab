"""将已导出的 Python 正确性偏好 JSONL 变成 DPO/RM 共用的冻结格式。"""

import argparse
import ast
import builtins
import hashlib
import json
import re
import symtable
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from qwen_posttrain.artifacts import ROOT, sha256, verified_tasks
from prepare_sft import CODE_BLOCK, eval_ngrams, overlaps_eval, words


ALGORITHM_MODULES = {"math", "typing", "collections", "itertools", "functools", "heapq", "bisect", "re", "string", "statistics", "fractions", "decimal", "operator", "__future__"}
MAX_VERIFIED_PER_FUNCTION = 10


def function_pair_reason(question: str, chosen: str, rejected: str) -> str | None:
    """检查一对代码是否适合本项目的“纯 Python 算法函数”训练。

    返回 None 表示通过；否则返回可统计的拒绝原因。这是格式/任务匹配检查，
    **不证明 chosen 正确**。两段代码已由 python_code 保证可解析。
    为贴近函数调用评测，只接受函数、常量和少量算法标准库，排除读 stdin、
    打印答案、文件操作和依赖外部框架的程序。保守过滤可能舍弃可用样本，
    但不会擅自删改程序，把上游测试过的答案变成另一份代码。
    """
    if "```" in question or any(marker in question.casefold() for marker in ("standard input", "stdin", "-----input-----", "input format", "sample input")):
        return "program_or_embedded_code_prompt"
    if any(marker in question.casefold() for marker in (
        "database", "authentication", "authorization", "user_id", "sender", "recipient", "optimization_level",
        "discount", "product_name", "list of products", "inventory", "invoice", "purchase",
        "customer", "employee", "phone number", "dscp", "fantasy gaming",
        "pybids", "tensorflow", "pytorch", "numpy", "pandas", "django", "flask", "given a python class",
    )):
        return "engineering_prompt"
    trees = [ast.parse(code) for code in (chosen, rejected)]
    signatures = []
    for tree in trees:
        if any(isinstance(n, ast.FunctionDef) and any(a.arg in {"self", "cls"} for a in n.args.posonlyargs + n.args.args) for n in tree.body):
            return "method_needs_class_context"
        if any(not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)) and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)) for node in tree.body):
            return "not_function_module"
        signatures.append({(n.name, ast.dump(n.args, include_attributes=False)) for n in tree.body if isinstance(n, ast.FunctionDef)})
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [a.name.split('.')[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [(node.module or '').split('.')[0]]
                if node.level:
                    return "external_dependency"
            else:
                modules = []
            if any(module not in ALGORITHM_MODULES for module in modules):
                return "external_dependency"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"input", "print", "open", "eval", "exec", "compile", "__import__"}:
                return "io_or_dynamic_execution"
            if isinstance(node, ast.Pass) or (isinstance(node, ast.Constant) and node.value is Ellipsis):
                return "placeholder"
            if isinstance(node, ast.Attribute) and node.attr in {"astype", "reshape", "to_numpy", "iloc", "cuda", "detach", "backward"}:
                return "array_or_tensor_dependency"
        # AST 本来就忽略注释；再移除函数/模块 docstring，防止只有文档变化的
        # 两份代码被误认为功能正确性的正负样本。仅用于比较，不改输出代码。
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
    if not signatures[0].intersection(signatures[1]):
        return "different_function_interface"
    if ast.dump(trees[0], include_attributes=False) == ast.dump(trees[1], include_attributes=False):
        return "same_code_ast"
    # AST 能通过不代表代码自包含。例如函数使用 expected，却没有参数或全局
    # 定义，运行时仍会 NameError。symtable 理解函数/闭包/推导式的作用域，
    # 比简单收集所有变量名可靠；这里仅排除明确缺失的全局名字。
    for code in (chosen, rejected):
        try:
            table = symtable.symtable(code, "<preference>", "exec")
        except SyntaxError:
            return "invalid_scope"
        defined = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()} | set(dir(builtins))
        pending = [table]
        while pending:
            scope = pending.pop()
            if any(s.is_referenced() and s.is_global() and s.get_name() not in defined for s in scope.get_symbols()):
                return "undefined_global"
            pending.extend(scope.get_children())
    return None


def overlaps_eval_function(chosen: str, rejected: str, names: set[str]) -> bool:
    """补充题干 n-gram 检查：拒绝与评测入口同名的函数，包括命名风格变体。

    例如 hexKey 与 hex_key 会归一化为同一个名字。抽查发现上游题干可被
    大幅改写，仅靠连续 12 词会漏掉这种情况。此规则故意保守，可能排除
    同名但不同的算法题；它仍不能识别所有改名且改写题干的语义重复题。
    """
    return any(re.sub(r"[^a-z0-9]", "", node.name.casefold()) in names
               for code in (chosen, rejected) for node in ast.walk(ast.parse(code))
               if isinstance(node, ast.FunctionDef))


def python_code(text: str) -> str | None:
    """规范化一份偏好答案；无法作为 Python 函数答案时返回 None。

    输入可以是裸代码或完整的一个 `````python`` 代码块。先移除 UTF-8 BOM 和首尾空白，
    再解开代码块；AST 解析成功且模块顶层至少有一个 ``def`` 才接受。这样 DPO/RM 看到
    的 chosen、rejected 都是“可作为函数题答案的代码”，不会把解释性自然语言误当目标。
    返回时统一补一个换行，令 token 化和文件哈希稳定。
    """
    text = text.lstrip("\ufeff").strip()
    block = CODE_BLOCK.fullmatch(text)
    if block:
        text = block.group(1)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    if not any(isinstance(node, ast.FunctionDef) for node in tree.body):
        return None
    return text.rstrip() + "\n"


def main() -> None:
    """把上游偏好 JSONL 冻结为 DPO 与奖励模型共用的数据集。

    每行只允许 Python + Functional Correctness/FC；随后规范 chosen/rejected、排除相同
    答案、检测评测题重合、按题干去重、检查两支答案都不超过 1024 token。保留的数据用
    同一题干哈希作稳定 95/5 train/valid 划分。最后的锁记录来源名字和全部摘要，因此
    DPO 与 Reward Model 不会悄悄使用不同版本的偏好对。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="每行含 input/prompt、chosen、rejected 的 JSONL")
    parser.add_argument("--source-name", required=True, help="数据集名字和固定版本，写入锁文件")
    parser.add_argument("--verified-functions", action="store_true", help="冻结 export_verified_preference.py 的执行验证函数对")
    args = parser.parse_args()
    source = args.source.resolve()
    verified_meta = None
    review_path = ROOT / "data/preference-review.json"
    example_audit_path = ROOT / "data/preference-example-audit.json"
    reviewed_exclusions = {}
    example_exclusions = set()
    if args.verified_functions:
        verified_meta = json.loads(source.with_suffix(".lock.json").read_text())
        if verified_meta["profile"] != "execution-verified-functions-v1" or verified_meta["export_sha256"] != sha256(source):
            raise ValueError("执行验证导出的类型或哈希不匹配")
        if verified_meta["worker_sha256"] != sha256(ROOT / "scripts/code_reward_worker.py"):
            raise ValueError("执行验证 worker 已改变，需要重新核对测试结果")
        # 原测试全过仍可能遗漏题干约束。人工确认不适合的原始 ID 单独留档，
        # 不改正确答案或测试；锁定清单哈希，让后续构建保留相同排除决策。
        review = json.loads(review_path.read_text())
        if review["source_revision"] != verified_meta["revision"] or review["profile"] != verified_meta["profile"]:
            raise ValueError("人工审查清单与执行来源不一致")
        reviewed_exclusions = review["excluded_source_ids"]
        if example_audit_path.exists():
            audit = json.loads(example_audit_path.read_text())
            if (audit["source_sha256"] != sha256(source)
                    or audit["worker_sha256"] != sha256(ROOT / "scripts/code_reward_worker.py")
                    or audit["audit_script_sha256"] != sha256(ROOT / "scripts/audit_preference_examples.py")):
                raise ValueError("题干示例审查与来源、worker 或审查脚本版本不一致")
            example_exclusions = set(audit["excluded_source_ids"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / "base", local_files_only=True)
    tasks = verified_tasks()
    ngrams = eval_ngrams(tasks)
    eval_names = {re.sub(r"[^a-z0-9]", "", task["entry_point"].casefold()) for task in tasks}
    output = ROOT / "data" / "preference"
    lock_path = ROOT / "data" / "preference.lock.json"
    if output.exists() or lock_path.exists():
        raise FileExistsError("偏好数据已冻结，不自动覆盖")
    output.mkdir(parents=True)
    counts = Counter()
    seen = set()
    seen_chosen = set()
    function_counts = Counter()
    provenance = []
    sources = Counter()
    # 不把 SFT/RL 验证题重新用于偏好训练。已有 SFT 训练题允许复用，但验证
    # 题必须继续保持独立；这里只检测归一化后的相同题干，不能保证语义去重。
    heldout = set()
    heldout_hashes = {}
    for file in (ROOT / "data" / "sft" / "valid.jsonl", ROOT / "data" / "rl" / "valid.jsonl"):
        if file.exists():
            heldout_hashes[str(file.relative_to(ROOT))] = sha256(file)
            heldout.update(" ".join(words(json.loads(line)["prompt"])) for line in file.read_text().splitlines())
    with source.open(encoding="utf-8") as raw, (output / "train.jsonl").open("w", encoding="utf-8") as train, (output / "valid.jsonl").open("w", encoding="utf-8") as valid:
        for line in raw:
            counts["source_rows"] += 1
            item = json.loads(line)
            if args.verified_functions and item.get("source_id") in reviewed_exclusions:
                counts["manual_review_excluded"] += 1
                continue
            if args.verified_functions and item.get("source_id") in example_exclusions:
                counts["prompt_example_failed"] += 1
                continue
            if args.verified_functions and item.get("source") != "OPENCODEINSTRUCT_EXECUTION":
                counts["non_algorithm_source"] += 1
                continue
            # 上游导出时必须给出语言和偏好维度；防止把其它语言、代码风格
            # 或安全性偏好混入“Python 功能正确性”实验。
            if item.get("language", "").casefold() != "python" or item.get("aspect", "").casefold() not in ("functional correctness", "fc"):
                counts["wrong_subset"] += 1
                continue
            question = (item.get("input") or item.get("prompt") or "").strip()
            chosen = python_code(item.get("chosen") or "")
            rejected = python_code(item.get("rejected") or "")
            if not question or '"""' in question or chosen is None or rejected is None or chosen == rejected:
                counts["format_rejected"] += 1
                continue
            if args.verified_functions and (item.get("chosen_score") != 1.0 or not 0.5 <= item.get("rejected_score", 1.0) < 1 or len(item.get("tests", [])) < 5):
                counts["invalid_execution_evidence"] += 1
                continue
            if args.verified_functions:
                check_question = question
                if args.verified_functions:
                    # 本项目原 SFT 题干的代码围栏是输入/输出样例。只在任务类别
                    # 检查时略去它们；写入训练集的题干保持原文，测试证据另存。
                    check_question = re.sub(r"```[\s\S]*?```", "", question)
                    check_question = re.sub(r"sample input|input format", "", check_question, flags=re.I)
                reason = function_pair_reason(check_question, chosen, rejected)
                if reason:
                    counts[reason] += 1
                    continue
            if overlaps_eval(question, ngrams):
                counts["eval_overlap_rejected"] += 1
                continue
            if args.verified_functions and overlaps_eval_function(chosen, rejected, eval_names):
                counts["eval_function_name_overlap"] += 1
                continue
            normalized = " ".join(words(question))
            if normalized in heldout:
                counts["previous_validation_overlap"] += 1
                continue
            key = hashlib.sha256(normalized.encode()).hexdigest()
            if key in seen:
                counts["duplicate_prompt"] += 1
                continue
            prompt = f'"""\n{question}\n"""\n\n'
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            answer_lengths = [len(tokenizer.encode(answer, add_special_tokens=False)) + 1 for answer in (chosen, rejected)]
            if prompt_len + max(answer_lengths) > 1024 or (args.verified_functions and max(answer_lengths) > 512):
                counts["too_long"] += 1
                continue
            chosen_tree = ast.parse(chosen)
            if args.verified_functions:
                # 两份相同实现可能只有 docstring 不同。去重时略去文档，防止
                # 这样的实现分别落入 train/valid；实际写入的代码仍保持原样。
                for node in ast.walk(chosen_tree):
                    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                        node.body = node.body[1:]
            chosen_key = hashlib.sha256(ast.dump(chosen_tree, include_attributes=False).encode()).hexdigest()
            if chosen_key in seen_chosen:
                counts["duplicate_chosen_code"] += 1
                continue
            if args.verified_functions:
                # 上游大量题只是同一函数的近似改写，如 factorial、binary_search。
                # 以第一个顶层函数名作保守的多样性代理，最多留 10 对；不同命名
                # 仍可能描述同一算法，所以它不能代替语义去重。
                function_name = next(node.name for node in chosen_tree.body if isinstance(node, ast.FunctionDef))
                function_key = re.sub(r"[^a-z0-9]", "", function_name.casefold())
                if function_counts[function_key] >= MAX_VERIFIED_PER_FUNCTION:
                    counts["function_name_cap"] += 1
                    continue
                function_counts[function_key] += 1
            seen.add(key)
            seen_chosen.add(chosen_key)
            row = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            # 执行验证来源只使用 SFT train。如果复用 SFT 原来的划分哈希，
            # 所有题都会再次进入 train，得不到 DPO valid。用独立盐做偏好划分，
            # 保证 DPO/RM train 与 valid 按题分开；SFT/RL 原 valid 仍完全排除。
            split_key = hashlib.sha256(("preference-v1:" + normalized).encode()).hexdigest() if args.verified_functions else key
            stream = valid if int(split_key[:8], 16) % 20 == 0 else train
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["valid" if stream is valid else "train"] += 1
            sources[item.get("source", "unknown")] += 1
            provenance.append({"prompt_sha256": key, "chosen_ast_sha256": chosen_key, "split": "valid" if stream is valid else "train", "source": item.get("source"), "source_id": item.get("source_id"), "prompt_tokens": prompt_len, "chosen_tokens": answer_lengths[0], "rejected_tokens": answer_lengths[1]})
            if args.verified_functions:
                provenance[-1].update({field: item[field] for field in ("tests", "chosen_score", "rejected_score", "mutation")})
    if not counts["train"] or not counts["valid"]:
        raise ValueError("偏好数据没有得到非空的训练/验证划分；请检查字段与过滤结果")
    lock = {
        "source_name": args.source_name,
        "source_sha256": sha256(source),
        "eval_sha256": sha256(ROOT / "eval.jsonl"),
        "counts": dict(counts),
        "train_sha256": sha256(output / "train.jsonl"),
        "valid_sha256": sha256(output / "valid.jsonl"),
        "profile": "execution-verified-functions-v1" if args.verified_functions else "python-fc",
        "source_counts": dict(sources),
        "label_validation": "Docker verified supplied training tests; not proof of full correctness" if args.verified_functions else "upstream labels only; no local execution verification",
        "prepare_script_sha256": sha256(Path(__file__)),
        "previous_validation_sha256": heldout_hashes,
        "tokenizer_source": json.loads((ROOT / "models.lock.json").read_text())["base"],
        "max_sequence_tokens": 1024,
        "max_answer_tokens": 512 if args.verified_functions else None,
        "max_pairs_per_function_name": MAX_VERIFIED_PER_FUNCTION if args.verified_functions else None,
        "split_salt": "preference-v1:" if args.verified_functions else "",
    }
    if verified_meta:
        lock["execution_export"] = verified_meta
        lock["execution_export_lock_sha256"] = sha256(source.with_suffix(".lock.json"))
        lock["manual_review_sha256"] = sha256(review_path)
        if example_audit_path.exists():
            lock["example_audit_sha256"] = sha256(example_audit_path)
    (output / "provenance.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in provenance), encoding="utf-8")
    lock["provenance_sha256"] = sha256(output / "provenance.jsonl")
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
