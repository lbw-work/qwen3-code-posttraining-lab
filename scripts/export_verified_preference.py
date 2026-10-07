"""用现有 SFT 函数答案和 RL 训练测试，构造经 Docker 验证的单处错误偏好。"""

import argparse
import ast
import hashlib
import io
import json
import random
import re
import subprocess
import tokenize
from collections import Counter

from transformers import AutoTokenizer

from code_reward import score_batch
from qwen_posttrain.artifacts import ROOT, sha256, verified_tasks
from prepare_preference import function_pair_reason, overlaps_eval_function
from prepare_sft import eval_ngrams, overlaps_eval, words
from train_common import locked_jsonl


CHANGES = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "==",
           "+": "-", "-": "+", "*": "+", "and": "or", "or": "and",
           "True": "False", "False": "True", "0": "1", "1": "0"}
MIN_REJECTED_SCORE = 0.5


def mutations(code: str, seed: int, count: int) -> list[tuple[str, dict]]:
    """返回最多 count 个候选；每个只替换源代码中的一个运算符/常量 token。

    tokenize 能区分代码、注释和字符串，所以不会把 docstring 中的“0”也改掉。
    使用原始行列坐标拼接代码，保留缩进、注释与其余格式，避免正负标签同时
    与“代码被整体重新格式化”相关。AST 只用于检查语法，不执行第三方代码。
    为减少无意义的超时负例，跳过整个 while 区域及函数定义首行；其余候选
    是否真的改变功能，要由后面的 Docker 测试判断，不能只凭变异存在就下结论。
    """
    tree = ast.parse(code)
    skipped = [(n.lineno, n.end_lineno) for n in ast.walk(tree) if isinstance(n, ast.While)]
    headers = {n.lineno for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    lines = code.splitlines(keepends=True)
    candidates = []
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        old = token.string
        if token.type not in (tokenize.OP, tokenize.NAME, tokenize.NUMBER) or old not in CHANGES:
            continue
        row, col = token.start
        if row in headers or any(start <= row <= end for start, end in skipped):
            continue
        new = CHANGES[old]
        changed = lines.copy()
        changed[row - 1] = changed[row - 1][:col] + new + changed[row - 1][token.end[1]:]
        candidate = "".join(changed)
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        candidates.append((candidate, {"line": row, "column": col, "old": old, "new": new}))
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def main() -> None:
    """筛选题目、执行正负答案、保留部分测试失败的负例并锁定导出。

    只读取已有 SFT/RL 的 train，不取它们的 valid。每题先产生少量单处变异，
    每批 20 题一起送进项目已有的 Docker 工具，降低容器启动开销。
    chosen 必须通过全部原有测试；rejected 必须通过至少一半、但并非全部测试。
    如果多个负例合格，选通过比例最高的一份，使训练信号偏向边界/逻辑差异。
    tests 留在导出证据中，最终训练 JSONL 仍只包含 prompt/chosen/rejected。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=2000, help="最多保留多少个不同题目的偏好对")
    args = parser.parse_args()
    if args.limit < 1 or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        parser.error("limit 必须为正数，run-id 只能包含字母、数字、下划线和连字符")
    sft_path, _, sft_lock = locked_jsonl(ROOT / "data/sft", "sft")
    rl_path, _, rl_lock = locked_jsonl(ROOT / "data/rl", "rl")
    # image ID 固定后执行：导出期间若标签重新指向别的镜像，必须拒绝冻结。
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "qwen-code-eval:0.3.1", "--format", "{{.Id}}"], text=True,
    ).strip()
    output = ROOT / "data/raw/verified-preference" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / "pairs.partial.jsonl"
    final = output / "pairs.jsonl"
    tests_by_id = {r["source_id"]: r["tests"] for r in map(json.loads, rl_path.read_text().splitlines())}
    rows = list(map(json.loads, sft_path.read_text().splitlines()))
    random.Random(42).shuffle(rows)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/base", local_files_only=True)
    tasks = verified_tasks()
    spans = eval_ngrams(tasks)
    names = {re.sub(r"[^a-z0-9]", "", t["entry_point"].casefold()) for t in tasks}
    counts = Counter()
    seen = set()
    pending = []

    def verify_batch(stream) -> None:
        """执行 pending 中的题，按返回分数与候选索引一一对应写入合格记录。

        每题的第 0 个候选就是 chosen，后面是变异代码。index 随题目推进，
        防止不同题目的分数错配。只有整批执行成功才写入，容器错误会直接报错。
        """
        codes, tests = [], []
        for row, cases, candidates in pending:
            codes.extend([row["completion"]] + [code for code, _ in candidates])
            tests.extend([cases] * (len(candidates) + 1))
        scores = score_batch(codes, tests)
        index = 0
        for row, cases, candidates in pending:
            positive = scores[index]
            negative_scores = scores[index + 1:index + len(candidates) + 1]
            index += len(candidates) + 1
            counts["executed_questions"] += 1
            counts["executed_candidates"] += len(candidates) + 1
            if positive != 1.0:
                counts["chosen_failed_local_tests"] += 1
                continue
            eligible = [(score, i) for i, score in enumerate(negative_scores) if MIN_REJECTED_SCORE <= score < 1]
            if not eligible:
                counts["no_partial_failure_negative"] += 1
                continue
            if counts["accepted"] >= args.limit:
                continue
            # 同分时取候选列表里最靠前的一份，固定随机种子使选择可复现。
            score, best = max(eligible, key=lambda pair: (pair[0], -pair[1]))
            rejected, mutation = candidates[best]
            record = {"language": "Python", "aspect": "Functional Correctness",
                      "source": "OPENCODEINSTRUCT_EXECUTION", "source_id": row["source_id"],
                      "input": row["prompt"].removeprefix('"""\n').removesuffix('\n"""\n\n'),
                      "chosen": row["completion"], "rejected": rejected,
                      "tests": cases, "chosen_score": positive, "rejected_score": score,
                      "mutation": mutation}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["accepted"] += 1
        stream.flush()
        pending.clear()
        print(json.dumps(dict(counts), ensure_ascii=False), flush=True)

    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            if counts["accepted"] >= args.limit:
                break
            counts["scanned_sft_train"] += 1
            cases = tests_by_id.get(row["source_id"], [])
            if len(cases) < 5 or len(set(cases)) < 5:
                counts["insufficient_tests"] += 1
                continue
            key = " ".join(words(row["prompt"]))
            if key in seen:
                continue
            seen.add(key)
            if overlaps_eval(row["prompt"], spans) or overlaps_eval_function(row["completion"], row["completion"], names):
                counts["eval_overlap"] += 1
                continue
            # 原 SFT 题干常有 fenced 输入输出样例；只检查代码是否为独立函数，
            # 保留题干原文。工程题仍由移除示例块后的自然语言部分检查。
            question = row["prompt"].removeprefix('"""\n').removesuffix('\n"""\n\n')
            question = re.sub(r"```[\s\S]*?```", "", question)
            question = re.sub(r"sample input|input format", "", question, flags=re.I)
            seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
            candidates = []
            prompt_tokens = len(tokenizer.encode(row["prompt"], add_special_tokens=False))
            for code, change in mutations(row["completion"], seed, 6):
                if function_pair_reason(question, row["completion"], code):
                    continue
                length = len(tokenizer.encode(code, add_special_tokens=False)) + 1
                if length > 512 or prompt_tokens + length > 1024:
                    continue
                candidates.append((code, change))
            if not candidates:
                counts["no_usable_mutation"] += 1
                continue
            pending.append((row, cases, candidates))
            if len(pending) == 20:
                verify_batch(stream)
        if pending:
            verify_batch(stream)
    if not counts["accepted"]:
        raise ValueError("没有得到执行验证偏好，保留 partial 文件供排查")
    after = subprocess.check_output(["docker", "image", "inspect", "qwen-code-eval:0.3.1", "--format", "{{.Id}}"], text=True).strip()
    if after != image_id:
        raise ValueError("Docker 镜像在导出期间发生变化，不冻结输出")
    temporary.rename(final)
    meta = {"profile": "execution-verified-functions-v1", "counts": dict(counts), "seed": 42,
            "repo": sft_lock["repo"], "revision": sft_lock["revision"],
            "source_file": sft_lock["source_file"], "source_sha256": sft_lock["source_sha256"],
            "sft_train_sha256": sft_lock["train_sha256"], "rl_train_sha256": rl_lock["train_sha256"],
            "eval_sha256": sft_lock["eval_sha256"], "docker_image_id": image_id,
            "export_sha256": sha256(final), "builder_sha256": sha256(ROOT / "scripts/export_verified_preference.py"),
            "worker_sha256": sha256(ROOT / "scripts/code_reward_worker.py"),
            "min_rejected_score": MIN_REJECTED_SCORE,
            "mutations_per_question": 6, "min_distinct_tests": 5,
            "verification": "chosen passes all supplied training tests; rejected passes some and fails some"}
    final.with_suffix(".lock.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"导出完成：{final}", flush=True)


if __name__ == "__main__":
    main()
