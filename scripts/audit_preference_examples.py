"""执行题干中可解析的示例，剔除示例与 chosen 不一致的偏好题。"""

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

from code_reward import score_batch
from qwen_posttrain.artifacts import ROOT, sha256


# 只处理明确用代码围栏给出一组 Sample Input/Output 的题。
# 其它写法保留为“未检查”，不能声称全数据均通过示例验证。
EXAMPLES = re.compile(
    r"\*\*Sample Input:\*\*\s*```(?:python)?\s*\n(.*?)\n```\s*"
    r"\*\*Sample Output:\*\*\s*```(?:python)?\s*\n(.*?)\n```",
    re.S | re.I,
)


def example_test(record: dict, sample_input: str, sample_output: str) -> str | None:
    """把一组简单题干示例转成断言；无法无歧义理解时返回 None。

    只用 ast.literal_eval 读取输入和预期输出，不在主机执行源数据代码。
    单个字面量用于单参数函数；多条赋值按函数参数名对应；两个空格分隔的
    整数等字面量用于多参数函数。若题干允许任意输出顺序，就跳过严格相等
    检查，避免把多解题误判为错误。真正运行断言仍交给受限 Docker。
    """
    if any(word in record["input"].casefold() for word in
           ("order does not matter", "any order", "order of the", "return any one")):
        return None
    functions = [node for node in ast.parse(record["chosen"]).body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return None
    function = next((node for node in functions if node.name in record["input"]), functions[0])
    params = [arg.arg for arg in function.args.posonlyargs + function.args.args]
    if not params or len(sample_output) > 1000:
        return None
    try:
        expected = ast.literal_eval(ast.parse(sample_output.strip(), mode="eval").body)
    except (SyntaxError, ValueError, TypeError, MemoryError):
        return None

    source = sample_input.strip()
    try:
        statements = ast.parse(source).body
        if len(statements) == 1 and isinstance(statements[0], ast.Expr) and len(params) == 1:
            arguments = [ast.literal_eval(statements[0].value)]
        elif statements and all(isinstance(node, ast.Assign) and len(node.targets) == 1
                                and isinstance(node.targets[0], ast.Name) for node in statements):
            values = {node.targets[0].id: ast.literal_eval(node.value) for node in statements}
            if not all(name in values for name in params):
                return None
            arguments = [values[name] for name in params]
        else:
            return None
    except (SyntaxError, ValueError, TypeError, MemoryError):
        # 例如题干把两个整数写成 "4 2"。仅在参数数目恰好匹配时拆开。
        pieces = source.split()
        if len(params) < 2 or len(pieces) != len(params):
            return None
        try:
            arguments = [ast.literal_eval(ast.parse(piece, mode="eval").body) for piece in pieces]
        except (SyntaxError, ValueError, TypeError):
            return None
    call = f'{function.name}({", ".join(map(repr, arguments))})'
    return f"assert {call} == {expected!r}"


def main() -> None:
    """复核原始偏好，保存可疑 ID 与锁；结果只用于保守排除。

    一题可有多组示例，只要其中一组执行不通过，就记录该题。失败也可能来自
    示例格式的歧义或运行资源限制，因此这是保守剔除，不是对来源错误率的
    精确估计。未能解析的示例不参与判断，并在统计里明确报告覆盖范围。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = ROOT / "data/preference-example-audit.json"
    if output.exists():
        raise FileExistsError("示例审查已冻结，不自动覆盖")
    export = json.loads(source.with_suffix(".lock.json").read_text())
    if export["export_sha256"] != sha256(source) or export["worker_sha256"] != sha256(ROOT / "scripts/code_reward_worker.py"):
        raise ValueError("原始导出或执行 worker 的哈希不匹配")
    image = subprocess.check_output(
        ["docker", "image", "inspect", "qwen-code-eval:0.3.1", "--format", "{{.Id}}"], text=True,
    ).strip()
    planned = []
    rows = [json.loads(line) for line in source.open()]
    for row in rows:
        for sample_input, sample_output in EXAMPLES.findall(row["input"]):
            test = example_test(row, sample_input, sample_output)
            if test:
                planned.append((row, test, sample_input, sample_output))
    print(f"原始题 {len(rows)}；可解析示例 {len(planned)}", flush=True)
    suspects = []
    for start in range(0, len(planned), 100):
        batch = planned[start:start + 100]
        scores = score_batch([row["chosen"] for row, *_ in batch], [[test] for _, test, *_ in batch])
        for (row, test, sample_input, sample_output), score in zip(batch, scores, strict=True):
            if score < 1:
                suspects.append({"source_id": row["source_id"], "sample_input": sample_input,
                                 "sample_output": sample_output, "test": test})
        print(f"已查 {start + len(batch)}；失败 {len(suspects)}", flush=True)
    after = subprocess.check_output(
        ["docker", "image", "inspect", "qwen-code-eval:0.3.1", "--format", "{{.Id}}"], text=True,
    ).strip()
    if after != image:
        raise ValueError("Docker 镜像在示例审查期间变化")
    suspects_path = source.parent / "example-audit-suspects.jsonl"
    suspects_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in suspects))
    report = {"profile": "prompt-example-execution-audit-v1", "source_sha256": sha256(source),
              "export_lock_sha256": sha256(source.with_suffix(".lock.json")),
              "audit_script_sha256": sha256(Path(__file__)),
              "worker_sha256": sha256(ROOT / "scripts/code_reward_worker.py"),
              "docker_image_id": image, "raw_rows": len(rows), "checked_examples": len(planned),
              "failed_examples": len(suspects), "excluded_source_ids": sorted({item["source_id"] for item in suspects}),
              "suspects_sha256": sha256(suspects_path),
              "interpretation": "Conservative exclusions where a parseable prompt example failed in Docker; unparsed examples were not checked."}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"示例审查完成：{len(report['excluded_source_ids'])} 道可疑题；{output}", flush=True)


if __name__ == "__main__":
    main()
