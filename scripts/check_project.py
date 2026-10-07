"""只读检查发布输入与六阶段结果；不下载模型、不训练、不执行生成代码。"""

import json
import re
import xml.etree.ElementTree as ET
from urllib.parse import unquote

from qwen_posttrain.artifacts import ROOT, sha256, verified_tasks


def check_data(eval_hash: str) -> None:
    """按锁文件核对五类当前数据的内容、数量以及附带审核证据。

    不重写锁文件，也不把当前构建脚本的哈希写回旧实验。builder_sha256
    记录当时构建数据的代码；worker 和质量证据则仍须匹配冻结版本。
    """
    for name in ("sft", "rl", "preference", "ppo", "grpo"):
        lock = json.loads((ROOT / "data" / f"{name}.lock.json").read_text())
        if lock["eval_sha256"] != eval_hash:
            raise ValueError(f"{name} 使用了不同的评测版本")
        for split in ("train", "valid"):
            path = ROOT / "data" / name / f"{split}.jsonl"
            if sha256(path) != lock[f"{split}_sha256"]:
                raise ValueError(f"数据哈希不匹配：{path}")
            # 逐行解析以拒绝坏 JSON；不将大训练集整体留在内存中。
            with path.open() as stream:
                count = sum(1 for line in stream if isinstance(json.loads(line), dict))
            if count != lock["counts"][split]:
                raise ValueError(f"{name}/{split} 数量不匹配")
        for key, path in (
            ("provenance_sha256", ROOT / "data" / name / "provenance.jsonl"),
            ("audit_sha256", ROOT / "data" / name / "audit.jsonl"),
            ("quality_sha256", ROOT / "data" / f"{name}-quality.json"),
            ("semantic_review_sha256", ROOT / "data" / f"{name}-semantic-review.json"),
            ("worker_sha256", ROOT / "scripts/code_reward_worker.py"),
            ("quality_audit_sha256", ROOT / "data/raw/grpo-preparation/quality-audit.jsonl"),
            ("quality_script_sha256", ROOT / lock.get("quality_script_file", "")),
        ):
            if key in lock and sha256(path) != lock[key]:
                raise ValueError(f"{name} 审核证据不匹配：{key}")


def check_results(tasks: list[dict], eval_hash: str) -> None:
    """从逐题记录复核已有成绩，避免只检查总分而漏掉缺题或重复生成。

    每份样本和判分都必须恰好覆盖同一套题；Plus 成绩同时要求 base 与
    plus 为 pass。这里仅读取结果，不能运行候选代码或覆盖 summary。
    """
    expected = {suite: {t["task_id"] for t in tasks if t["suite"] == suite}
                for suite in ("humaneval", "mbpp")}
    stages = set()
    for folder in sorted((ROOT / "results").glob("*/*")):
        if folder.parent.name in ("smoke", "analysis", "comparisons") or not (folder / "summary.json").exists():
            continue
        summary = json.loads((folder / "summary.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        if summary["stage"] != config["stage"] or summary["eval_sha256"] != eval_hash or config["eval_sha256"] != eval_hash:
            raise ValueError(f"评测版本或阶段不一致：{folder}")
        if config["limit_per_suite"] is not None or config["generation"] != {"do_sample": False, "max_new_tokens": 512}:
            raise ValueError(f"不是固定协议的完整评测：{folder}")
        for suite, task_ids in expected.items():
            with (folder / f"{suite}.samples.jsonl").open() as stream:
                sample_ids = [json.loads(line)["task_id"] for line in stream]
            result_path = folder / f"{suite}.samples_eval_results.json"
            judged = json.loads(result_path.read_text())["eval"]
            if len(sample_ids) != len(task_ids) or set(sample_ids) != task_ids or set(judged) != task_ids:
                raise ValueError(f"生成或判分题目不完整：{folder}/{suite}")
            if any(len(attempts) != 1 for attempts in judged.values()):
                raise ValueError(f"不是每题一次生成：{folder}/{suite}")
            passed = sum(a[0]["base_status"] == a[0]["plus_status"] == "pass" for a in judged.values())
            metric = summary["suites"][suite]
            if metric["passed"] != passed or metric["total"] != len(task_ids) or metric["pass_at_1"] != passed / len(task_ids):
                raise ValueError(f"摘要与逐题成绩不同：{folder}/{suite}")
            if metric["result_sha256"] != sha256(result_path):
                raise ValueError(f"判分文件已改变：{folder}/{suite}")
        stages.add(summary["stage"])
    if not {"base", "full_sft", "lora_sft", "dpo", "ppo", "grpo"}.issubset(stages):
        raise ValueError("发布结果缺少六阶段中的必要阶段")


def check_docs() -> None:
    """检查首页、文档和 SVG 的本地引用；网络链接不在此检查中请求。

    模型卡包含 Hugging Face 的 YAML 头和仓库专用引用，因此不按项目
    文档解析。目录内的历史记录可以保留过去状态，不据此改写旧实验。
    """
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = path.read_text()
        for pair in re.findall(r"\]\(([^)]+)\)|(?:src|href)=\"([^\"]+)\"", text):
            target = next(value for value in pair if value)
            if target.startswith(("https:", "http:", "mailto:")):
                continue
            filename, _, anchor = target.partition("#")
            destination = (path.parent / unquote(filename)).resolve() if filename else path
            if not destination.exists():
                raise ValueError(f"文档链接失效：{path.relative_to(ROOT)} → {target}")
            if anchor and destination.name == "README.md" and f'id="{anchor}"' not in destination.read_text():
                raise ValueError(f"首页锚点缺失：{target}")
    for path in (ROOT / "docs/assets").glob("*.svg"):
        ET.parse(path)
    tour = json.loads((ROOT / ".tours/architect-code-posttraining.tour").read_text())
    for step in tour["steps"]:
        if not 1 <= step["line"] <= len((ROOT / step["file"]).read_text().splitlines()):
            raise ValueError(f"源码导读位置失效：{step['file']}")


def main() -> None:
    """依次核对冻结题库、训练输入、正式结果和文档；失败直接返回非零退出码。"""
    tasks = verified_tasks()
    eval_hash = sha256(ROOT / "eval.jsonl")
    check_data(eval_hash)
    check_results(tasks, eval_hash)
    check_docs()
    print("项目检查通过：五类冻结数据、542 道题、六阶段逐题结果、文档与 SVG。")


if __name__ == "__main__":
    main()
