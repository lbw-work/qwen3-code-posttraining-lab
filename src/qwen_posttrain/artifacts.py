"""仅依赖标准库的项目路径、文件哈希与冻结评测读取。

供数据处理、训练和评测共同调用，避免为读取 JSON 而导入模型和 GPU 依赖。
本项目采用 editable 安装：该模块始终位于 <项目>/src/qwen_posttrain/。
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    """计算文件内容的 SHA-256。

    参数是一个实际文件路径，返回 64 位十六进制摘要。这里按 1 MiB 分块读取，
    因此即使以后评测数据很大，也不会把整份文件一次装进内存。它不是业务逻辑，
    而是整个实验可比性的门卫：同名文件只要内容变了，摘要就会变。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_tasks() -> list[dict]:
    """读取并验证本次要生成的全部评测题。

    返回的每个字典至少有 ``suite``、``task_id``、``prompt``、``entry_point``。
    调用顺序是：先校验总清单 ``eval.jsonl``，再校验两份 EvalPlus 原始题库，
    最后才解析题目。任何一步不一致都抛异常，而不是带着被改过的题继续跑，
    因为那会让不同模型的 pass@1 无法比较。
    """
    lock = json.loads((ROOT / "eval.lock.json").read_text(encoding="utf-8"))
    manifest = ROOT / "eval.jsonl"
    if sha256(manifest) != lock["eval_jsonl_sha256"]:
        raise ValueError("eval.jsonl 与冻结时的哈希不一致")
    for dataset in lock["datasets"].values():
        if sha256(ROOT / dataset["file"]) != dataset["sha256"]:
            raise ValueError(f"评测题库被修改：{dataset['file']}")
    with manifest.open(encoding="utf-8") as stream:
        tasks = [json.loads(line) for line in stream]
    if len(tasks) != sum(item["count"] for item in lock["datasets"].values()):
        raise ValueError("评测题数与锁文件不一致")
    return tasks
