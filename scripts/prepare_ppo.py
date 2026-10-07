"""从冻结偏好集提取 PPO 题干，并与原 RL 来源逐条核对。"""

import json

from qwen_posttrain.artifacts import ROOT, sha256
from train_common import locked_jsonl


MAX_PROMPT_TOKENS = 512  # 与 train_ppo.py 当前的题干长度一致。


def main() -> None:
    """为 PPO 冻结 train/valid 题干、来源 ID 和版本锁。

    DPO 和奖励模型读取同一批偏好对，PPO 只需要这些题的 prompt。
    provenance 按偏好数据写入顺序排列，因此逐行对应 chosen/rejected 文件。
    核对 source_id 在原 RL train 中存在，且 prompt 完全相同，避免误把
    SFT/RL valid、不同版本的题干或评测题混进来。超出 PPO 512 token
    上限的题直接排除，训练时不会再悄悄截断。
    """
    preference_train, preference_valid, preference_lock = locked_jsonl(ROOT / "data/preference", "preference")
    rl_train, _, rl_lock = locked_jsonl(ROOT / "data/rl", "rl")
    output = ROOT / "data/ppo"
    lock_path = ROOT / "data/ppo.lock.json"
    if output.exists() or lock_path.exists():
        raise FileExistsError("PPO 数据已冻结，不自动覆盖")
    original = {row["source_id"]: row["prompt"] for row in map(json.loads, rl_train.open())}
    provenance = [json.loads(line) for line in (ROOT / "data/preference/provenance.jsonl").open()]
    if sha256(ROOT / "data/preference/provenance.jsonl") != preference_lock["provenance_sha256"]:
        raise ValueError("偏好来源记录的哈希不匹配")

    counts = {"train": 0, "valid": 0, "over_prompt_limit": {"train": 0, "valid": 0}}
    output.mkdir(parents=True)
    seen = set()
    for split, source in (("train", preference_train), ("valid", preference_valid)):
        details = (row for row in provenance if row["split"] == split)
        with source.open() as rows, (output / f"{split}.jsonl").open("w") as target:
            for row, meta in zip(map(json.loads, rows), details, strict=True):
                prompt = row["prompt"]
                # 通过 source_id -> 原始题干的逐字比较确认来源一致。
                if meta["source_id"] in seen or original.get(meta["source_id"]) != prompt:
                    raise ValueError("偏好题与 RL 训练题的 ID 或题干不匹配")
                seen.add(meta["source_id"])
                if meta["prompt_tokens"] > MAX_PROMPT_TOKENS:
                    counts["over_prompt_limit"][split] += 1
                    continue
                target.write(json.dumps({"source_id": meta["source_id"], "prompt": prompt}, ensure_ascii=False) + "\n")
                counts[split] += 1
    if not counts["train"] or not counts["valid"]:
        raise ValueError("PPO 数据划分为空")
    lock = {
        "profile": "ppo-function-prompts-v1",
        "preference_lock_sha256": sha256(ROOT / "data/preference.lock.json"),
        "rl_lock_sha256": sha256(ROOT / "data/rl.lock.json"),
        "preference_train_sha256": preference_lock["train_sha256"],
        "preference_valid_sha256": preference_lock["valid_sha256"],
        "rl_train_sha256": rl_lock["train_sha256"],
        "eval_sha256": preference_lock["eval_sha256"],
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "counts": counts,
        "train_sha256": sha256(output / "train.jsonl"),
        "valid_sha256": sha256(output / "valid.jsonl"),
        "script_sha256": sha256(ROOT / "scripts/prepare_ppo.py"),
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
