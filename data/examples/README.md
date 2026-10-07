# 数据格式样例

本目录只用于人工阅读数据格式。每个 `*.sample.jsonl` 恰有 10 行，每行都是来源
数据集的一条完整、未截断 JSON 记录。抽样种子、来源文件和 SHA-256 均记录在
[`samples.lock.json`](samples.lock.json)；重新执行下方脚本会得到同一批样例。

```bash
python scripts/sample_datasets.py
```

## 文件和阶段的关系

| 样例文件 | 记录字段 | 被哪些阶段使用 |
| --- | --- | --- |
| `evaluation.sample.jsonl` | `suite`、`task_id`、`prompt`、`entry_point` | Base 与全部训练后模型的统一评测 |
| `sft.sample.jsonl` | `source_id`、`prompt`、`completion`、两个 token 数 | Full SFT、LoRA SFT |
| `rl.sample.jsonl` | `source_id`、`prompt`、`tests` | 原始训练测试来源核对 |
| `grpo.sample.jsonl` | `source_id`、`prompt`、`tests`、`entry_point` | GRPO 专用执行审核训练题 |
| `preference.sample.jsonl` | `prompt`、`chosen`、`rejected` | DPO、Reward Model |
| `ppo.sample.jsonl` | `source_id`、`prompt` | PPO 策略采样 |

正式的 [`preference.sample.jsonl`](preference.sample.jsonl) 已从执行验证训练集抽取 10 对；更易阅读的完整题干、正负代码、变异位置与测试分数见[展开样例](../../docs/preference-examples.md)。DPO 与 Reward Model 共用此格式与同一份正式数据。

训练集与验证集的字段结构相同。本目录抽取的是训练集，目的是让你先看清模型实际
读取的 JSON 形状；验证集只用于训练阶段选择 checkpoint，不参与正式 EvalPlus 分数。

PPO 当前只读取训练题干；`ppo.valid.jsonl` 保留相同划分以便核对来源，并未在 PPO 循环中用于选择 checkpoint。见[构建说明](../../docs/ppo-data.md)。
