# 执行验证的 Python 函数偏好数据

本轮正式方案使用已冻结的 OpenCodeInstruct 函数答案与训练测试，为 DPO 和 Reward Model 提供同一份偏好对。目标是学习避免边界、比较、布尔和算术逻辑错误。

## 本轮产物

| 项目 | 实际数量 |
| --- | ---: |
| 扫描 SFT 训练记录 | 23,312 |
| Docker 执行题目 | 8,300 |
| Docker 执行正例和变异候选 | 41,026 |
| 通过执行筛选的候选偏好对 | 5,000 |
| 可解析并执行的题干示例 | 3,381 |
| 示例执行失败 | 151 条，涉及 144 道题 |
| 正式筛选时排除示例失败 | 142 对 |
| 去除相同 chosen 实现 | 607 对 |
| 同名函数超过每名 10 对上限 | 744 对 |
| 工程／业务题过滤 | 136 对 |
| 人工审查排除 | 6 对 |
| 正式训练集 | 3,212 对 |
| 正式验证集 | 153 对 |

正式 v2 文件为 `data/preference/train.jsonl`、`valid.jsonl`、`provenance.jsonl`；当前冻结 v2 随仓库发布；旧 v1 与中间冻结副本已在项目清理中移出工作目录。全部保留样本各有 10 条原测试；chosen 全过，rejected 通过率为 50%–90%，中位数 80%。版本与哈希见 [`preference.lock.json`](../data/preference.lock.json)，人工排除原因见[审查清单](../data/preference-review.json)，示例审查的覆盖范围与可疑 ID 见[执行报告](../data/preference-example-audit.json)，逐项检查见[质量报告](../data/preference-quality.json)。[10 条完整样例](preference-examples.md)列出题干、两份代码、变异位置和 Docker 复跑分数。DPO `dpo-v2` 与奖励模型已完成服务器训练，DPO 评测结果见项目 README。

## 数据从哪里来

来源为 [NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)，提交号 `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`，原始分片 `data/train-00000-of-00050.parquet`。来源及原始文件 SHA-256 已在 [`sft.lock.json`](../data/sft.lock.json) 固定。本次只从已有 SFT/RL **训练集**取题目、代码与测试，不使用它们的验证集，也不读取 EvalPlus 隐藏测试。

数据是 OpenCodeInstruct（CC BY 4.0）的派生版本；修改包括函数任务筛选、单 token 逻辑变异、训练测试执行、去重和新的偏好划分。保留原始 `source_id` 以便追溯。

## 构建流程与设计理由

```text
data/sft/train.jsonl 的题干与答案
              + data/rl/train.jsonl 的同题测试
  → export_verified_preference.py
      每题生成最多 6 份单 token 变异
      在受限 Docker 中执行 chosen 和所有候选
      保留 chosen=100%、50%≤rejected<100% 的一对
  → data/raw/verified-preference/functions-v2/pairs.jsonl
      保存训练字段、测试、通过率和变异位置
  → audit_preference_examples.py
      执行能解析的题干示例，记录正例不一致的原始 ID
  → prepare_preference.py --verified-functions
      校验导出证据、长度、重合、重复、示例审查与划分
  → data/preference/{train,valid,provenance}.jsonl
              + data/preference.lock.json
  → train_dpo.py / train_reward.py
```

[`export_verified_preference.py`](../scripts/export_verified_preference.py) 的 `mutations()` 使用 Python 的 tokenize，只替换一个代码 token，保留原缩进、注释和 docstring。可变异比较符、布尔值、逻辑符、算术符和 0/1 常量。它跳过 while 区域与函数定义首行，以减少无限循环负例和接口变动。

“改过代码”不等于“有错误”。`verify_batch()` 使用现有 [`code_reward.py`](../scripts/code_reward.py) 在禁网、只读、限资源 Docker 中执行双方。每题至少需要 5 条不同测试。正例必须全过，负例必须通过至少一半、但不是全部测试；多个负例符合条件时取通过率最高的一份，优先保留局部错误。未出现可验证差异的候选不会获得负标签。

[`audit_preference_examples.py`](../scripts/audit_preference_examples.py) 只解析明确的 Python 字面量或按参数名列出的赋值，生成题干示例断言并在相同 Docker 环境执行。5,000 对中有 3,381 条示例可解析；151 条断言失败，涉及 144 道题。失败可能是题干、正例或示例格式问题，因此保守排除；无法解析的示例仍可能存在错误，不能把这个数字解释成精确的题目错误率。

[`prepare_preference.py`](../scripts/prepare_preference.py) 检查原始导出、worker 和示例审查哈希，排除人工确认的问题 ID 与示例失败 ID，再过滤非函数代码、业务题、外部依赖、缺失全局变量、重复实现、评测题重合和超长记录。同名入口函数最多保留 10 对，防止 factorial 等常见题占据过多训练样本；这仍不能识别所有改名后的同类题。每支答案含 EOS 最多 512 token，与题干合并最多 1,024 token。代码不会被整体重排，也不会自动把错误答案修成另一份程序。

## 训练与验证怎么划分

先按规范题干和 chosen AST（忽略注释和 docstring）去重，每题保留一对。偏好划分使用 `SHA256("preference-v1:" + 规范题干)` 的前 8 位，模 20 等于零时进入 valid，其余进入 train，约为 95/5。所有样本都来自 SFT train，因此需要独立划分盐；直接使用 SFT 原划分会让偏好 valid 为空。

DPO/RM 的 valid 不参与它们的参数更新，但这些题的正确答案在之前 SFT 中出现过。因此这里的验证 loss 用于选择偏好训练 checkpoint，不能解释为全新的算法题泛化成绩。最终能力比较继续使用原冻结的 HumanEval+/MBPP+。

## 保存的格式与证据

训练文件只包含三个字段：

```json
{"prompt": "题干与原有输入输出样例", "chosen": "原正确候选代码", "rejected": "单处逻辑变异代码"}
```

`provenance.jsonl` 另存题干哈希、原始 ID、划分、token 数、测试原文、双方通过率和变异行列坐标。训练器读取三字段数据，测试不附加到模型输入中。锁文件保存来源、文件哈希、tokenizer 版本、构建脚本、执行 worker 和 Docker image ID。

原始执行导出、smoke 输出与进度日志属于数据准备产物；正式训练只需 `data/preference/`、`data/preference.lock.json` 及项目锁定的模型。单卡 4090 用于之后训练，这一数据准备阶段在本地 Docker 完成。

## 复现命令

在包含既有 SFT/RL 数据和 tokenizer 的项目环境运行：

```bash
python scripts/export_verified_preference.py --run-id functions-v2 --limit 5000
python scripts/audit_preference_examples.py \
  --source data/raw/verified-preference/functions-v2/pairs.jsonl
python scripts/prepare_preference.py \
  --source data/raw/verified-preference/functions-v2/pairs.jsonl \
  --source-name nvidia/OpenCodeInstruct@8f3ba5bafe4d6e8db46082cf7ae6741bc370604d+execution-mutations-v2 \
  --verified-functions
python scripts/sample_datasets.py
```

已有同名导出、示例审查或正式数据时脚本会拒绝覆盖。另一个构建实验应使用新 run-id 并保留原有冻结数据；训练时明确记录所使用的版本。

## 能说明什么

执行结果证明的是给定训练测试下的偏好差异。源测试可能覆盖不足或含错误，复杂度上限、未覆盖边界和语义污染仍需要独立检查。题干 n-gram 和评测入口名称检查是保守的文本过滤，不能证明彻底消除语义重复。

负例来自规则变异，并非当前 `lora-v1` 模型的真实生成。这批题来自 SFT 训练集，来源测试和可解析示例都不能证明所有自然语言约束正确；人工抽查确实发现过与示例冲突的题。此版本适合开展更大规模、可追溯的 DPO/RM 实验，但不能承诺显著提升。后续可用模型真实生成与相同训练测试构建新的版本。是否提高算法题通过率，仍以训练后的同口径评测为准。
