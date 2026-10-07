# 实验复现与阶段命令

[返回项目首页](../README.md) · [GRPO 服务器与 Mac 判分](grpo-mac-reward.md)

以下命令需要先满足各阶段的数据、权重和运行环境条件。首次浏览可先读首页的快速开始与代码路线。

## 环境与数据准备

服务器建议 Python 3.11，先按 CUDA 驱动安装匹配的 PyTorch，再安装：

公开仓库随附已处理的 `data/sft/`、`data/rl/`、对应锁文件及 Base 判分结果。
服务器克隆后无需重新运行 `prepare_sft.py` 或 `prepare_rl.py`；这两个脚本用于从原始数据重新构建，遇到已有数据会停止，以免覆盖冻结版本。
Base 和官方参考模型的权重没有随 GitHub 仓库上传；服务器运行 `download_models.py`，按 `models.lock.json` 的提交号下载。已训练的 Full SFT / LoRA SFT 权重从[首页的 Hugging Face 模型列表](../README.md#models)获取。

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
docker build -f Dockerfile.eval -t qwen-code-eval:0.3.1 .
python scripts/download_models.py
```

SFT 数据源是固定版本的 [NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) 的第一份 100,000 行分片。脚本只保留原数据记录的测试全过、单个 Python 代码块、可解析且含顶层函数的样本，并按题干去重、与公开评测题干做连续词重合检查。当前得到训练 26,805 条、验证 1,386 条；`data/sft.lock.json`、`data/rl.lock.json` 保存源文件与输出哈希。连续词检查不能保证发现所有语义近似题，正式报告应如实注明。

随仓库发布的 SFT 和 RL 数据是从 NVIDIA OpenCodeInstruct（[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)）筛选并转换得到的版本；原始仓库、固定提交号、原始分片哈希及转换后的哈希见 `data/sft.lock.json` 和 `data/rl.lock.json`。仓库未包含原始 Parquet 分片。

偏好数据 v2 已在本地构建并冻结：训练 **3,212 对**、验证 **153 对**，供 DPO 与 Reward Model 共用；旧 v1 已在项目清理中移出工作目录。来源为同一固定版本的 OpenCodeInstruct（CC BY 4.0），只使用已有 SFT/RL train；通过单 token 逻辑变异生成候选，在隔离 Docker 中执行。chosen 必须通过全部原测试，rejected 必须通过至少一半、且至少失败一项。5,000 对执行候选还经过题干示例复测、重复实现去除、常见函数名数量限制和人工抽查。详细设计、证据与局限见[执行验证数据说明](../docs/verified-preference-data.md)，格式见[10 条完整随机样例](../docs/preference-examples.md)。

```bash
python scripts/export_verified_preference.py --run-id functions-v2 --limit 5000
python scripts/audit_preference_examples.py --source data/raw/verified-preference/functions-v2/pairs.jsonl
python scripts/prepare_preference.py \
  --source data/raw/verified-preference/functions-v2/pairs.jsonl \
  --source-name nvidia/OpenCodeInstruct@8f3ba5bafe4d6e8db46082cf7ae6741bc370604d+execution-mutations-v2 \
  --verified-functions
python scripts/sample_datasets.py
```

这些命令需要已有 SFT/RL 数据、Base tokenizer 和 Docker 镜像；已有同名数据时拒绝覆盖。正式数据位于本地 `data/preference/`，当前已随仓库发布；复制到训练服务器时须同时带上 [`preference.lock.json`](../data/preference.lock.json)。[质量报告](../data/preference-quality.json)记录来源、示例审查、哈希/划分检查及 10 对样例的 Docker 复跑结果。当前偏好主线使用 OpenCodeInstruct 执行验证数据；已移除未采用的数据导出路线。

执行通过率只是源训练测试下的证据，不能保证所有边界正确或保证 DPO 提升。偏好 valid 的题曾用于 SFT train，因此不能将其 loss 视为新题泛化成绩；最终仍用原冻结 HumanEval+/MBPP+ 比较。

## 各阶段命令

SFT 已在服务器完成；以下命令用于复现训练，`<编号>` 是脚本输出目录中的运行编号：

```bash
python scripts/train_sft.py --mode full
python scripts/train_sft.py --mode lora
```

服务器需同时具备 `data/preference/`、`data/preference.lock.json`、`data/ppo/` 和 `data/ppo.lock.json`。同一张 4090 应按 **DPO → Reward Model → PPO** 顺序训练；PPO 必须等奖励模型完成。`data/ppo/` 的生成和检查见[数据说明](../docs/ppo-data.md)。GRPO 另需 `data/grpo/`、`data/grpo.lock.json` 、`data/grpo-quality.json` 和 `data/grpo-semantic-review.json`；本地预检、上传、服务器冒烟与 tmux 指令见[完整说明](../docs/grpo-training.md)：

```bash
python scripts/train_dpo.py --sft-run runs/lora_sft/<编号>
python scripts/train_reward.py
python scripts/train_ppo.py --sft-run runs/lora_sft/<编号> --reward-run runs/reward/<编号>
python scripts/train_grpo.py --sft-run runs/lora_sft/<编号>
```

DPO、PPO、GRPO 的训练循环和损失都由本项目的 PyTorch 代码实现，不调用 TRL 训练器：DPO 对同一题的 chosen/rejected 计算 policy 与冻结 reference 的答案概率差；PPO 用奖励模型分数、KL、GAE 和裁剪损失更新 LoRA 与价值头；GRPO 每题采样 4 份代码，用训练题自带测试的通过比例计算组内优势，再做两遍裁剪更新。GRPO 的 `beta=0` 不额外驻留 reference；相同得分的组不更新。SFT 与奖励模型仍使用 TRL。

每次训练独立保存 `final_model/`、`training_meta.json` 和日志；DPO、GRPO 保存中间适配器检查点，PPO 另存价值头。SFT、DPO、奖励模型用各自的验证 loss 选择模型；GRPO 用专用开发集全测试通过题数选择 best，另存 `last_model/`。公开评测集不参与选择。GRPO 的 `--max-steps` 表示最多采样多少组题目；默认 200 组、每 50 组验证，题干和答案分别上限 512 token，同分组不更新。PPO 的 `--max-steps` 表示最多执行多少条 rollout。正式训练要求恰好一张 CUDA 卡；GRPO 的 `--preflight-only` 可在本地 MPS 采样判分且不更新参数。DPO `dpo-v2` 更新 201 步，PPO `ppo-v2` 完成 1,000 条 rollout；GRPO `grpo-v1` 完成 200 组与 192 次更新，并已完成本地统一评测。

## 所有模型用同一口径评测

```bash
python scripts/generate_eval.py --stage base --model models/base
python scripts/score_eval.py results/base/<编号>
python scripts/summarize_eval.py results/base/<编号>
```

Full SFT 用 `--stage full_sft --model runs/full_sft/<编号>/final_model`；LoRA SFT、DPO、PPO、GRPO 分别指向自己的 `final_model/`。LoRA 评测会自动加载本项目锁定的 Base 底座再叠加适配器。参考模型可用 `--stage reference --model models/reference` 单独测，不进入必需的六阶段集合。

固定生成口径：EvalPlus 原始 prompt；贪心解码；每题只生成一次；最多 512 个新 token；直接判模型原始代码，不做会误删辅助函数的自动清洗。只允许完整 164+378 题进入正式汇总。模型生成代码只在禁网、只读根文件系统和资源受限的 Docker 内执行。

最后显式传入六个阶段的结果目录，避免脚本误选“最新一次”：

```bash
python scripts/compare_eval.py \
  results/base/<编号> results/full_sft/<编号> results/lora_sft/<编号> \
  results/dpo/<编号> results/ppo/<编号> results/grpo/<编号>
```

它在 `results/comparisons/<编号>/` 保存表格和完整元数据。每个阶段自己的逐题输出、逐题判分、配置、pass@1、生成耗时与训练日志仍在原目录。`ppo-v2` 显式使用 4 小时墙钟上限并在 6,312 秒内完成 1,000 步；墙钟不是精确 GPU 使用时长。本轮完整六阶段对比见[报告](../results/comparisons/six-stage-grpo-v1/comparison.md)。

## 本机检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python scripts/check_project.py
```

这些检查覆盖训练题干去重、PPO token 对齐和优势计算；GRPO 另验证了小型 Qwen3 的真实 LoRA 梯度更新、冻结参数不变、旧概率固定和同分组跳过。Docker 数据审核及原 LoRA SFT 真实采样记录见[GRPO 说明](../docs/grpo-training.md)。本地检查不能替代 CUDA 训练与最终 EvalPlus 评测。 AutoDL 实例通过 SSH 隧道调用 Mac Docker 奖励服务；本地已通过 18 项测试及真实 HTTP/Docker 探针，本轮跨机连接和 CUDA 正式训练已完成。操作见[Mac Docker 奖励指南](../docs/grpo-mac-reward.md)。
