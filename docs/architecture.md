# 项目架构与开发约定

[返回首页](../README.md) · [实验复现](reproduction.md) · [本次清理](repository-cleanup.md)

## 目录职责

```text
qwen3-code-posttraining-lab/
├── src/qwen_posttrain/        # 可安装的公共实现
│   ├── artifacts.py          # 项目根目录、分块 SHA-256、冻结评测读取（标准库）
│   └── policy.py             # 题目顺序与生成 token 概率（PyTorch）
├── scripts/                  # 数据、训练、评测与诊断的命令入口
├── tests/                    # CPU 数学、真实小模型更新及 HTTP 边界回归
├── data/                     # 当前五类冻结输入、样例、来源与审核记录
├── results/                  # 六阶段代码、逐题判分、摘要与训练元数据
├── docs/                     # 复现、设计、分析、模型卡与图示
├── .github/workflows/        # CPU / Docker / 数据与结果完整性检查
├── models/                   # 本地 Base / 参考模型，不进入 Git
├── runs/                     # 本地训练权重、检查点与运行日志，不进入 Git
├── eval.jsonl                # 唯一公开评测题干清单
├── eval.lock.json            # 题库版本及哈希
├── models.lock.json          # Base / 参考模型提交号
├── pyproject.toml            # 公共包的 editable 安装配置
└── requirements.txt          # 经过项目验证的依赖版本；PyTorch 按硬件另装
```

## 依赖方向

```text
数据处理 / 训练 / 评测入口 ──→ artifacts.py
DPO 复核 / PPO / GRPO      ──→ policy.py
训练入口                   ──→ train_common.py（目录、数据契约、SFT 起点）
GRPO / 数据审核            ──→ code_reward.py ──→ 隔离 Docker worker
```

数据处理和结果汇总只为读取锁文件时，不应导入生成入口或模型依赖。GRPO 共用
token 概率与题目顺序，不再导入 PPO 训练入口。阶段专属的损失函数与训练循环仍在
对应 `train_*.py` 中，便于逐函数阅读；不额外引入 Trainer 抽象层。

`code_reward_worker.py` 保持独立标准库实现，直接只读挂载到 Docker；容器无需
安装项目包。冻结数据绑定它的哈希，修改 worker 必须重新审核执行证据。

## 命令分组

| 职责 | 入口 |
| --- | --- |
| 固定来源 | `download_models.py`、`freeze_eval.py` |
| 构建输入 | `prepare_sft.py`、`prepare_rl.py`、`prepare_preference.py`、`prepare_ppo.py`、`prepare_grpo.py` |
| 执行审核 | `export_verified_preference.py`、`audit_preference_examples.py`、`audit_grpo_data.py` |
| 训练 | `train_sft.py`、`train_dpo.py`、`train_reward.py`、`train_ppo.py`、`train_grpo.py` |
| 奖励服务 | `code_reward.py`、`code_reward_worker.py`、`serve_code_reward.py` |
| 统一评测 | `generate_eval.py`、`score_eval.py`、`summarize_eval.py`、`compare_eval.py` |
| 分析与检查 | `analyze_dpo.py`、`audit_dpo_data.py`、`sample_datasets.py`、`check_project.py` |

## 环境与贡献前检查

从项目根目录执行，先按硬件安装 PyTorch，再运行：

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
export OMP_NUM_THREADS=4
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
python scripts/check_project.py
```

本项目使用源码 checkout 的 editable 安装，不提供脱离数据目录的 wheel 部署。
`python scripts/train_*.py` 等现有入口保持不变；新环境和服务器同步代码后都需要
安装公共包。`check_project.py` 只读检查，不需要权重、CUDA 或 Docker。

Docker 探针和 CUDA 冒烟是两个独立验收步骤：CPU 测试通过不能证明 GPU 环境已
验证，也不能证明完整训练有能力提升。每次训练用新 `run-id`，不能覆盖已有实验。

GitHub 自动检查固定 Ubuntu 24.04，并按官方提交 SHA 锁定 checkout / setup-python 动作（Node.js 24），避免运行器标签和动作标签漂移。它不下载大模型、不运行正式训练。

## 数据、模型与审计记录规则

- GitHub 发布 SFT、RL、偏好 v2、PPO、GRPO 的当前冻结 JSONL、锁文件及六阶段结果。
- 原始 Parquet、中间数据和模型权重不进入 Git；只有锁定的 GRPO 审核脚本与逐题审核证据从 `data/raw/` 单独纳入发布。
- Full SFT / LoRA SFT 的已发布权重通过首页 Hugging Face 链接获取。DPO / PPO / GRPO 权重仍是本地产物，不能把 JSON 摘要当成可加载权重。
- `results/<stage>/<run-id>/training_meta.json` 是已有运行元数据的原样副本。原机器路径、实际耗时和历史代码哈希不因目录整理而改写。
- `builder_sha256` 等字段指向当时生成数据的代码。整理后的公共导入会改变脚本文件哈希；旧锁仍记录真实历史，不重新签成“新版本数据”。
- 密码、SSH 私钥、奖励令牌与 `.env` 不得提交；令牌通过环境变量及项目外权限受限文件使用。

这是一套可复现实验工程规范；当前 GRPO 仍没有严格断点恢复，也没有多卡训练与在线模型服务。这些限制在首页继续明确保留。
