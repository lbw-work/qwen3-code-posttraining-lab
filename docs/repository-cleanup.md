# 项目整理记录 · 2026-10-07

[返回首页](../README.md) · [架构与开发约定](architecture.md)

## 保留的实验基准

Base `20260922T010155Z`、Full SFT `full-v1`、LoRA SFT `lora-v1`、DPO `dpo-v2`、PPO `ppo-v2`、GRPO `grpo-v1` 的正式结果、训练数据、权重和检查点保留。Base / 官方参考权重保留在本地 models；GRPO best 与 last 都保留。

SFT、RL、偏好 v2、PPO、GRPO 的冻结数据、锁文件、质量报告和来源记录保留。GRPO 原始执行审核与真实预检记录有证据价值，保留。早期标准答案自检也保留，它说明公开评测环境的已知限制。

`execution-verified-functions-v1`、`ppo-function-prompts-v1` 等是数据格式协议名称，并不代表需要删除的旧实验。没有根据文件名中的 v1/v2 自动删除数据。

## 移出工作目录的旧版本和重复副本

以下内容移到本机废纸篓中的独立 `qwen3-cleanup-…` 目录，保留恢复途径。本次移出约 3.42 GiB。正式产物未移走。

| 原项目路径 | 文件内容大小 |
| --- | ---: |
| `data/raw/verified-preference/functions-v1` | 27.2 MiB |
| `data/raw/verified-preference/smoke-v1` | 0.1 MiB |
| `data/raw/verified-preference/functions-v2/initial-freeze` | 14.9 MiB |
| `data/raw/verified-preference/functions-v2/second-freeze` | 14.9 MiB |
| `data/raw/verified-preference/functions-v2/pre-example-audit` | 14.9 MiB |
| `results/smoke/local-reference-20260920` | 0.0 MiB |
| `results/smoke/local-base-20260920` | 0.0 MiB |
| `results/smoke/20260921T121635Z` | 0.0 MiB |
| `runs/hf_export` | 3370.2 MiB |
| `dpo-v2-results.tar.gz` | 63.6 MiB |
| `dpo-v2-results.sha256` | 0.0 MiB |

旧 Themis 导出脚本 `scripts/export_themis.py` 已废弃，此次发布同步移除。DPO v3–v5 在之前清理中已经移除，本次检查未发现残留。

Hugging Face 导出副本的模型卡、许可证与 SHA256SUMS 保留到 `docs/modelcards/`；实际训练权重仍在对应正式 runs 中。零散的传输校验记录和 DPO 日志归入 `docs/evidence/`，其中路径反映当时布局，不改写成新位置。

## 代码与发布边界

- 提取 `src/qwen_posttrain/artifacts.py` 与 `policy.py`，消除数据处理对模型生成入口、GRPO 对 PPO 入口的反向依赖。
- 保留 `scripts/` 作为终端命令入口及逐函数学习路径，训练损失与更新公式不变。
- 添加 editable 包配置、发布完整性检查和 GitHub CPU/Docker 自动检查。
- 当前偏好数据与五阶段训练元数据副本随仓库发布；权重、原始 Parquet、机器凭据和私有运行目录仍不提交。
- 当前脚本哈希会因导入整理而改变；历史数据锁、逐题结果和训练元数据不伪造或重签。

验证范围包括 CPU 回归、Docker 执行探针、冻结输入与六阶段逐题成绩检查，以及清理前后正式文件哈希对照。没有重新训练或重新生成六阶段代码。

## 本次本地验收结果

- 19 项单元测试通过；新增公共包在项目目录外的隔离导入检查。
- 冻结输入、542 道题、六阶段逐题判分与摘要一致，文档链接和 SVG 校验通过。
- 实际 Docker 探针：正确、错误、语法错误、无限循环分别返回 `[1,0,0,0]`。
- 清理前后 201 个保护文件的 SHA-256 一致，覆盖正式输入、权重、逐题评测与冻结 worker。
- 保留原始评测日志和样例中的空格；Git 属性只对这类历史证据关闭空白告警，不重新格式化实验输出。
