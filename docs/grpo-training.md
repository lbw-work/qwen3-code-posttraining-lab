# GRPO：本地准备、单卡验证与正式训练

本阶段从原 LoRA SFT `lora-v1` 出发，学习在同一道 Python 函数题上提高测试通过率。`grpo-v1` 已完成 200 组服务器训练与固定 542 题评测；最佳模型来自第 150 组。结果见[首页](../README.md#results)及[训练元数据](../results/grpo/grpo-v1/training_meta.json)。以下保留数据构建、预检和复现流程；文末本地验收段落记录的是当时状态。

## 1. 数据从哪里来，为什么重新筛选

原 `data/rl/` 是固定版本 NVIDIA OpenCodeInstruct 的训练题干与测试，和 SFT 使用相同来源。GRPO 新入口读取 `data/grpo/`，不是直接把原 26,805 道训练题全部交给奖励函数。原数据、DPO v2、PPO 和已有评测结果继续保留。

准备过程：

1. `prepare_grpo.py` 将 RL 与 SFT 的 `source_id` 对齐，从 SFT 参考代码判断是否为独立的标准库函数题；排除外部依赖、标准输入、工程配置题及不完整类上下文。
2. 使用实际 tokenizer 限制题干至 512 token、参考答案连 EOS 至 512 token。超长题直接剔除，不能截断后继续奖励一个与题干不完整匹配的答案。
3. 保留 5～20 条不同 AST 的单条 `assert`，按规范题干和代码 AST 去重；每个入口函数名每个划分最多五题。题干连续词或函数名与固定 EvalPlus 重合的题排除。该检查不能保证检测所有语义近似题。
4. 开发集只从原 SFT valid 取题，排除与原 SFT train 相同的题干/实现；再排除 GRPO train/valid 之间的相同入口名、代码、题干与来源 ID。开发集曾用于 SFT 验证，不能声称是从未触碰的最终测试集。
5. 在无网络 Docker 内执行参考代码和全返回 None 的空实现探针。参考代码必须全过，空实现不能全过。题干中可明确解析的示例也必须通过。
6. `audit_grpo_data.py` 扩展检查编号示例，检查每条断言是否调用参考函数，执行恒零输出及最多三份运算符/边界变异。剔除新增示例失败、恒零全过、未调用函数的断言，以及至少两份变异均全过的题。后者是保守过滤：变异可能等价，不能把幸存变异数量当成真实错误率。
7. 从断言实际调用的函数确定公共入口，修正把 helper 当入口的记录；排除多公共入口题，再用真实入口重新核查 train/valid 重合及每名最多五题。可解析且通过的新增示例追加到训练测试。原初筛数据仅在本地 `data/raw/grpo-preparation/initial-data/` 保留用于追溯；逐题审查和质量报告保留。模型真实采样成功与否不用于筛选或改写标签。

参考答案放在 `provenance.jsonl`，模型训练只读取以下结构：

```json
{
  "source_id": "来源记录 ID",
  "prompt": "\"\"\"\n完整题意和原始示例\n\"\"\"\n\n",
  "tests": ["assert f(2) == 3", "其余真实来源测试或已验证题干示例"],
  "entry_point": "f"
}
```

这个示意只有两条测试用于展示格式，实际数据至少五条不同断言。十条完整随机训练样例见[展开格式](grpo-data-examples.md)，JSONL 原件在 `data/examples/grpo.sample.jsonl`。

`data/grpo.lock.json` 绑定 train/valid、溯源、审查报告、执行 worker、tokenizer 和固定评测文件的 SHA-256。来源许可证为 OpenCodeInstruct 的 CC BY 4.0，原提交和分片哈希继续由 `data/sft.lock.json`、`data/rl.lock.json` 追溯。

测试全过仍不能证明所有输入正确。没有自动推断题意并伪造期望值；无法明确解析的题干示例会记录为未检查。可解析示例、原测试、变异和探针共同降低奖励噪声，但训练后的公开评测才是能力变化的证据。

## 2. 训练到底做什么

核心实现见 `scripts/train_grpo.py`，每个函数都有中文说明。

| 函数 | 输入 → 输出 | 需要看懂的地方 |
| --- | --- | --- |
| `load_grpo_data` | 冻结文件 → train/valid | 哈希、划分、长度和实际 token ID；兼容保存时 JSON 默认字段补全，但不能接受实际分词变化 |
| `reward_environment` | Docker 探针 → 镜像 ID | 正确/错误实现分别获得 1/0；失败先停止，不把容器故障当正常奖励 |
| `generate_group` | 一个题干 → 四份原始代码、固定旧概率 | `eval()` 关闭 dropout；四份代码从同一个未更新策略采样；答案 token 的 shifted log probability 与采样温度一致 |
| `group_advantages` | 四个测试比例 → 四个优势 | 只在同一道题内标准化；全部同分时返回零，不产生有用方向 |
| `grpo_loss` | 新旧 token 概率、答案优势 → 损失 | token 比值裁剪、每份答案长度取平均；正负优势的裁剪方向不同 |
| `update_group` | 固定四份 rollout → 两次 LoRA 更新 | 每遍累积四份梯度再更新；第二遍旧概率保持不变；零优势组不调用优化器 |
| `evaluate_dev` | 冻结 valid → 贪心逐题分数 | 只用开发集选择 checkpoint，不使用 EvalPlus 隐藏测试 |
| `main` | 配置 → 独立运行目录 | 先数据和奖励预检；正式训练从同一个 SFT 分叉；保留真实 rollout、best 和 last |

初始参数：四候选/题；采样温度 0.8、top-p 1、top-k 0；题干 512/答案 512 token；学习率 `5e-6`；裁剪 0.2；每个有效组更新两遍；梯度范数上限 1；单卡 bf16、LoRA、gradient checkpointing。正式运行默认最多 200 组，每 50 组做开发集贪心评估。

当前实现 `beta=0`，没有 reference KL 或价值网络。这是组内相对奖励与裁剪策略优化的简化 GRPO；裁剪不保证真实 KL 有界。先做小规模运行，以开发集和输出检查决定后续预算，不能预先承诺显著提升。

`--max-steps 200` 指 200 道题的采样组，最多 800 份候选、400 次优化器更新。全同分组跳过，因此实际更新更少。训练 metadata 同时保存组数、有效组、零方差组、更新次数及长度上限命中数。

正式训练先评估 SFT 起点并保存为最佳候选。后续只有开发集“全部测试通过题数”严格提高才替换 best；相同分数保留更早的版本。`last_model/` 保存末次权重，`final_model/` 保存 best。若 best 仍为第 0 组，要如实报告未在开发集超过 SFT，不能把 final_model 的原模型分数归因于 GRPO 提升。

## 3. 本地可以执行的检查

以下命令在 Mac 项目目录执行，使用已有环境：

```bash
cd /Users/eternity/Desktop/qwen3-code-posttrain
export PYTHON_BIN=/opt/anaconda3/envs/qwen3-posttrain/bin/python
export OMP_NUM_THREADS=4
"$PYTHON_BIN" -m pip install --no-deps -e .

# 原始数据已准备好时只校验，不必重复构建。下述两个构建脚本拒绝覆盖。
# "$PYTHON_BIN" scripts/prepare_grpo.py
# "$PYTHON_BIN" scripts/audit_grpo_data.py
# "$PYTHON_BIN" scripts/audit_grpo_data.py --finalize

"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" scripts/sample_datasets.py

# 本地 MPS 只采样并判分，不创建优化器；换新的 run-id 防止覆盖。
"$PYTHON_BIN" -u scripts/train_grpo.py \
  --sft-run runs/lora_sft/lora-v1 \
  --run-id local-grpo-preflight-v3 \
  --preflight-only --max-steps 8 --max-hours 1
```

预检输出在 `runs/grpo_preflight/<编号>/`。`rollouts.jsonl` 是真实生成代码和执行分；`train_log.jsonl` 给出组内优势、同分组与长度命中；`training_meta.json` 的 `optimizer_steps` 应为 0。`signal_observed` 仅表示出现不同分数的组，不代表已训练或能力提高。

有大量零分组时先核查截断、代码协议、函数名和测试格式；有大量满分同分组时说明题目太简单或测试太弱。不要给同分组人工加奖励差，也不要用参考答案冒充真实 rollout。

## 4. AutoDL 上传、远程奖励、冒烟与正式训练

当前 AutoDL 实例没有 Docker。采用 **4090 生成与更新 → SSH 反向隧道 → Mac Docker 判分**，沿用原 worker、镜像和执行限制，不在服务器主机上执行生成代码。完整的 Mac 服务启动、更新包上传、隧道验证、CUDA 冒烟、tmux 正式训练与下载指令见 [Mac Docker 奖励操作指南](grpo-mac-reward.md)。

服务器需已有 `models/base/` 和完整 `runs/lora_sft/lora-v1/`。按照操作指南从当前 Git 提交生成源码包并上传，令牌单独传输。服务器 `QWEN_REWARD_URL` 指向隧道回环端口，`QWEN_REWARD_TOKEN` 从权限 600 的文件读取；每次进入新 tmux shell 都要重新设置。

远程调用校验实际镜像 ID 与 worker 哈希；网络、认证或容器故障停止训练，不能当零分处理。本轮 HTTP/Docker、跨机 SSH 转发与 CUDA 训练已验证；新服务器仍须重新执行环境检查。Mac 息屏可以，但须保持开机、网络、Docker、服务与隧道；不要合盖。tmux 不会替 Mac 保持奖励服务。

若使用真正支持 Docker 的服务器且不设置 `QWEN_REWARD_URL`，仍走原本的本机 Docker 判分入口。正式训练后按 README 的固定 542 题与相同解码协议评测。

## 本地验收记录（历史：正式训练前）

2026-10-06 至 2026-10-07 已完成以下本地检查，完整记录见 [grpo-local-validation.json](grpo-local-validation.json)：

- 最终冻结 **2,005 道训练题、124 道开发题**。初筛 2,176 题经执行补审排除 33 题，再按真实公共入口排除 13 题，另经针对性语义复核排除 1 道错配题；入口修正及每项排除 ID/原因保存在 `data/grpo-quality.json`。
- 审查匹配 2,318 组题干示例，其中 1,662 组可明确解析；执行 4,137 份逻辑变异，3,664 份至少失败一项测试。这些是初筛池的审查覆盖统计，不是最终训练集的模型分数；510 道初筛题没有适用的运算符变异，仍由参考答案、空实现和恒零探针验证。
- 原 LoRA SFT 在 MPS 上生成 8 题、32 份真实代码，**6/8 组有奖励差异**，22 份代码通过全部该题训练测试，9 份部分通过、1 份零分；没有语法错误或 512 token 长度上限命中。耗时约 339 秒，参数更新次数为 0。
- 这八题在入口补审后全部保留；核对采样时的训练文件哈希、题干和测试完全一致，并将 32 份原始代码在最终数据上再次 Docker 判分，得分一致。原始采样元数据保留其真实输入版本；最终复判另存 `runs/grpo_preflight/local-grpo-final-rescore-v1/scores.json`，没有修改或重新包装原始运行记录。
- 全部 18 项单元测试通过，小型 Qwen3 的真实 LoRA 更新、冻结底座、固定旧概率、同分组零更新均通过；脚本编译检查通过。Docker 正确、错误、语法错误及无限循环探针分别得分 `[1,0,0,0]`。
- 原 SFT/RL/偏好/PPO 数据锁继续匹配，固定评测仍为 542 题。正式 GRPO 模型尚不存在；八题抽样不能代表完整训练集或公开基准表现，也不是训练提升证据。

GPU 显存、CUDA bf16 反向、跨机 SSH 奖励隧道及真实训练后的能力变化，需要服务器冒烟与正式评测确认。首轮保持 200 组的小规模预算，检查有效组占比、开发集相对第 0 组的变化、best 与 last 差异，再决定是否扩大。数据检查和保留 SFT 起点降低了噪声与选错 checkpoint 的风险，无法保证公开题通过率显著提高。

2026-10-07 补充：已排除代码分类题与求和测试错配的记录；对全题池进行同类规则扫描并复核命中。记录见 `data/grpo-semantic-review.json`，训练入口校验其哈希。本次审查针对已发现的错配类型，不是全量人工语义审核或正确性保证。
