# AutoDL GPU 训练 + Mac Docker 奖励判分

4090 服务器负责生成代码和更新 LoRA；Mac 保留原 Docker worker 和执行限制，通过带令牌的 SSH 反向隧道返回测试通过比例。服务器不需要安装 Docker。所有请求只监听回环地址，令牌单独传输，不放进项目或准备包。

## 1. Mac 终端 A：启动奖励服务

先打开 Docker Desktop，等待 Docker 就绪。这个终端需一直保留：

```bash
cd /Users/eternity/Desktop/qwen3-code-posttrain
/opt/anaconda3/envs/qwen3-posttrain/bin/python -m pip install --no-deps -e .
/opt/anaconda3/envs/qwen3-posttrain/bin/python -u scripts/serve_code_reward.py \
  --port 8765 --token-file /tmp/qwen-grpo-reward.token
```

首次启动创建权限 600 的随机令牌文件；重启复用令牌，不会打印内容。启动时检查真实 Docker 镜像和 worker。

## 2. Mac 终端 B：上传更新包，然后保留 SSH 隧道

源码包从当前已提交的 Git 版本生成，包含公共包、入口、冻结数据和文档，不包含本地权重或令牌。先提交/拉取需要复现的版本，再生成包。以下端口和目录为本次服务器示例，重新租机时按控制台更新。

```bash
git archive --format=tar.gz --output=/tmp/qwen-posttrain-source.tar.gz HEAD
# 校验文件中只写文件名，服务器才能在自己的目录直接校验。
(cd /tmp && shasum -a 256 qwen-posttrain-source.tar.gz > qwen-posttrain-source.tar.gz.sha256)

scp -P 25602 \
  /tmp/qwen-posttrain-source.tar.gz \
  /tmp/qwen-posttrain-source.tar.gz.sha256 \
  /tmp/qwen-grpo-reward.token \
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/

caffeinate -i ssh -p 25602 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -N -R 127.0.0.1:8765:127.0.0.1:8765 \
  root@connect.bjb1.seetacloud.com
```

SSH 如提示密码，正常手动输入。隧道成功后无输出、一直等待属于正常现象。若提示转发失败或 `administratively prohibited`，先解决隧道问题，不能开始训练。本轮已验证 HTTP → Docker 与服务器反向转发；更换服务器仍须重做探针。

Mac 可息屏，但保持开机、网络连通、Docker 正常、两个终端运行；不要合盖。服务器 tmux 只保护训练进程，不保护 Mac 或隧道。服务或隧道故障将报错停止训练，不会记作零奖励；当前训练入口不支持自动断点续训。

## 3. 服务器终端：解压、检查文件和远程奖励

用另一个终端 SSH 登录服务器：

```bash
ssh -p 25602 root@connect.bjb1.seetacloud.com
```

以下均在服务器执行：

```bash
cd /root/autodl-tmp
sha256sum -c qwen-posttrain-source.tar.gz.sha256
mkdir -p /root/autodl-tmp/qwen3-code-posttrain
cd /root/autodl-tmp/qwen3-code-posttrain
tar -xzf /root/autodl-tmp/qwen-posttrain-source.tar.gz
chmod 600 /root/autodl-tmp/qwen-grpo-reward.token

export PYTHON_BIN=/root/autodl-tmp/conda-envs/qwen-code/bin/python
export OMP_NUM_THREADS=4
export HF_HOME=/root/autodl-tmp/hf-cache
export QWEN_REWARD_URL=http://127.0.0.1:8765
export QWEN_REWARD_TOKEN="$(cat /root/autodl-tmp/qwen-grpo-reward.token)"

"$PYTHON_BIN" -m pip install --no-deps -e .
"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" scripts/check_project.py
"$PYTHON_BIN" - <<'CHECK'
import sys
import torch
sys.path.insert(0, 'scripts')
from train_grpo import reward_environment
from code_reward import score_batch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.is_bf16_supported()
print('GPU:', torch.cuda.get_device_name(0))
print('奖励环境:', reward_environment())
scores = score_batch([
    'def f(x): return x + 1',
    'def f(x): return 0',
    'def f(:',
    'def f(x):\n    while True: pass',
], [['assert f(1) == 2']] * 4)
assert scores == [1, 0, 0, 0], scores
print('远程 Docker 探针通过:', scores)
CHECK
```

项目检查失败时不能跳过。源码包不含权重，另须准备 Base 与原 LoRA SFT；`train_grpo.py` 会检查 tokenizer、worker 和冻结输入。镜像 ID 和 worker 哈希必须匹配冻结的数据审核环境。测试通过后再做 CUDA 冒烟。

## 4. 服务器 tmux：10 组冒烟

```bash
tmux new -s grpo-reproduction-v1
```

进入 tmux 后重新设置环境，避免新 shell 丢失变量：

```bash
cd /root/autodl-tmp/qwen3-code-posttrain
export PYTHON_BIN=/root/autodl-tmp/conda-envs/qwen-code/bin/python
export OMP_NUM_THREADS=4
export HF_HOME=/root/autodl-tmp/hf-cache
export QWEN_REWARD_URL=http://127.0.0.1:8765
export QWEN_REWARD_TOKEN="$(cat /root/autodl-tmp/qwen-grpo-reward.token)"
set -o pipefail

"$PYTHON_BIN" -u scripts/train_grpo.py \
  --sft-run runs/lora_sft/lora-v1 \
  --run-id smoke-grpo-reproduction-v1 \
  --max-steps 10 --eval-every 10 --max-hours 1 \
  2>&1 | tee /root/autodl-tmp/smoke-grpo-reproduction-v1.log
```

冒烟前和结束均评估 124 道开发题，因此不只是十组采样的时间。只在上一个命令成功退出后检查：

```bash
"$PYTHON_BIN" - <<'CHECK'
import json
from pathlib import Path
p = Path('runs/grpo/smoke-grpo-reproduction-v1')
m = json.loads((p / 'training_meta.json').read_text())
print(json.dumps(m, ensure_ascii=False, indent=2))
assert m['completed_groups'] == 10
assert m['optimizer_steps'] > 0
assert m['reward_backend'] == 'ssh_tunnel_mac_docker'
for folder in ('final_model', 'last_model'):
    f = p / folder / 'adapter_model.safetensors'
    assert f.is_file() and f.stat().st_size > 0
    print(f, f.stat().st_size)
CHECK
```

若显存溢出、奖励请求失败、没有有效更新或组数不足，先排查，不进入正式训练。已有同名 run-id 时用新编号；入口拒绝覆盖。

## 5. 同一服务器 tmux：正式训练

确认冒烟检查通过后，单独执行下面命令。从原 SFT 重新开始，不读取冒烟权重：

```bash
"$PYTHON_BIN" -u scripts/train_grpo.py \
  --sft-run runs/lora_sft/lora-v1 \
  --run-id grpo-reproduction-v1 \
  --max-steps 200 --eval-every 50 --max-hours 4 \
  2>&1 | tee /root/autodl-tmp/grpo-reproduction-v1.log
```

200 步是 200 组题目、每组四候选；同分组跳过优化，所以不等于 200 次参数更新。记录有效组、优化器次数、best 与 last。`final_model` 是开发集选出的 best；若 best_group 为 0，表示未超过 SFT 起点，不能声称 GRPO 已提升。4 小时在组边界检查，评估和保存可能使实际时间更长。

`Ctrl+B` 再 `D` 离开 tmux；`tmux attach -t grpo-reproduction-v1` 返回。Mac 两个终端仍须保持运行。

## 6. Mac 下载训练记录

训练结束后在 Mac 的新终端执行：

```bash
cd /Users/eternity/Desktop/qwen3-code-posttrain
mkdir -p runs/grpo
rsync -avh --progress -e 'ssh -p 25602' \
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/qwen3-code-posttrain/runs/grpo/grpo-reproduction-v1/ \
  runs/grpo/grpo-reproduction-v1/
scp -P 25602 root@connect.bjb1.seetacloud.com:/root/autodl-tmp/grpo-reproduction-v1.log \
  runs/grpo/grpo-reproduction-v1/server.log
```

再按项目统一生成 → EvalPlus Docker 判分 → 汇总流程评测，不能拿训练题奖励或开发集分数当 HumanEval+/MBPP+ 成绩。
