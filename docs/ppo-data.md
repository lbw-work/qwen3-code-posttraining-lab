# PPO 题干数据

PPO 策略从 LoRA SFT 出发；冻结参考策略也来自同一 LoRA SFT。奖励模型先用 DPO 共用的 `data/preference/` 训练。PPO 策略每步只从 `data/ppo/train.jsonl` 随机抽取一个函数题干，然后采样代码、由奖励模型打分并更新策略；它不读取该题的参考答案或训练测试。

## 本次冻结结果

| 划分 | 偏好题数 | 超过 PPO 512 token 上限 | PPO 保留题数 |
| --- | ---: | ---: | ---: |
| train | 3,212 | 11 | 3,201 |
| valid | 153 | 0 | 153 |

来源与文件哈希写在 [`data/ppo.lock.json`](../data/ppo.lock.json)。[`prepare_ppo.py`](../scripts/prepare_ppo.py) 核对偏好 provenance 的来源 ID、原 RL train 的逐字题干，以及两份输入的锁；输出每行只有 `source_id` 和 `prompt`。PPO 脚本只读 train，valid 用来检查划分；当前 PPO 没有按 valid 指标选择 checkpoint。实际格式可看[10 条随机样例](../data/examples/ppo.sample.jsonl)。

```bash
python scripts/prepare_ppo.py
python scripts/sample_datasets.py
```

已存在的数据与锁不会被覆盖。服务器运行 PPO 时需上传 `data/ppo/`、`data/ppo.lock.json`，还需奖励模型训练所用的 `data/preference/`、`data/preference.lock.json`；当前偏好 v2、PPO 数据与锁文件均随仓库提供。

## 与 DPO 比较时的边界

两条路线使用同一 LoRA SFT 起点和同源函数题。PPO 代码现按固定种子洗牌，一轮内每题最多抽到一次；默认 200 步只会使用 3,201 道训练题中的 200 道。先做短程 smoke，再根据 4090 的实际速度显式设置 `--max-steps` 和 `--max-hours`；训练元数据会记录实际完成步数。当前 PPO 不按验证集选最佳 checkpoint，奖励模型分数也是学习得到的偏好分，不等于代码测试通过率。最终性能仍须用锁定的 HumanEval+/MBPP+ 独立评测。
