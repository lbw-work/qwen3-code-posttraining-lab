"""PPO、GRPO 和 DPO 复核共用的题目遍历及 token 概率计算。"""

import random

import torch


def prompt_order(size: int, seed: int):
    """逐轮打乱题目索引；一轮内每题最多采到一次。

    PPO 和 GRPO 共用这条遍历规则；一轮覆盖全部题目后才开始下一轮。
    因此步数小于题目数时，同轮不会重复抽到已经处理的题目。
    跑完整轮后重新洗牌，保持后续轮次随机且可由种子复现。
    """
    rng = random.Random(seed)
    while True:
        order = list(range(size))
        rng.shuffle(order)
        yield from order


def action_logprobs(
    logits: torch.Tensor, ids: torch.Tensor, prompt_length: int, temperature: float = 1.0,
) -> torch.Tensor:
    """取出 rollout 中每个生成 token 在策略分布下的对数概率。

    ``ids`` 形状为 ``[1, L+T]``，前 L 个是 prompt，后 T 个是模型采样的动作；``logits``
    形状为 ``[1, L+T, V]``，位置 j 的 V 维向量预测 ids 的 j+1 位置。因此切片从 L-1
    开始、去掉最后一个无目标位置，得到 ``[T, V]``；再按 actions gather 后返回 ``[T]``。
    这个错一格，PPO 会把某个 token 的概率归给前一个或后一个动作，训练就失去意义。
    当采样使用温度 T 时，先将 logits 除以 T 再求概率，保证 old/new 概率与
    实际采样分布一致；DPO 不采样，调用时保留默认 T=1。
    """
    # logits[0, j] 预测 ids[0, j+1]。因此第一个生成 token
    # ids[0, prompt_length] 对应 logits[0, prompt_length-1]。
    # 采样使用 temperature 时，旧/新策略概率必须也来自同一个温度分布。
    selected = (logits[0, prompt_length - 1:-1].float() / temperature).log_softmax(-1)
    actions = ids[0, prompt_length:].unsqueeze(-1)
    return selected.gather(-1, actions).squeeze(-1)
