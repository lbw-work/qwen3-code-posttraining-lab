"""GRPO 的关键约束：测试有效性、同分组跳过、真实 LoRA 梯度更新。

小模型使用随机初始化的本地 Qwen3，不下载权重。这里的人工 token 和奖励只是
算式测试输入，不会进入正式数据，也不能作为模型代码能力的证据。
"""

import sys
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_grpo import checked_tests, implementation_key, null_answer
from audit_grpo_data import expanded_examples
from train_grpo import group_advantages, update_group
from qwen_posttrain.policy import action_logprobs


class GrpoTest(unittest.TestCase):
    def test_numbered_examples_are_paired_without_guessing(self):
        prompt = '**Sample Input 1:**\n```\n2\n```\n**Sample Output 1:**\n```\n3\n```\n'
        prompt += '**Sample Input 2:**\n```\n5\n```\n**Sample Output 2:**\n```\n6\n```'
        tests, matched = expanded_examples(prompt, 'def f(x): return x + 1')
        self.assertEqual(matched, 2)
        self.assertEqual(tests, ['assert f(2) == 3', 'assert f(5) == 6'])
        self.assertEqual(expanded_examples(prompt.replace('Output 2:', 'Output 3:'), 'def f(x): return x + 1')[1], 1)

    def test_tests_require_distinct_assertions(self):
        tests = [f'assert f({i}) == {i}' for i in range(5)]
        self.assertEqual(checked_tests(tests + ['assert f(0)==0']), tests)
        self.assertEqual(checked_tests(tests[:4]), [])
        self.assertEqual(checked_tests(tests + ['f(7)']), [])
        self.assertEqual(checked_tests(tests + ['assert True; print(1)']), [])

    def test_reference_fingerprint_and_probe(self):
        code = 'def f(x):\n    """说明"""\n    return x + 1\n'
        self.assertEqual(implementation_key(code), implementation_key('def f(x):\n    return x + 1 # 注释\n'))
        self.assertNotEqual(implementation_key(code), implementation_key(null_answer(code)))
        self.assertIn('return None', null_answer(code))

    def test_bad_rewards_abort(self):
        for scores in ([0, 1], [0, 0, 0, float('nan')], [0, 0, 0, 1.1]):
            with self.assertRaises(ValueError):
                group_advantages(scores)

    def test_real_lora_update_and_zero_group(self):
        # 验证实际 Transformer 前向、答案概率、反向及优化器，而非只验证公式。
        torch.manual_seed(7)
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8, attention_dropout=0.0)
        policy = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(
            r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'],
            lora_dropout=0.0, task_type='CAUSAL_LM'))
        policy.eval()
        optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=1e-3)
        rollouts = []
        with torch.no_grad():
            for token in range(4, 8):
                ids = torch.tensor([[1, 2, token, 3]])
                old = action_logprobs(policy(ids).logits, ids, 2, 0.8).detach().clone()
                rollouts.append((ids, old))
        before = {name: p.detach().clone() for name, p in policy.named_parameters()}
        old_probs = [old.clone() for _, old in rollouts]
        outcome = update_group(policy, optimizer, rollouts, 2, group_advantages([1, 0, 0, 0]))
        self.assertEqual(outcome['updates'], 2)
        changed = []
        for name, parameter in policy.named_parameters():
            self.assertTrue(torch.isfinite(parameter).all())
            if not torch.equal(before[name], parameter):
                changed.append(name)
                self.assertTrue(parameter.requires_grad)
                self.assertIn('lora_', name)
        self.assertTrue(changed, '有效组应当真的改变 LoRA 参数')
        for (_, old), snapshot in zip(rollouts, old_probs):
            self.assertTrue(torch.equal(old, snapshot))
            self.assertFalse(old.requires_grad)

        # 同分组既不能更新参数，也不能增加 Adam 的 step（包括权重衰减）。
        before_zero = {name: p.detach().clone() for name, p in policy.named_parameters()}
        steps = [state['step'].clone() for state in optimizer.state.values()]
        skipped = update_group(policy, optimizer, rollouts, 2, group_advantages([1] * 4))
        self.assertEqual(skipped['updates'], 0)
        self.assertTrue(all(torch.equal(before_zero[n], p) for n, p in policy.named_parameters()))
        self.assertTrue(all(torch.equal(step, state['step']) for step, state in zip(steps, optimizer.state.values())))


if __name__ == '__main__':
    unittest.main()
