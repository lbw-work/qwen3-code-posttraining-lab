"""只检查会改变实验结论的关键算式与输入约束，不运行完整训练。"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_sft import eval_ngrams, overlaps_eval  # noqa: E402
from prepare_preference import function_pair_reason, overlaps_eval_function, python_code  # noqa: E402
from train_dpo import dpo_loss  # noqa: E402
from train_grpo import grpo_loss, group_advantages  # noqa: E402
from qwen_posttrain.policy import action_logprobs, prompt_order
from train_ppo import gae, ppo_loss  # noqa: E402
from export_verified_preference import mutations  # noqa: E402
from audit_preference_examples import example_test  # noqa: E402


class TrainingContractTest(unittest.TestCase):
    def test_preference_prompt_example_becomes_executable_assertion(self):
        row = {"input": "Return f(a, b).", "chosen": "def f(a, b): return a + b\n"}
        self.assertEqual(example_test(row, "3 5", "8"), "assert f(3, 5) == 8")
        self.assertIsNone(example_test({**row, "input": "Return any order."}, "3 5", "8"))

    def test_ppo_prompt_order_covers_all_questions_before_repeating(self):
        order = prompt_order(5, 42)
        self.assertEqual(set(next(order) for _ in range(5)), set(range(5)))
        self.assertEqual(set(next(order) for _ in range(5)), set(range(5)))

    def test_preference_mutates_code_tokens_only(self):
        original = 'def total(xs):\n    """0 + 1 >= 0"""\n    # 0 + 1 >= 0\n    return sum(xs) + 1\n'
        variants = mutations(original, 42, 6)
        self.assertTrue(variants)
        self.assertEqual(variants, mutations(original, 42, 6))
        for code, change in variants:
            self.assertIn('"""0 + 1 >= 0"""', code)
            self.assertIn('# 0 + 1 >= 0', code)
            self.assertEqual(change['line'], 4)
            self.assertNotEqual(code, original)
        self.assertFalse(mutations('def f(n):\n    while n > 0:\n        n -= 1\n    return n\n', 42, 6))

    def test_algorithm_preference_filters(self):
        good = "def total(xs):\n    return sum(xs)\n"
        bad = "def total(xs):\n    return sum(xs[:-1])\n"
        self.assertIsNone(function_pair_reason("Return the sum of a list.", good, bad))
        self.assertEqual(function_pair_reason("Read from standard input.", good, bad), "program_or_embedded_code_prompt")
        self.assertEqual(function_pair_reason("Calculate a discount for a product.", good, bad), "engineering_prompt")
        self.assertEqual(function_pair_reason("Adjust optimization_level and chunk_size.", good, bad), "engineering_prompt")
        self.assertEqual(function_pair_reason("Sum a list", good, good + "# changed comment\n"), "same_code_ast")
        self.assertEqual(function_pair_reason("Sum a list", good, "def other(xs): return 0"), "different_function_interface")
        self.assertEqual(function_pair_reason("Sum a list", "import pandas\n" + good, bad), "external_dependency")
        self.assertEqual(function_pair_reason("Sum a list", good, "def total(xs): return int(input())"), "io_or_dynamic_execution")
        self.assertEqual(function_pair_reason("Sum a list", good, "def total(xs): return expected + sum(xs)"), "undefined_global")
        self.assertIsNone(function_pair_reason("Sum a list", "def total(xs):\n    def inner(x): return x + len(xs)\n    return sum(inner(x) for x in xs)", bad))
        self.assertTrue(overlaps_eval_function("def hexKey(s): return len(s)", "def hexKey(s): return 0", {"hexkey"}))
        self.assertFalse(overlaps_eval_function(good, bad, {"hexkey"}))

    def test_preference_requires_code_only(self):
        self.assertEqual(python_code("\ufeff```python\ndef f(x):\n    return x\n```"), "def f(x):\n    return x\n")
        self.assertIsNone(python_code("Here is a solution:\ndef f(x): return x"))

    def test_eval_overlap(self):
        task = [{"prompt": "Write a Python function to reverse every word in a sentence."}]
        spans = eval_ngrams(task)
        self.assertTrue(overlaps_eval(task[0]["prompt"], spans))
        self.assertFalse(overlaps_eval("Implement a binary search tree insertion", spans))

    def test_ppo_token_alignment_and_advantage(self):
        ids = torch.tensor([[0, 1, 2, 3]])  # 前 2 个是题目，后 2 个是模型动作。
        logits = torch.zeros(1, 4, 5)
        logits[0, 1, 2] = 10  # 位置 1 预测第一个答案 token 2。
        logits[0, 2, 3] = 10  # 位置 2 预测第二个答案 token 3。
        self.assertEqual(action_logprobs(logits, ids, 2).shape, (2,))
        self.assertTrue(torch.all(action_logprobs(logits, ids, 2) > -0.001))
        self.assertTrue(torch.all(action_logprobs(logits, ids, 2, temperature=0.8) > -0.001))
        advantages, returns = gae(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0]), lam=1.0)
        self.assertTrue(torch.allclose(advantages, torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(returns, torch.tensor([1.0, 1.0])))
        total, policy, value = ppo_loss(
            torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2),
            advantages, returns,
        )
        self.assertTrue(torch.isfinite(total))
        self.assertAlmostEqual(policy.item(), -1.0)
        self.assertAlmostEqual(value.item(), 0.5)

    def test_dpo_prefers_chosen_relative_to_frozen_reference(self):
        # 两个模型起点相同时，优势差为 0，DPO 损失应为 ln(2)。
        chosen = torch.tensor(-2.0, requires_grad=True)
        rejected = torch.tensor(-2.0, requires_grad=True)
        loss = dpo_loss(chosen, rejected, torch.tensor(-2.0), torch.tensor(-2.0))
        self.assertAlmostEqual(loss.item(), 0.693147, places=5)
        loss.backward()
        self.assertLess(chosen.grad.item(), 0)  # 梯度下降会提高 chosen 的 logp。
        self.assertGreater(rejected.grad.item(), 0)  # 梯度下降会降低 rejected 的 logp。

    def test_grpo_group_advantage_and_clipped_loss(self):
        advantages = group_advantages([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(advantages.mean().item(), 0.0)
        self.assertGreater(advantages[0].item(), 0)
        self.assertTrue(torch.all(advantages[1:] < 0))
        self.assertTrue(torch.equal(group_advantages([0.0] * 4), torch.zeros(4)))
        # 正优势的答案已比旧策略提高很多时，clip 后不再继续推动它。
        new = torch.tensor([1.0, 1.0], requires_grad=True)
        loss = grpo_loss(new, torch.zeros(2), torch.tensor(1.0))
        loss.backward()
        self.assertTrue(torch.equal(new.grad, torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
