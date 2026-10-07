"""对照报告的关键约束：不能误计通过状态，也不能把生成代码变成网页脚本。"""

import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "analyze_dpo", Path(__file__).resolve().parents[1] / "scripts/analyze_dpo.py",
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class DpoAnalysisTest(unittest.TestCase):
    def test_plus_pass_cannot_hide_base_failure(self):
        good = {"base_status": "pass", "plus_status": "pass"}
        bad = {"base_status": "fail", "plus_status": "pass"}
        self.assertEqual(analysis.transition(good, bad), "lost")
        self.assertEqual(analysis.transition(bad, good), "gained")

    def test_model_text_remains_text_in_html(self):
        text = '</pre><script>alert("model output")</script>'
        model = {"raw_completion": text, "solution": text,
                 "judgement": {"base_status": "fail", "plus_status": "fail"}, "static_facts": {}}
        case = {"task_id": "Example/1", "entry_point": "f", "prompt": text,
                "transition": "lost", "suite": "mbpp", "sft": model, "dpo": model, "review": None}
        html = analysis.render_html([case], {"mbpp": dict.fromkeys(analysis.LABELS, 0)})
        self.assertNotIn(text, html)
        self.assertIn('&lt;/pre&gt;&lt;script&gt;', html)
        self.assertEqual(html.count('<script>'), 1)  # 仅页面自身的筛选脚本。


if __name__ == "__main__":
    unittest.main()
