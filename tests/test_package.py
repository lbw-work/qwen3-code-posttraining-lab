"""检查公共包在项目目录外也能读取正确的冻结输入，且不加载训练依赖。"""

import subprocess
import sys
import tempfile
import unittest


class PackageTest(unittest.TestCase):
    def test_artifact_reader_is_installed_and_lightweight(self):
        # -I 禁用当前目录与 PYTHONPATH，验证 editable 安装而不是偶然的路径导入。
        code = "import sys; from qwen_posttrain.artifacts import verified_tasks; assert len(verified_tasks()) == 542; assert 'torch' not in sys.modules"
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run([sys.executable, "-I", "-c", code], cwd=folder, check=True)
