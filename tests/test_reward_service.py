"""验证真实 HTTP 边界：非法请求不执行代码，容器异常不能伪造奖励。"""

import hashlib
import json
import os
import sys
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from code_reward import ROOT, remote_request, score_batch
from serve_code_reward import Handler


class RewardServiceTest(unittest.TestCase):
    def test_auth_input_and_docker_failure_are_fail_closed(self):
        server = HTTPServer(('127.0.0.1',0),Handler)
        server.token = 'test-token-'*4
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        info = {'worker_sha256':hashlib.sha256((ROOT/'scripts/code_reward_worker.py').read_bytes()).hexdigest(),
                'docker_image_id':json.loads((ROOT/'data/grpo.lock.json').read_text())['docker_image_id']}
        try:
            with patch.dict(os.environ,{'QWEN_REWARD_URL':f'http://127.0.0.1:{server.server_port}',
                                        'QWEN_REWARD_TOKEN':server.token}), \
                 patch('serve_code_reward.execution_info',return_value=info), \
                 patch('serve_code_reward.score_local',side_effect=RuntimeError('Docker unavailable')) as execute:
                with self.assertRaises(HTTPError) as error:
                    remote_request('/score',{'completions':['def f(): return 1'],'tests':['invalid']})
                self.assertEqual(error.exception.code,400)
                execute.assert_not_called()
                with self.assertRaises(HTTPError) as error:
                    score_batch(['def f(): return 1'],[['assert f()==1']])
                self.assertEqual(error.exception.code,503)
                os.environ['QWEN_REWARD_TOKEN']='wrong'
                with self.assertRaises(HTTPError) as error:
                    remote_request('/health')
                self.assertEqual(error.exception.code,401)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_external_url_is_rejected_before_network(self):
        with patch.dict(os.environ,{'QWEN_REWARD_URL':'http://example.com:8765','QWEN_REWARD_TOKEN':'test'}):
            with self.assertRaises(ValueError): remote_request('/health')


if __name__ == '__main__':
    unittest.main()
