"""在 Mac 回环地址提供 Docker 判分，经 SSH 反向隧道供 GPU 服务器调用。"""

import argparse
import hmac
import json
import os
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from code_reward import execution_info, score_local


def read_token(path: Path) -> str:
    """首次创建随机令牌文件并限制为当前用户可读；重启复用同一令牌。

    令牌文件放在 /tmp 而非项目目录，不打印其内容。服务器通过 scp 接收副本，
    两端读取同一个值；此令牌与 SSH 密码无关。
    """
    if not path.exists():
        with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as stream:
            stream.write(secrets.token_urlsafe(32)+'\n')
    path.chmod(0o600)
    token = path.read_text().strip()
    if len(token) < 32:
        raise ValueError('奖励令牌太短或为空')
    return token


class Handler(BaseHTTPRequestHandler):
    """只有带正确令牌的健康检查和判分请求被接受。

    先检查身份与体积，再解析 JSON。用户代码不在本进程 exec，score_local
    继续启动原来的受限 Docker 与独立 worker；容器故障返回 HTTP 错误。
    """
    def reply(self, status: int, payload: dict) -> None:
        """明确返回 JSON 与 HTTP 状态；不把异常转换成正常奖励数组。"""
        body = json.dumps(payload,ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def authorized(self) -> bool:
        """常量时间比较认证头；错误请求没有任何执行副作用。"""
        ok = hmac.compare_digest(self.headers.get('Authorization',''),f'Bearer {self.server.token}')
        if not ok: self.reply(401,{'error':'unauthorized'})
        return ok

    def do_GET(self) -> None:
        """健康检查验证本次实际镜像和 worker，服务器再与冻结数据核对。"""
        if not self.authorized(): return
        if self.path != '/health': self.reply(404,{'error':'unknown endpoint'}); return
        try: self.reply(200,execution_info())
        except Exception: self.reply(503,{'error':'Docker unavailable'})

    def do_POST(self) -> None:
        """一次最多 100 份代码、8 MiB 请求；每份代码仍受原执行超时限制。"""
        if not self.authorized(): return
        if self.path != '/score': self.reply(404,{'error':'unknown endpoint'}); return
        try:
            size = int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 8*1024*1024: raise ValueError('invalid size')
            payload = json.loads(self.rfile.read(size))
            codes,tests = payload['completions'],payload['tests']
            if not isinstance(codes,list) or not isinstance(tests,list) or not 1 <= len(codes) <= 100 or len(codes) != len(tests):
                raise ValueError('invalid batch')
            if not all(isinstance(code,str) for code in codes) or not all(isinstance(cases,list) and 1 <= len(cases) <= 128 and all(isinstance(test,str) for test in cases) for cases in tests):
                raise ValueError('invalid code or tests')
        except (ValueError,TypeError,KeyError):
            self.reply(400,{'error':'invalid request'}); return
        try:
            info = execution_info()
            scores = score_local(codes,tests)
            if execution_info() != info: raise RuntimeError('execution environment changed')
            self.reply(200,{**info,'scores':scores})
        except Exception:
            self.reply(503,{'error':'Docker scoring failed; training must stop'})

    def log_message(self, format, *args) -> None:
        """日志只记录请求方法/路径和状态码，既不记录令牌也不记录代码正文。"""
        print(format % args,flush=True)


def main() -> None:
    """启动前验证 Docker，固定监听 127.0.0.1；SSH 隧道是唯一远程入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--token-file',type=Path,default=Path('/tmp/qwen-grpo-reward.token'))
    args = parser.parse_args()
    # 防止 Mac 误继承服务器环境变量，导致服务调用自己而死锁。
    os.environ.pop('QWEN_REWARD_URL',None)
    info = execution_info()
    # ponytail: 单卡顺序判分使用同步 HTTPServer；多训练任务并发时才考虑队列。
    server = HTTPServer(('127.0.0.1',args.port),Handler)
    server.timeout = 30
    server.token = read_token(args.token_file)
    print(f'Mac Docker reward listening on 127.0.0.1:{args.port}; {info}',flush=True)
    try: server.serve_forever()
    finally: server.server_close()


if __name__ == '__main__':
    main()
