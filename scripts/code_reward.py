"""把 GRPO 的候选代码送入隔离容器，返回每份代码通过的测试比例。"""

import json
import hashlib
import math
import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler


from qwen_posttrain.artifacts import ROOT


def remote_request(path: str, payload=None) -> dict:
    """经本机回环端口访问 SSH 隧道，不使用代理或公开网络服务。

    URL 必须为 http://127.0.0.1:端口；随机令牌通过请求头发送，不打印到日志。
    连接中断、认证失败、容器异常全部抛错，不能把基础设施故障伪装成零分。
    """
    url = os.environ['QWEN_REWARD_URL'].rstrip('/')
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or not parsed.port or parsed.path or parsed.query or parsed.fragment or parsed.username:
        raise ValueError('奖励 URL 必须指向 SSH 隧道的 http://127.0.0.1:端口')
    token = os.environ.get('QWEN_REWARD_TOKEN')
    if not token:
        raise ValueError('未设置 QWEN_REWARD_TOKEN')
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = Request(url + path, data=body, headers={'Authorization':f'Bearer {token}', 'Content-Type':'application/json'})
    count = len(payload['completions']) if payload is not None else 0
    with build_opener(ProxyHandler({})).open(request, timeout=15*count+60) as response:
        result = json.load(response)
    expected = hashlib.sha256((ROOT/'scripts/code_reward_worker.py').read_bytes()).hexdigest()
    if result.get('worker_sha256') != expected:
        raise ValueError('Mac 与服务器的奖励 worker 不一致')
    lock_path = ROOT/'data/grpo.lock.json'
    if lock_path.exists() and result.get('docker_image_id') != json.loads(lock_path.read_text())['docker_image_id']:
        raise ValueError('远端 Docker 镜像与 GRPO 数据审核镜像不一致')
    return result


def execution_info() -> dict:
    """返回真正执行代码的机器的 worker 哈希与镜像 ID，供预检和溯源。"""
    if os.environ.get('QWEN_REWARD_URL'):
        return remote_request('/health')
    return {'worker_sha256':hashlib.sha256((ROOT/'scripts/code_reward_worker.py').read_bytes()).hexdigest(),
            'docker_image_id':subprocess.check_output(['docker','image','inspect','qwen-code-eval:0.3.1','--format','{{.Id}}'],text=True).strip()}


def score_batch(completions: list[str], tests: list[list[str]]) -> list[float]:
    """服务器配置隧道时送到 Mac；否则仍使用本机 Docker。

    只传候选代码和训练测试，不传权重。返回值必须数量匹配、有限且在 [0,1]；
    任一不满足直接停止。Mac 服务调用 score_local，避免递归访问隧道。
    """
    if len(completions) != len(tests):
        raise ValueError('候选代码与测试必须一一对应')
    if os.environ.get('QWEN_REWARD_URL'):
        scores = remote_request('/score', {'completions':completions,'tests':tests})['scores']
    else:
        scores = score_local(completions,tests)
    if len(scores) != len(completions) or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
        raise ValueError('奖励数量或数值非法')
    return [float(x) for x in scores]


def score_local(completions: list[str], tests: list[list[str]]) -> list[float]:
    """安全地给一批候选代码打执行分，返回每个样本的通过比例。

    输入的两个列表按位置一一对应；函数将它们序列化为 JSON 标准输入，启动一次受限 Docker
    容器，并由 worker 再为每份候选代码启动独立 Python 子进程。Docker 只读、禁网且不挂载
    权重或训练目录；返回值长度必须等于输入长度，否则宁可报错也不让 GRPO 错配奖励。
    """
    if len(completions) != len(tests):
        raise ValueError("候选代码与测试必须一一对应")
    samples = [{"code": code, "tests": cases} for code, cases in zip(completions, tests, strict=True)]
    command = [
        "docker", "run", "-i", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "64", "--memory", "1g", "--cpus", "1",
        "--tmpfs", "/tmp:rw,exec,size=256m", "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{ROOT / 'scripts'}:/runner:ro", "qwen-code-eval:0.3.1",
        "python", "-I", "/runner/code_reward_worker.py",
    ]
    # 只传测试与生成代码，不把模型权重和训练目录挂载进容器。
    try:
        completed = subprocess.run(
            command, input=json.dumps(samples, ensure_ascii=False), text=True,
            capture_output=True, timeout=15 * len(samples) + 30, check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"执行奖励容器失败：{error.stderr.strip()}") from error
    scores = json.loads(completed.stdout)
    if len(scores) != len(samples):
        raise ValueError("容器返回的分数数量不正确")
    return [float(value) for value in scores]
