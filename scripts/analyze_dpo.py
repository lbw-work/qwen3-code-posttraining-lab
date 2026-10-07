"""只读取已有判分，导出 SFT 与 DPO 的逐题对照；不生成、不执行代码、不重新判分。"""

import argparse
import ast
from collections import Counter
import difflib
from html import escape
import json
from pathlib import Path


from qwen_posttrain.artifacts import ROOT, sha256
LABELS = {"lost": "退化", "gained": "改善", "both_pass": "均通过", "both_fail": "均失败"}


def read_jsonl(path: Path) -> dict:
    """按 task_id 索引 JSONL，并拒绝重复题目，防止后面的字典覆盖掩盖缺失记录。"""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["task_id"] in rows:
            raise ValueError(f"重复 task_id：{path} / {row['task_id']}")
        rows[row["task_id"]] = row
    return rows


def passed(attempt: dict) -> bool:
    """沿用项目的 Plus 通过定义：原始测试和新增测试都必须 pass。

    不能只看 plus_status：例如 Mbpp/71 的 DPO 是 base=fail、plus=pass，
    仍必须算失败，否则逐题变化和原 summary.json 会不一致。
    """
    return attempt["base_status"] == "pass" and attempt["plus_status"] == "pass"


def transition(before: dict, after: dict) -> str:
    """将同一题的两次判分映射到四个互斥类别，所有题恰好属于其中一类。"""
    a, b = passed(before), passed(after)
    if a and b:
        return "both_pass"
    if a:
        return "lost"
    return "gained" if b else "both_fail"


def load_run(folder: Path, stage: str, tasks: dict) -> tuple[dict, dict]:
    """读取一个阶段，并校验摘要、逐题判分、生成记录和题目清单互相一致。

    返回 (按 task_id 索引的样本, 按 suite 索引的统计)。sample 保留原始 completion，
    judge 保留真正被评分的完整 solution，两者都展示，避免把拼接后的代码误称为模型输出。
    本函数校验的是评测证据；它不能证明训练数据无泄漏或服务器权重来源正确。
    """
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if config["stage"] != stage or summary["stage"] != stage:
        raise ValueError(f"阶段不符：{folder}")
    if config["eval_sha256"] != sha256(ROOT / "eval.jsonl") or summary["eval_sha256"] != config["eval_sha256"]:
        raise ValueError(f"评测题库哈希不符：{folder}")
    if config["limit_per_suite"] is not None or config["generation"] != {"do_sample": False, "max_new_tokens": 512}:
        raise ValueError(f"不是固定贪心 512 token 的完整评测：{folder}")
    rows, counts = {}, {}
    for suite in ("humaneval", "mbpp"):
        path = folder / f"{suite}.samples_eval_results.json"
        expected = {key for key, row in tasks.items() if row["suite"] == suite}
        judges = json.loads(path.read_text(encoding="utf-8"))["eval"]
        samples = read_jsonl(folder / f"{suite}.samples.jsonl")
        if set(samples) != expected or set(judges) != expected:
            raise ValueError(f"题目覆盖不完整：{folder} / {suite}")
        if sha256(path) != summary["suites"][suite]["result_sha256"]:
            raise ValueError(f"判分文件与摘要不符：{path}")
        for key, attempts in judges.items():
            if len(attempts) != 1 or attempts[0]["task_id"] != key:
                raise ValueError(f"不是每题一次判分：{key}")
            if samples[key]["solution"] != attempts[0]["solution"]:
                raise ValueError(f"生成代码与被评分代码不同：{key}")
            rows[key] = {"sample": samples[key], "judge": attempts[0]}
        counts[suite] = {"passed": sum(passed(x[0]) for x in judges.values()), "total": len(expected)}
        for field, value in counts[suite].items():
            if summary["suites"][suite][field] != value:
                raise ValueError(f"计数与摘要不符：{folder} / {suite}")
    return rows, counts


def static_facts(solution: str, entry_point: str) -> dict:
    """静态解析被评分代码，记录语法、入口和顶层 assert，不执行模型输出。

    SyntaxError 是可直接核实的事实。入口缺失只检查模块顶层定义；它不尝试推断动态
    创建的函数。顶层 assert 数量也只是风险线索，不能自动判定这些断言一定失败。
    不使用 exec/eval，也不调用生成代码中的任何函数。
    """
    facts = {"syntax_error": None, "entry_defined": None, "top_level_asserts": None}
    try:
        tree = ast.parse(solution)
    except SyntaxError as error:
        facts["syntax_error"] = {"message": error.msg, "line": error.lineno, "text": error.text}
        return facts
    facts["entry_defined"] = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
        for node in tree.body
    )
    facts["top_level_asserts"] = sum(isinstance(node, ast.Assert) for node in tree.body)
    return facts


def build_cases(tasks: dict, before: dict, after: dict, reviews: dict) -> tuple[list, dict]:
    """统计全部题目的变化，但只导出改善/退化题，便于先阅读最有信息的案例。

    人工 review 和自动事实分别保存：前者是阅读题意与代码后的解释，后者来自原始
    判分及静态解析。未写 review 的改善题明确标为未分类，不用规则猜测算法根因。
    """
    cases = []
    counts = {suite: Counter({key: 0 for key in LABELS}) for suite in ("humaneval", "mbpp")}
    for key, task in tasks.items():
        kind = transition(before[key]["judge"], after[key]["judge"])
        counts[task["suite"]][kind] += 1
        if kind not in ("lost", "gained"):
            continue
        models = {}
        for name, row in (("sft", before[key]), ("dpo", after[key])):
            models[name] = {
                "raw_completion": row["sample"]["raw_completion"],
                "solution": row["judge"]["solution"],
                "judgement": {field: row["judge"][field] for field in
                              ("base_status", "plus_status", "base_fail_tests", "plus_fail_tests")},
                "static_facts": static_facts(row["judge"]["solution"], task["entry_point"]),
            }
        cases.append({**task, "transition": kind, **models, "review": reviews.get(key)})
    cases.sort(key=lambda row: (row["transition"] != "lost", row["suite"], int(row["task_id"].split("/")[1])))
    if set(reviews) - {row["task_id"] for row in cases}:
        raise ValueError("人工审阅文件包含没有发生变化的题目，需核对来源")
    return cases, counts


def render_html(cases: list, counts: dict) -> str:
    """生成独立 HTML：支持筛选，并排展示原始 completion，折叠展示完整 solution 和 diff。

    所有题干、代码和人工说明均经过 HTML 转义，因此模型输出中的标签不会变成网页脚本。
    页面不请求网络资源，也不执行生成代码；筛选脚本只读取卡片的文本和类别属性。
    """
    cards = []
    categories = sorted({row["review"]["category"] for row in cases if row["review"]})
    for row in cases:
        review = row["review"] or {"category": "未分类", "note": "尚未逐题分类；请先阅读原始代码与判分。"}
        panes = []
        for key, title in (("sft", "LoRA SFT"), ("dpo", "DPO")):
            model = row[key]
            judgement = model["judgement"]
            panes.append(f'<section><h3>{title} · base={escape(judgement["base_status"])} / plus={escape(judgement["plus_status"])}</h3>'
                         f'<p class="muted">原始 completion · {len(model["raw_completion"])} 字符（不是 token 数）</p>'
                         f'<pre>{escape(model["raw_completion"])}</pre>'
                         f'<details><summary>完整被评分代码 solution</summary><pre>{escape(model["solution"])}</pre></details>'
                         f'<details><summary>判分证据与静态事实</summary><pre>{escape(json.dumps({"judgement": judgement, "static_facts": model["static_facts"]}, ensure_ascii=False, indent=2))}</pre></details></section>')
        difference = "".join(difflib.unified_diff(
            row["sft"]["raw_completion"].splitlines(keepends=True),
            row["dpo"]["raw_completion"].splitlines(keepends=True), fromfile="SFT completion", tofile="DPO completion",
        ))
        anchor = row["task_id"].replace("/", "-")
        cards.append(f'<article id="{escape(anchor)}" data-kind="{row["transition"]}" data-suite="{row["suite"]}" data-category="{escape(review["category"], quote=True)}">'
                     f'<h2>{escape(row["task_id"])} · {LABELS[row["transition"]]} <span class="badge">{escape(review["category"])}</span></h2>'
                     f'<p><strong>入口函数：</strong>{escape(row["entry_point"])}</p>'
                     f'<p class="note"><strong>人工静态审阅：</strong>{escape(review["note"])}</p>'
                     f'<details><summary>题目 prompt（点击展开）</summary><pre>{escape(row["prompt"])}</pre></details>'
                     f'<div class="columns">{"".join(panes)}</div>'
                     f'<details><summary>SFT → DPO 的逐行差异</summary><pre>{escape(difference)}</pre></details></article>')
    rows = "".join(f'<tr><td>{suite}</td>' + "".join(f'<td>{count[key]}</td>' for key in LABELS) + '</tr>' for suite, count in counts.items())
    options = "".join(f'<option>{escape(category)}</option>' for category in categories + ["未分类"])
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DPO 逐题对照 · LoRA SFT → DPO</title><style>
body{font:16px/1.65 system-ui,sans-serif;background:#f3f5f8;color:#202c40;margin:0;padding:28px;max-width:1600px;margin:auto}
h1{font-size:30px}h2{font-size:23px}h3{font-size:18px}article,header{background:white;border:1px solid #dce2eb;border-radius:14px;padding:24px;margin:20px 0}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}.columns section{min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fb;padding:16px;border-radius:8px;font:13px/1.6 ui-monospace,monospace}
.muted{color:#66758a;font-size:13px}.badge{font-size:13px;background:#e9eef9;padding:4px 10px;border-radius:6px}.note{background:#fff7e6;padding:12px;border-left:3px solid #cf9e36}
summary{cursor:pointer;color:#244e88;margin:8px 0}select,input{font:inherit;padding:8px;border:1px solid #bdc8d7;border-radius:6px;max-width:100%}.filters{display:flex;gap:12px;flex-wrap:wrap;align-items:center}table{border-collapse:collapse}th,td{padding:8px 16px;border-bottom:1px solid #dce2eb;text-align:left}
[hidden]{display:none!important}@media(max-width:850px){.columns{grid-template-columns:1fr}body{padding:12px}article,header{padding:16px}}
</style><header><h1>DPO 逐题对照</h1><p>LoRA SFT lora-v1 → DPO dpo-v2 · 固定贪心生成 · EvalPlus 原始测试与新增测试均通过才算通过。</p>
<p>仅分析已有结果。人工分类描述失败现象，不证明训练根因。空失败列表不代表没有失败，也不能据此认定是运行超时。JSON 中的元组可能显示为数组，不据此推断原始测试参数类型。</p>
<table><tr><th>评测集</th><th>退化</th><th>改善</th><th>均通过</th><th>均失败</th></tr>''' + rows + '''</table>
<p>推荐先看退化题，再切换改善题。原始 completion 是模型输出；solution 是被评分的完整代码。题目只供诊断，不加入训练。</p>
<div class="filters"><label>变化 <select id="kind"><option value="lost">退化</option><option value="gained">改善</option><option value="">全部变化</option></select></label>
<label>题库 <select id="suite"><option value="">全部</option><option value="humaneval">HumanEval+</option><option value="mbpp">MBPP+</option></select></label>
<label>分类 <select id="category"><option value="">全部</option>''' + options + '''</select></label>
<label>搜索 <input id="query" placeholder="题号、函数名或代码关键词"></label><span id="shown" aria-live="polite"></span></div></header>''' + "".join(cards) + '''
<script>
const cards=Array.from(document.querySelectorAll('article'));
const controls=['kind','suite','category','query'].map(id=>document.getElementById(id));
function filter(){const [kind,suite,category,query]=controls.map(el=>el.value);let shown=0;
for(const card of cards){const match=(!kind||card.dataset.kind===kind)&&(!suite||card.dataset.suite===suite)&&(!category||card.dataset.category===category)&&(!query||card.textContent.toLowerCase().includes(query.toLowerCase()));card.hidden=!match;if(match)shown++;}
document.getElementById('shown').textContent=`显示 ${shown} / ${cards.length} 道变化题`;}
controls.forEach(el=>el.addEventListener('input',filter));filter();
function revealAnchor(){const card=document.getElementById(location.hash.slice(1));if(!card||card.tagName!=='ARTICLE')return;
controls[0].value=card.dataset.kind;controls[1].value='';controls[2].value='';controls[3].value='';filter();card.scrollIntoView();}
window.addEventListener('hashchange',revealAnchor);revealAnchor();
</script></html>'''


def main() -> None:
    """校验输入后一次性导出 cases.json、index.html、report.md，拒绝覆盖已有分析。

    输入路径固定为本次已经完成的两次评测；只提供输出目录参数，便于重新运行验证。
    不引入模型加载、自动修复或额外生成，因此这一步可用普通 Python 3 在本地完成。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/dpo-v2-vs-lora-v1")
    args = parser.parse_args()
    tasks = read_jsonl(ROOT / "eval.jsonl")
    lock = json.loads((ROOT / "eval.lock.json").read_text(encoding="utf-8"))
    if sha256(ROOT / "eval.jsonl") != lock["eval_jsonl_sha256"]:
        raise ValueError("eval.jsonl 与锁不一致")
    source = ROOT / "results/lora_sft/lora-v1"
    target = ROOT / "results/dpo/dpo-v2"
    before, before_counts = load_run(source, "lora_sft", tasks)
    after, after_counts = load_run(target, "dpo", tasks)
    for field in ("device", "torch", "transformers", "python", "prompt_format"):
        configs = [json.loads((folder / "config.json").read_text(encoding="utf-8")) for folder in (source, target)]
        if configs[0][field] != configs[1][field]:
            raise ValueError(f"两阶段评测环境/协议不同：{field}")
    review_path = ROOT / "docs/dpo-case-review.json"
    reviews = json.loads(review_path.read_text(encoding="utf-8"))
    cases, counts = build_cases(tasks, before, after, reviews)
    inputs = [ROOT / "eval.jsonl", ROOT / "eval.lock.json", review_path, Path(__file__)]
    for folder in (source, target):
        inputs += [folder / name for name in ("config.json", "summary.json")]
        for suite in ("humaneval", "mbpp"):
            inputs += [folder / f"{suite}.{suffix}" for suffix in ("samples.jsonl", "samples_eval_results.json")]
    manifest = {"sources_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
                "counts": counts, "before": before_counts, "after": after_counts,
                "review_method": "人工静态阅读；未重跑生成代码；不作为训练根因证据", "cases": cases}
    lines = ["# DPO 逐题审阅", "", "打开 index.html 查看并排代码；cases.json 保留原始证据及来源哈希。", "",
             "所有人工分类为静态代码审阅，没有重跑测试。未知 token 数，不断言达到 512 上限。", "",
             "| 题库 | 退化 | 改善 | 均通过 | 均失败 |", "| --- | ---: | ---: | ---: | ---: |"]
    for suite, count in counts.items():
        lines.append(f"| {suite} | " + " | ".join(str(count[key]) for key in LABELS) + " |")
    lines += ["", "## 逐题阅读索引", "", "| 题目 | 变化 | 人工分类 | 说明 |", "| --- | --- | --- | --- |"]
    for row in cases:
        review = row["review"] or {"category": "未分类", "note": "请阅读页面中的代码与判分。"}
        anchor = row["task_id"].replace("/", "-")
        note = review["note"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| [{row['task_id']}](index.html#{anchor}) | {LABELS[row['transition']]} | {review['category']} | {note} |")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cases.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "index.html").write_text(render_html(cases, counts), encoding="utf-8")
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": counts, "changed_cases": len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
