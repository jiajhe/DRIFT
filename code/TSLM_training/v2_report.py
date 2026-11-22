"""Summarize the four requested comparisons from locked predictions, with no imputed scores."""

import hashlib
import json
import re

import numpy as np
from v2_features import ROOT, BASE
from prompt_templates_v2 import TASKS, user_parts, evidence

RUN = ROOT / "runs/drift-tslm-v2.1"
BASELINES = ROOT / "runs/drift-tslm-v2"


def folder(variant):
    return (RUN if variant == "drift" else BASELINES) / variant


CASES = (("2024-01-02", 0), ("2024-07-01", 1), ("2025-12-29", 2))


def read(p):
    return json.loads(p.read_text())


def evidence_check(row, outputs):
    def sentences(text):
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if sentence.strip()
        ]

    with np.load(BASE / (row["date"] + ".npz"), allow_pickle=False) as day:
        supported = {
            sentence
            for variant in range(9)
            for sentence in sentences(
                evidence(
                    row["task"], row["answer"], day, row["i"], row.get("b"), variant
                )
            )
        }
    review = {}
    for model, output in outputs.items():
        body = output.partition("<reasoning>")[2].partition("</reasoning>")[0]
        claims = sentences(body)
        review[model] = dict(
            supported=sum(s in supported for s in claims),
            total=len(claims),
            other_wording=[s for s in claims if s not in supported],
        )
    return review


def main():
    variants = {}
    random = {}
    for v, modes in [
        ("drift", ["real", "shuffled", "zero", "local"]),
        ("returns", ["real"]),
        ("text", ["text"]),
    ]:
        variants[v] = {m: read(folder(v) / ("test-" + m + ".json")) for m in modes}
        random[v] = {
            m: read(folder(v) / ("test_random_pairs-" + m + ".json")) for m in modes
        }
        for m in modes:
            if random[v][m][TASKS[3]]["n"] != 1000:
                raise ValueError("Incomplete whole-market pair evaluation")
            if any(variants[v][m][t]["n"] != 1000 for t in TASKS):
                raise ValueError("Incomplete held-out task")
    real = variants["drift"]["real"]
    differences = {
        t: {
            m: round(100 * (real[t]["accuracy"] - variants["drift"][m][t]["accuracy"]), 6)
            for m in ("shuffled", "zero", "local")
        }
        for t in TASKS
    }
    grounded = all(
        differences[t][m] >= 15 - 1e-9 for t in TASKS[:3] for m in ("zero", "shuffled")
    )
    beats = all(
        real[t]["accuracy"]
        > max(
            variants["returns"]["real"][t]["accuracy"],
            variants["text"]["text"][t]["accuracy"],
        )
        for t in TASKS
    )
    gate = "passed" if grounded else "failed"
    xs = [json.loads(x) for x in (BASE / "tasks/test.jsonl").read_text().splitlines()]
    pred = {
        k: [
            json.loads(x)
            for x in (folder(k) / "test-real.jsonl").read_text().splitlines()
        ]
        for k in ("drift", "returns")
    }
    cases = []
    for day, t in CASES:
        i = next(j for j, r in enumerate(xs) if r["date"] == day and r["task"] == t)
        r = xs[i]
        for k, rows in pred.items():
            assert (
                rows[i]["date"],
                rows[i]["i"],
                rows[i]["task"],
                rows[i]["answer"],
            ) == (day, r["i"], t, r["answer"])
        cases.append(
            dict(
                date=day,
                target=r["ticker"],
                task=TASKS[t],
                question=user_parts(r)[1].strip(),
                drift=pred["drift"][i]["text"],
                baseline=pred["returns"][i]["text"],
                gold=r["answer"],
                reference_evidence=r["evidence"],
                evidence_check=evidence_check(
                    r, {k: rows[i]["text"] for k, rows in pred.items()}
                ),
            )
        )
    selected = {}
    for variant in ("drift", "returns"):
        by_step = {}
        for step in (3000, 4000, 5000) if variant == "drift" else (1500, 2250, 3000):
            metrics = read(folder(variant) / f"valid-{step}.json")
            if any(metrics[t]["n"] != 500 for t in TASKS):
                raise ValueError("Incomplete validation checkpoint")
            by_step[step] = sum(metrics[t]["accuracy"] for t in TASKS) / len(TASKS)
        selected[variant] = dict(
            step=max(by_step, key=by_step.get), accuracy_by_step=by_step
        )
    with (ROOT / "checkpoints/TSLM_training/v2.1/drift/best_valid.pt").open("rb") as f:
        weights_sha = hashlib.file_digest(f, "sha256").hexdigest()
    result = dict(
        model_revision="v2.1",
        checkpoint_sha256=weights_sha,
        protocol=dict(
            training_steps={"alignment": 750, "joint": 5000},
            forecast_target="close(t+2) / close(t+1) - 1",
            component_normalization="train-only, separate mean and scale for each [3,18] slot",
            local_width=72,
            answer_token_weight=3.0,
            pairwise_readout="BF16 next-token logits at the winner field; equal logits count as half a win",
            baseline_joint_steps=3000,
            stats_sha256=hashlib.sha256((BASE / "train_stats.json").read_bytes()).hexdigest(),
            task_sha256={
                name: hashlib.sha256((BASE / "tasks" / (name + ".jsonl")).read_bytes()).hexdigest()
                for name in ("train", "valid", "test", "test_random_pairs")
            },
            validation_selection="highest 2023 four-task mean exact accuracy; predeclared steps",
            test_reuse="same 2024–2025 questions previously reported for v2",
        ),
        pairwise_scores=read(RUN / "drift/pairwise_scores.json"),
        previous_v2={
            m: read(BASELINES / "drift" / f"test-{m}.json")
            for m in ("real", "shuffled", "zero")
        },
        validation=selected,
        gate=gate,
        beats_both_on_all_tasks=beats,
        validation_selected=True,
        variants=variants,
        random_pairs=random,
        drops_pp=differences,
        examples=cases,
    )
    out = ROOT / "reports/v2_evaluation.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(out)
    card_path = ROOT / "web/model-card.json"
    card = read(card_path)
    card["evaluation"] = result
    card_path.write_text(
        json.dumps(card, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    write_summary(result)
    from v2_judges_readme import write_readme

    write_readme(result, card)
    print("grounding", gate, "beats both on every task", beats)
    for t in TASKS:
        print(t, real[t]["accuracy"], differences[t])
    return result


def write_summary(result):
    path = ROOT / "result_9.md"
    start = "<!-- DRIFT-TSLM v2 -->"
    end = "<!-- end DRIFT-TSLM v2 -->"
    original = path.read_text()
    head, blank, old = original.partition("\n")
    if start in old:
        before, current = old.split(start, 1)
        _, after = current.split(end, 1)
        old = before + after
    old = old.lstrip()
    if old.startswith("## v1 历史记录"):
        old = old.partition("\n")[2].lstrip()
    modes = [
        ("纯文本零样本 Qwen", "text", "text"),
        ("仅 20 日收益率的 OpenTSLM + Qwen", "returns", "real"),
        ("DRIFT-TSLM v2", "drift", "real"),
        ("同日错配 DRIFT", "drift", "shuffled"),
        ("全零 DRIFT", "drift", "zero"),
        ("仅本地时序、Z 和局部分数", "drift", "local"),
    ]

    def pct(x):
        return f"{100 * x:.1f}%"

    lines = [
        start,
        "## DRIFT-TSLM v2 · 锁定测试结果",
        "",
        "结构来自论文 seed 0，四种子为 0、2、5、9。主模型读取目标和上游的 20 日、157 因子轨迹与定向流。单股票 25 个数值 token，双股票 20 个。Qwen3.5-4B 使用 750 步投影对齐和 5,000 步全层 LoRA。JSON 区间按三倍权重训练。local 槽位 72 维，含当日四种子集成分数和 3 个按训练集标准化的占比。三个流分量各自拟合 18 维尺度。收益率基线保留原 3,000 步等权 SFT，在同一批测试题上比较。纯文本为零样本。三者都采用自由贪心生成，上限 168 个新 token。",
        "",
        "训练 2018-01-02 至 2022-12-28（1,257 日、24,000 题），验证 2023-01-03 至 2023-12-27（248 日、2,000 题）。主测试 2024-01-02 至 2025-12-29，四任务各 1,000 题，各覆盖全部 500 个测试日。各段均剔除了最后两个交易日的观察样本。本轮方案是在已看到上一版测试结果后制定的，同一测试集用于复测，权重只按 2023 验证集选择。下表要求说明含 2–4 句、JSON 合法且所有字段正确。置信度衡量模型分数差相对历史阈值的大小。",
        "",
        "| 输入 | 信号归因 | 屏蔽单边 | 角色与状态 | 前后五分位配对 | 全市场随机配对（另 1,000 题） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    selection = []
    for variant, name in (("drift", "DRIFT"), ("returns", "收益率基线")):
        val = result["validation"][variant]
        values = " / ".join(
            f"{100 * score:.2f}%" for score in val["accuracy_by_step"].values()
        )
        selection.append(
            f"{name}（第 {' / '.join(map(str, val['accuracy_by_step']))} 步）{values}，选中第 {val['step']} 步"
        )
    lines[7:7] = [
        "各预定验证点的四任务整题正确率均值为 " + "。".join(selection) + "。",
        "",
    ]
    for name, v, m in modes:
        task = result["variants"][v][m]
        rand = result["random_pairs"][v][m][TASKS[3]]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    *[pct(task[t]["accuracy"]) for t in TASKS],
                    pct(rand["accuracy"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "同一批题的旧 v2 / 本轮整题正确率依次为 "
        + "，".join(
            f"{name} {pct(result['previous_v2']['real'][task]['accuracy'])} / {pct(result['variants']['drift']['real'][task]['accuracy'])}"
            for name, task in zip(
                ("归因", "反事实", "状态", "配对"), TASKS, strict=True
            )
        )
        + "。",
    ]
    lines.extend(
        [
            "",
            "分字段正确率要求输出格式合法。以下两项胜者准确率只检查两只股票中谁的真实未来收益更高。",
            "",
            "| 输入 | 驱动 / 流模式 / 上游 | 位移 / 敏感度 | 净角色 / 稳定性 / 市场状态 | 胜者 / 分数差分档 | 随机配对胜者 / 分数差分档 |",
            "|---|---|---|---|---|---|",
        ]
    )
    keys = (
        ("driver", "flow_mode", "top_sender"),
        ("score_shift", "sensitivity"),
        ("net_role", "role_stability", "market_regime"),
        ("winner", "confidence"),
    )

    def fields(metric, names):
        return " / ".join(pct(metric["fields"][k] / metric["n"]) for k in names)

    for name, v, m in modes:
        task = result["variants"][v][m]
        rand = result["random_pairs"][v][m][TASKS[3]]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    *[fields(task[t], ks) for t, ks in zip(TASKS, keys, strict=True)],
                    fields(rand, keys[3]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "符合输出格式的题数，每列分母均为 1,000。分字段正确率以全部题目为分母，格式无效的回答计错。",
        "",
        "| 输入 | 归因 | 反事实 | 角色与状态 | 极端配对 | 随机配对 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, v, m in modes:
        main = result["variants"][v][m]
        rand = result["random_pairs"][v][m][TASKS[3]]
        lines.append(
            "| "
            + " | ".join(
                [name, *[str(main[t]["format"]) for t in TASKS], str(rand["format"])]
            )
            + " |"
        )
    gate = "通过" if result["gate"] == "passed" else "未通过"
    lines += [
        "",
        f"**数值证据门控{gate}。** 真实输入减去同日错配、全零输入的正确率差（百分点）依次为 "
        + "，".join(
            f"{name} {result['drops_pp'][task]['shuffled']:+.1f} / {result['drops_pp'][task]['zero']:+.1f}"
            for task, name in zip(TASKS[:3], ("归因", "反事实", "状态"), strict=True)
        )
        + "。每项的两次差值均须至少达到 15 个百分点。",
        "",
        "四种子在本次 history-v3 的 247,304 个有标签股票日取得 IC 0.01687、RankIC 0.01808。对齐同一批 246,814 个历史股票日后，v3 回放为 0.01661、0.01795，原回放为 0.01672、0.01809。v3 多覆盖 490 个股票日。两次回放在共同股票上的逐日分数相关系数均值为 0.99985。v2 的股票预测任务给出配对胜者，这次训练没有产生新的逐股连续排序 IC。",
        "",
        "### 论文基准",
        "",
        "每格依次给出 IC、RankIC、等权做多头部五分位组合的累计毛收益。",
        "",
        "| 模型 | S&P 500 Alpha158 | CSI 300 Alpha158 |",
        "|---|---|---|",
    ]
    markets = read(ROOT / "web/model-card.json")["tables"][:2]
    for a, b in zip(markets[0]["rows"], markets[1]["rows"], strict=True):
        assert a[0] == b[0]
        lines.append(
            "| " + " | ".join([a[0], " / ".join(a[1:]), " / ".join(b[1:])]) + " |"
        )
    lines += [
        "",
        "### 原生集成与胜者 logits",
        "",
        "每组 1,000 对。按同日 ens_score 的 A 减 B 得到原生基准。Qwen 先自行生成证据段，再在同一输入、同一证据段和固定 JSON 字段前缀后计算 A 减 B 的 logit。A 为正类，BF16 的两项 logits 相等时记半胜。两组的模型自生成说明均完整。",
        "",
        "| 配对 | DRIFT 胜率 / AUC | Qwen logit 胜率 / AUC | 平分数（DRIFT / Qwen） |",
        "|---|---:|---:|---:|",
    ]
    for split, x in result["pairwise_scores"].items():
        a, b = x["native"], x["logits"]
        lines.append(
            f"| {'极端配对' if split == 'test' else '全市场随机配对'} | {100*a['win_rate']:.2f}% / {a['auc']:.4f} | {100*b['win_rate']:.2f}% / {b['auc']:.4f} | {a['ties']} / {b['ties']} |"
        )
    lines += ["", "### 三个预先选定的历史案例", ""]
    for case in result["examples"]:
        lines += [
            f"**{case['date']} · {case['target']} · {case['task']}**",
            "",
            case["question"],
            "",
            "历史事实依据：" + case["reference_evidence"],
            "证据句逐字核对（匹配当日正确事实的 9 套句式，其他措辞保留供人工核对）："
            + "，".join(
                f"{name} {case['evidence_check'][key]['supported']}/{case['evidence_check'][key]['total']}"
                for name, key in (("收益率基线", "returns"), ("DRIFT", "drift"))
            ),
            "核对答案：" + json.dumps(case["gold"], separators=(",", ":")),
            "",
            "仅收益率模型的原文",
            "",
            "```text",
            case["baseline"],
            "```",
            "",
            "DRIFT-TSLM v2 的原文",
            "",
            "```text",
            case["drift"],
            "```",
            "",
        ]
    data = read(ROOT / "reports/v2_timenet.json")
    replay = read(BASE / "manifest.json")["summary"]
    cf_error = max(replay["full_replay_errors"].values())
    lines.extend(
        [
            "### 数据、验证与 Demo",
            "",
            f"history-v3 有 {replay['days']:,} 个交易日。三分量复原最大绝对误差 {replay['max_flow_error']:.3g}。"
            f"10 天抽查、每日三条边，快捷反事实相对完整回放的最大误差 {cf_error:.3g}。"
            "两层 GRU 各 64 维，拼接两层逐日隐状态得到 128 维。训练文字按训练分位数描述留存份额。真实历史收益和反事实标签不进入用户文本。",
            "",
            f"TimeNet 本地注册表含 {data['core_stocks']} 只核心股、{data['sectors']} 个行业、{data['records']} 个记录。"
            f"四类任务按训练 / 验证 / 测试划分为 600 / 120 / 240 条。注册表 {data['registry_bytes']:,} 字节。"
            f"SDK 回读后 {data['checked_factor_values']:,} 个目标因子值逐位相同，与 Qlib 最大绝对误差 {data['max_abs_factor_error']:.3g}。"
            "研究源和公开源使用各自的版本标识。公开子集含 98 个经价格核对的观察、30 只核心股与 392 道题。",
            "",
            f"模型卡与静态 Demo 使用同一测试结论 {result['gate']}。静态版固定 AAPL、MSFT、NVDA，覆盖 2024-01-02、01-03、07-01、07-02 和 2025-12-26、12-29 六个测试日。共 18 个股票日、162 个模型回答和 90 次精确单边干预。交互可切换日期、目标、配对股和被屏蔽的上游。历史归档结果附权重 SHA-256，与测试报告一致。",
            "",
            "公开行情来自 Kaggle 固定版本 613，98 个观测对应 156 / 68 / 168 道训练、验证、测试题。官方构建器匿名下载与本地回退两次构建的结果经过 SDK 逐项比较，678,160 个 float32 数值和全部 392 道题相同。原行情版与公开版各自保留独立的数据集 ID。",

            "",
        ]
    )
    rebuilt = head + "\n\n" + "\n".join(lines + [end]) + "\n\n## v1 历史记录\n\n" + old
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(rebuilt)
    tmp.replace(path)


if __name__ == "__main__":
    main()
