"""Render the one-page competition entry using only measured results."""

from prompt_templates_v2 import TASKS
from v2_features import ROOT


def write_readme(result, card):
    pct = lambda x: f"{100 * x:.1f}%"
    lines = [
        "# MarketFlow · DRIFT-TSLM",
        "",
        "At a historical market close, which stocks drive a target’s model signal, and what happens if one incoming edge is removed? MarketFlow lets a judge select a stock, mask an edge, see DRIFT’s exact score change and read the language model’s independently generated explanation.",
        "",
        "![MarketFlow historical edge masking and Qwen answer](reports/v2_demo.png)",
        "",
        "**Run the demo** · `npm --prefix web run build`, then serve with `python3 -m http.server 8000 --directory web/dist`. The downloadable site includes 18 stock-days, four tasks and all 90 single-edge interventions. Optional GPU inference uses `bash web/serve.sh`. The original DRIFT bundle still runs with `./run.sh verify` and `./run.sh load --cell alpha158us_sp500_pit`.",
        "",
        "**Multi-Day Replay** · Replay two upstream edges over five sessions for 28 selected stock-days and inspect the residual score change after an edge reopens. [S&P 500 replay](reports/v2_multiday_sp500.png) · [CSI 300 replay](reports/v2_multiday_csi300.png)",
        "",
        "**Model** · 20 sessions × 157 Alpha158 factors → frozen seed-0 DRIFT → six projectors → Qwen3.5-4B with 32-layer LoRA. Stock names sit beside their numeric tokens (25 for single-stock questions, 20 for pairs). The prompt contains tickers, dates and the question. Training used 24,000 examples, 750 alignment steps and 5,000 joint steps with 3× weight on the answer. Validation selected step "
        + str(result["validation"]["drift"]["step"])
        + ".",
        "",
        "### 1. Held-out questions",
        "",
        "Each task contains 1,000 questions over 500 trading days in 2024–2025. Exact accuracy requires a valid explanation format and every JSON field correct. This revision reused the previously reported test set and selected weights on 2023 validation. The returns baseline retains its original 3,000-step budget.",
        "",
        "| Model | Attribution | Masked edge | Role & regime | Pair, both fields |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, v, m in [
        ("Zero-shot Qwen", "text", "text"),
        ("Returns-only OpenTSLM + Qwen", "returns", "real"),
        ("DRIFT + Qwen", "drift", "real"),
    ]:
        lines.append(
            "| "
            + " | ".join(
                [name, *[pct(result["variants"][v][m][t]["accuracy"]) for t in TASKS]]
            )
            + " |"
        )
    lines += [
        "",
        "### 2. Grounding test · " + result["gate"],
        "",
        "Every diagnostic task must lose at least 15 percentage points against both controls. Same-day mismatches preserve the observation date.",
        "",
        "| Task | Real | Mismatched | Zero | Real − mismatch / zero |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, t in zip(
        ("Attribution", "Masked edge", "Role & regime"), TASKS[:3], strict=True
    ):
        v = result["variants"]["drift"]
        drop = result["drops_pp"][t]
        lines.append(
            f"| {name} | {pct(v['real'][t]['accuracy'])} | {pct(v['shuffled'][t]['accuracy'])} | {pct(v['zero'][t]['accuracy'])} | {drop['shuffled']:+.1f} / {drop['zero']:+.1f} pp |"
        )
    lines += [
        "",
        "### 3. Which stock had the higher future return?",
        "",
        "The native reference ranks each pair by the four-seed DRIFT ensemble. The Qwen margin is the A logit minus the B logit at the winner field after its own generated explanation. Each column uses the same 1,000 pairs.",
        "",
        "| Pair cohort | Native win / AUC | Qwen margin win / AUC | Generated winner |",
        "|---|---:|---:|---:|",
    ]
    for name, split, group in [
        ("Top vs bottom quintile", "test", "variants"),
        ("Whole-market random", "test_random_pairs", "random_pairs"),
    ]:
        x = result["pairwise_scores"][split]
        m = result[group]["drift"]["real"][TASKS[3]]
        lines.append(
            f"| {name} | {100*x['native']['win_rate']:.2f}% / {x['native']['auc']:.4f} | {100*x['logits']['win_rate']:.2f}% / {x['logits']['auc']:.4f} | {pct(m['fields']['winner'] / m['n'])} |"
        )
    lines += [
        "",
        "### 4. Paper stock-ranking benchmarks",
        "",
        "Daily cross-sectional IC and RankIC, with cumulative gross top-quintile portfolio return.",
        "",
        "| Alpha158 market | LightGBM IC / RankIC / return | DRIFT ensemble IC / RankIC / return |",
        "|---|---:|---:|",
    ]
    for tab in card["tables"][:2]:
        models = {r[0]: r[1:] for r in tab["rows"]}
        lines.append(
            "| "
            + " | ".join(
                [
                    tab["market"],
                    " / ".join(models["LightGBM"]),
                    " / ".join(models["DRIFT"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "**TimeNet** · `10eit/drift-alpha158-flow-public` downloads a pinned price snapshot without an account. Its 98 matched observations cover 30 target stocks and 11 sectors, with four tasks per record. The original research-provider edition has 240 observations. [Connector and build](data/prompt_data/timenet_connectors/datasets/10eit/drift_alpha158_flow_public/README.md)",
        "",
        "Exact interventions refer to the frozen model. The interface replays historical dates. Full field accuracy, ablations and three paired examples are in [the results](result_9.md). [Model card screenshot](reports/v2_model_card.png) · [Static demo package](reports/marketflow-static.zip) · [GPU and web commands](web/README.md)",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(lines))
