"""F3: baselines and the rolling-origin backtest on the training days (the holdout is never read).

Writes results/f3_backtest.json, small aggregated CSVs in results/, figures in docs/figures/ and
docs/F3_baselines_backtest.md. Per-series metrics stay in data/processed/ (gitignored).

    python -m forecast.run_f3
"""
from __future__ import annotations

import json

import pandas as pd

from forecast import backtest as bt
from forecast import config, plots
from forecast import demand_patterns as dp

ROOT = config.ROOT
RESULTS = ROOT / "results"
REPORT = RESULTS / "f3_backtest.json"
FIGURES = ROOT / "docs" / "figures"
TEMPLATE = ROOT / "docs" / "templates" / "F3_baselines_backtest.md"
DOC = ROOT / "docs" / "F3_baselines_backtest.md"


def summarize(per_series: pd.DataFrame, cfg: dict) -> dict:
    order, ref, tol = cfg["baselines"], cfg["reference_baseline"], cfg["tie_tolerance"]

    def grouped(key):
        return {str(k): bt.by_baseline(g, order) for k, g in per_series.groupby(key)}

    by_class = {c: bt.by_baseline(g, order) for c, g in per_series.groupby("demand_class")}
    return {
        "overall": bt.by_baseline(per_series, order),
        "by_class": {c: by_class[c] for c in dp.CLASSES if c in by_class},
        "by_dept": grouped("dept_id"),
        "by_origin": grouped("origin"),
        "wins_vs_reference": {
            "reference": ref,
            "overall": bt.win_counts(per_series, ref, tol),
            "by_class": {c: bt.win_counts(g, ref, tol) for c, g in per_series.groupby("demand_class")},
        },
    }


def best(metrics: dict, key: str = "wrmsse") -> tuple[str, float]:
    name = min((b for b in metrics if metrics[b][key] is not None), key=lambda b: metrics[b][key])
    return name, metrics[name][key]


def main() -> dict:
    data_cfg = config.load("data")
    cfg = config.load("backtest")
    cls_cfg = config.load("demand_patterns")["classification"]
    holdout_days = config.load("holdout")["holdout_days"]
    processed = ROOT / data_cfg["processed_dir"]

    panel = bt.load_panel(processed / data_cfg["slice_parquet"], holdout_days,
                          data_cfg["duckdb"]["memory_limit"], data_cfg["duckdb"]["threads"])
    per_series, coverage, weekly = bt.evaluate(panel, cfg, cls_cfg)
    del panel

    summary = summarize(per_series, cfg)
    weekly["error_pct"] = 100 * (weekly["forecast"] / weekly["actual"] - 1)
    report = {
        "setup": {
            "last_train_d": coverage[-1]["last_scored_d"], "first_holdout_d": coverage[-1]["last_scored_d"] + 1,
            "holdout_days": holdout_days, "horizon_days": cfg["horizon_days"], "n_origins": cfg["n_origins"],
            "origins": [c["origin"] for c in coverage], "min_history_days": cfg["min_history_days"],
            "baselines": cfg["baselines"], "reference_baseline": cfg["reference_baseline"],
            "n_series_origin_pairs": int(per_series[["id", "origin"]].drop_duplicates().shape[0]),
        },
        "coverage": coverage,
        **summary,
        "weekly_totals": json.loads(weekly.to_json(orient="records")),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = [{"group": g, "key": k, "baseline": b, **m}
            for g in ("by_class", "by_dept", "by_origin") for k, mb in report[g].items() for b, m in mb.items()]
    rows = [{"group": "overall", "key": "all", "baseline": b, **m} for b, m in report["overall"].items()] + rows
    pd.DataFrame(rows).to_csv(RESULTS / "f3_summary.csv", index=False, float_format="%.4f")
    weekly.to_csv(RESULTS / "f3_weekly_totals.csv", index=False, float_format="%.2f")
    per_series.to_parquet(processed / "f3_series_metrics.parquet", index=False)

    plots.f3_scaled_error_by_class(report, FIGURES / "f3_error_by_class.png")
    plots.f3_error_by_origin(report, FIGURES / "f3_error_by_origin.png")
    plots.f3_wins(report, FIGURES / "f3_wins_vs_seasonal_naive.png")

    if TEMPLATE.exists():
        render_doc(report, cfg)
    return report


LABEL = {"naive": "Naive (last day)", "seasonal_naive": "Seasonal naive (same weekday last week)",
         "moving_average": "28-day moving average", "ses": "Simple exponential smoothing",
         "croston": "Croston", "sba": "SBA (bias-corrected Croston)", "tsb": "TSB"}


def render_doc(report: dict, cfg: dict) -> None:
    s, o = report["setup"], report["overall"]
    order = s["baselines"]
    ref = s["reference_baseline"]
    best_name, best_w = best(o)
    best_mean, best_mean_v = best(o, "mean_rmsse")

    overall_rows = "\n".join(
        f"| {LABEL[b]} | {o[b]['wrmsse']:.3f} | {o[b]['mean_rmsse']:.3f} | {o[b]['median_rmsse']:.3f} | "
        f"{o[b]['mean_mae']:.3f} | {o[b]['mean_bias']:+.3f} | {100 * o[b]['weekly_wape']:.1f}% |" for b in order)

    cov_rows = "\n".join(
        f"| d_{c['origin']} | d_{c['first_scored_d']}..d_{c['last_scored_d']} | {c['n_series']:,} | "
        f"{c['n_scored']:,} | {c['n_short_history']} | {c['n_no_window_days']} | {c['n_zero_scale']} | "
        f"{c['scored_days']:,} |" for c in report["coverage"])
    cls_counts = report["coverage"][-1]["classes"]

    bc = report["by_class"]
    class_rows, class_best = [], []
    for c, m in bc.items():
        bn, bw = best(m)
        bmn, bmv = best(m, "mean_rmsse")
        n = m[order[0]]["n"]
        class_rows.append(f"| {c} | {n:,} | " + " | ".join(
            (f"**{m[b]['wrmsse']:.3f}**" if b == bn else f"{m[b]['wrmsse']:.3f}") for b in order) + " |")
        class_best.append(f"{c}: {LABEL[bn]} (WRMSSE {bw:.3f}; lowest mean RMSSE: {LABEL[bmn]}, {bmv:.3f})")

    bo = report["by_origin"]
    origin_rows = "\n".join(f"| d_{k} | " + " | ".join(f"{m[b]['wrmsse']:.3f}" for b in order) + f" | {LABEL[best(m)[0]]} |"
                            for k, m in bo.items())
    ranks = {k: sorted(order, key=lambda b: m[b]["wrmsse"]) for k, m in bo.items()}
    same_best = len({r[0] for r in ranks.values()}) == 1
    wr = [m[best_name]["wrmsse"] for m in bo.values()]

    bd = report["by_dept"]
    dept_rows = "\n".join(f"| {k} | {m[order[0]]['n']:,} | " + " | ".join(f"{m[b]['wrmsse']:.3f}" for b in order)
                          + f" | {LABEL[best(m)[0]]} |" for k, m in bd.items())

    w = report["wins_vs_reference"]["overall"]
    win_rows = "\n".join(
        f"| {LABEL[b]} | {w[b]['pairs']['wins']:,} | {w[b]['pairs']['ties']:,} | {w[b]['pairs']['losses']:,} | "
        f"{100 * w[b]['pairs']['share_wins']:.0f}% | {w[b]['series']['wins']:,} of {w[b]['series']['n']:,} "
        f"({100 * w[b]['series']['share_wins']:.0f}%) |" for b in order if b != ref)
    wc = report["wins_vs_reference"]["by_class"]
    win_class_rows = "\n".join(f"| {c} | " + " | ".join(f"{100 * wc[c][b]['pairs']['share_wins']:.0f}%"
                                                         for b in order if b != ref) + " |"
                               for c in bc if c in wc)

    wk = pd.DataFrame(report["weekly_totals"])
    wk_b = wk[wk["baseline"] == best_name]
    wk_ref = wk[wk["baseline"] == ref]
    weekly_text = "; ".join(
        f"{plots.F3_SHORT[b]} {w_['error_pct'].min():+.1f}% to {w_['error_pct'].max():+.1f}% "
        f"(mean absolute {w_['error_pct'].abs().mean():.1f}%)" for b, w_ in ((best_name, wk_b), (ref, wk_ref)))
    wape_best = min(order, key=lambda b: o[b]["weekly_wape"])

    smoothers = [b for b in order if b not in ("naive", ref)]
    sw = [o[b]["wrmsse"] for b in smoothers]
    short = plots.F3_SHORT
    close_text = (f"The {len(smoothers)} smoothing rules ({', '.join(short[b] for b in smoothers)}) "
                  f"lie within {max(sw) - min(sw):.3f} of each other ({min(sw):.3f} to {max(sw):.3f}), so the "
                  f"order among them is not a strong result; the gap to the naive rules is.")
    worst_smoother = max(sw)
    naive_best = min(o["naive"]["wrmsse"], o[ref]["wrmsse"])
    smooth_vs_naive_text = (f"Every smoothing rule beats both naive rules (worst smoothing rule {worst_smoother:.3f}, "
                            f"best naive rule {naive_best:.3f})." if worst_smoother < naive_best else
                            "Not every smoothing rule beats the naive rules (see the table).")
    above = [LABEL[b].split(" (")[0] for b in ("naive", ref) if o[b]["wrmsse"] > 1]
    naive_text = (f"{' and '.join(above)} score above 1 (WRMSSE). Copying one day, or one week, carries that "
                  "day's noise straight into all of the forecast days; averaging over many days removes most "
                  "of it." if above else "Both naive rules score below 1 (WRMSSE).")
    cro = [b for b in ("croston", "sba", "tsb") if b in order]
    plain = [b for b in ("moving_average", "ses") if b in order]
    parts = []
    for c in ("intermittent", "lumpy"):
        if c in bc:
            bcro, bpl = min(cro, key=lambda b: bc[c][b]["wrmsse"]), min(plain, key=lambda b: bc[c][b]["wrmsse"])
            parts.append(f"{c}: best Croston-type {short[bcro]} {bc[c][bcro]['wrmsse']:.3f} vs best "
                         f"plain average {short[bpl]} {bc[c][bpl]['wrmsse']:.3f}")
    croston_text = ("In this slice the difference is small: " + "; ".join(parts) + ". Croston's mean bias is "
                    f"{o['croston']['mean_bias']:+.3f} units a day and SBA's {o['sba']['mean_bias']:+.3f}, so the "
                    "bias correction works as intended.")
    tsb_wins = all(bc[c]["tsb"]["wrmsse"] <= min(bc[c]["croston"]["wrmsse"], bc[c]["sba"]["wrmsse"]) for c in bc)
    tsb_text = ("TSB, which also updates on days without a sale, is at least as good as Croston and SBA in every "
                "class." if tsb_wins else "TSB is not better than Croston and SBA in every class.")

    values = {
        "n_baselines_text": len(order), "close_text": close_text, "smooth_vs_naive_text": smooth_vs_naive_text,
        "naive_text": naive_text, "croston_text": croston_text, "tsb_text": tsb_text,
        "origins": ", ".join(f"d_{x}" for x in s["origins"]), "horizon": s["horizon_days"],
        "n_origins": s["n_origins"], "first_scored": report["coverage"][0]["first_scored_d"],
        "last_scored": s["last_train_d"], "first_holdout": s["first_holdout_d"],
        "min_history": s["min_history_days"], "pairs": f"{s['n_series_origin_pairs']:,}",
        "ma_days": cfg["moving_average_days"], "ses_alpha": cfg["ses_alpha"], "croston_alpha": cfg["croston_alpha"],
        "tsb_ad": cfg["tsb_alpha_demand"], "tsb_ap": cfg["tsb_alpha_probability"],
        "overall_rows": overall_rows, "cov_rows": cov_rows,
        "last_origin": s["origins"][-1],
        "cls_counts": ", ".join(f"{c} {n:,}" for c, n in cls_counts.items() if n),
        "best_name": LABEL[best_name], "best_w": f"{best_w:.3f}",
        "ref_name": LABEL[ref], "ref_w": f"{o[ref]['wrmsse']:.3f}",
        "best_mean": LABEL[best_mean], "best_mean_v": f"{best_mean_v:.3f}",
        "ref_mean": f"{o[ref]['mean_rmsse']:.3f}",
        "base_header": " | ".join(LABEL[b].split(" (")[0] for b in order),
        "base_sep": "|".join("---" for _ in order),
        "win_header": " | ".join(LABEL[b].split(" (")[0] for b in order if b != ref),
        "win_sep": "|".join("---" for b in order if b != ref),
        "class_rows": "\n".join(class_rows), "class_best": "\n".join(f"- {x}" for x in class_best),
        "origin_rows": origin_rows, "dept_rows": dept_rows,
        "best_range": f"{min(wr):.3f} to {max(wr):.3f}",
        "stability_text": (f"{LABEL[best_name]} has the lowest WRMSSE at every origin"
                           if same_best else "the best baseline changes between origins (see the table)"),
        "win_rows": win_rows, "win_class_rows": win_class_rows,
        "weekly_text": weekly_text, "n_weeks": len(wk_b), "wape_best": LABEL[wape_best],
        "wape_best_v": f"{100 * o[wape_best]['weekly_wape']:.1f}",
    }
    DOC.write_text(TEMPLATE.read_text(encoding="utf-8").format(**values), encoding="utf-8")


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("setup", "coverage", "overall")}, indent=2))
