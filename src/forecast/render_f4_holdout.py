"""F4 part B (write-up): render docs/F4_holdout.md from saved results.

    python -m forecast.render_f4_holdout

Reads results/f4_holdout.json (written ONCE by forecast.run_f4_holdout), results/f4_backtest.json and
results/f1_data_report.json. It loads no data and no model and scores nothing; it only formats saved
numbers and draws one figure from them. The holdout is never re-run.
"""
from __future__ import annotations

import json

import pandas as pd

from forecast import config, plots
from forecast.demand_patterns import CLASSES
from forecast.run_f4 import LABEL, SHORT

ROOT = config.ROOT
RESULTS = ROOT / "results"
TEMPLATE = ROOT / "docs" / "templates" / "F4_holdout.md"
DOC = ROOT / "docs" / "F4_holdout.md"
FIGURE = ROOT / "docs" / "figures" / "f4_holdout_vs_backtest.png"


def pct(a: float, b: float) -> float:
    return 100 * (a / b - 1)


def values(h: dict, b: dict, f1: dict) -> dict:
    s, o, bo = h["setup"], h["overall"], b["overall"]
    order, name = s["methods"], "lgbm"
    ref = s["best_baseline_from_backtest"]
    sn = h["wins_vs_reference"]["reference"]
    baselines = [x for x in order if x != name]
    hold_best = min(baselines, key=lambda x: o[x]["wrmsse"])
    cov = h["coverage"][0]
    hr = f1["holdout_reserved"]
    final = h["model_final"]
    p = final["params"]

    wk = pd.DataFrame(h["weekly_totals"])
    wk["error_pct"] = 100 * (wk["forecast"] / wk["actual"] - 1)
    slice_bias = {m: pct(g["forecast"].sum(), g["actual"].sum()) for m, g in wk.groupby("baseline")}

    gap, gap_bt = pct(o[name]["wrmsse"], o[ref]["wrmsse"]), pct(bo[name]["wrmsse"], bo[ref]["wrmsse"])
    gap_sn = pct(o[name]["wrmsse"], o[sn]["wrmsse"])
    rank = sorted(order, key=lambda x: o[x]["wrmsse"])
    headline = (f"On the {hr['last_d'] - hr['first_d'] + 1} unseen days the global LightGBM model has a WRMSSE of "
                f"**{o[name]['wrmsse']:.3f}**, against **{o[ref]['wrmsse']:.3f}** for {LABEL[ref]} (the best "
                f"baseline in the backtest) and {o[sn]['wrmsse']:.3f} for seasonal naive: {abs(gap):.1f}% "
                f"{'lower' if gap < 0 else 'higher'} than {SHORT[ref]} (backtest: {abs(gap_bt):.1f}% "
                f"{'lower' if gap_bt < 0 else 'higher'}).")
    if hold_best == ref:
        best_text = f"{LABEL[ref]} is also the best baseline on the holdout."
    else:
        best_text = (f"On the holdout the best baseline is {SHORT[hold_best]} ({o[hold_best]['wrmsse']:.3f}), not "
                     f"{SHORT[ref]} ({o[ref]['wrmsse']:.3f}). The comparison baseline was fixed before the run "
                     f"(the best in the backtest), so {SHORT[ref]} stays the reference; against "
                     f"{SHORT[hold_best]} the model is {abs(pct(o[name]['wrmsse'], o[hold_best]['wrmsse'])):.1f}% "
                     f"{'lower' if o[name]['wrmsse'] < o[hold_best]['wrmsse'] else 'higher'}. The baselines are "
                     f"close together ({min(o[x]['wrmsse'] for x in baselines if x not in ('naive', sn)):.3f} to "
                     f"{max(o[x]['wrmsse'] for x in baselines if x not in ('naive', sn)):.3f} for the smoothing "
                     f"rules), as in F3.")

    overall_rows = "\n".join(
        f"| {'**' + LABEL[x] + '**' if x == name else LABEL[x]} | {o[x]['wrmsse']:.3f} | {bo[x]['wrmsse']:.3f} | "
        f"{o[x]['wrmsse'] - bo[x]['wrmsse']:+.3f} | {o[x]['mean_rmsse']:.3f} | {o[x]['mean_mae']:.3f} | "
        f"{o[x]['mean_bias']:+.3f} | {slice_bias[x]:+.1f}% | {100 * o[x]['weekly_wape']:.1f}% |" for x in order)

    bc, bbc = h["by_class"], b["by_class"]
    class_rows = "\n".join(
        f"| {c} | {m[name]['n']:,} | {m[name]['wrmsse']:.3f} | {m[ref]['wrmsse']:.3f} | {m[sn]['wrmsse']:.3f} | "
        f"{pct(m[name]['wrmsse'], m[ref]['wrmsse']):+.1f}% | "
        f"{pct(bbc[c][name]['wrmsse'], bbc[c][ref]['wrmsse']):+.1f}% |" for c, m in bc.items() if c in bbc)
    worse = [c for c, m in bc.items() if m[name]["wrmsse"] >= m[ref]["wrmsse"]]
    class_text = (f"The model is better than {SHORT[ref]} in every class, as in the backtest." if not worse
                  else f"The model is not better than {SHORT[ref]} for {', '.join(worse)}.")
    dept_rows = "\n".join(
        f"| {d} | {m[name]['n']:,} | {m[name]['wrmsse']:.3f} | {m[ref]['wrmsse']:.3f} | {m[sn]['wrmsse']:.3f} | "
        f"{pct(m[name]['wrmsse'], m[ref]['wrmsse']):+.1f}% |" for d, m in h["by_dept"].items())

    wr, wb = h["wins_vs_reference"], h["wins_vs_best_baseline"]

    def wline(x):
        return (f"{x['pairs']['wins']:,} | {x['pairs']['ties']:,} | {x['pairs']['losses']:,} | "
                f"{100 * x['pairs']['share_wins']:.0f}%")
    win_rows = (f"| vs {LABEL[sn]} | {wline(wr['overall'][name])} |\n"
                f"| vs {LABEL[ref]} (fixed reference) | {wline(wb['overall'])} |")
    win_class_rows = "\n".join(f"| {c} | {100 * w[name]['pairs']['share_wins']:.0f}% |"
                               for c, w in sorted(wr["by_class"].items(), key=lambda kv: CLASSES.index(kv[0])))

    def wk_cell(m, k):
        r = wk[(wk["baseline"] == m) & (wk["week"] == k)].iloc[0]
        return f"{r['forecast']:,.0f} ({r['error_pct']:+.1f}%)"
    weeks = sorted(wk["week"].unique())
    week_rows = "\n".join(
        f"| {k} | {wk[(wk['baseline'] == name) & (wk['week'] == k)].iloc[0]['actual']:,.0f} | "
        f"{wk_cell(name, k)} | {wk_cell(ref, k)} | {wk_cell(sn, k)} |" for k in weeks)
    wm = wk[wk["baseline"] == name]["error_pct"]
    wr_ = wk[wk["baseline"] == ref]["error_pct"]
    under = int((wm < 0).sum())

    better = [x for x in order if o[x]["wrmsse"] < bo[x]["wrmsse"]]
    easier_text = ((f"Every method scores better on the holdout than its backtest average" if len(better) == len(order)
                    else f"{len(better)} of {len(order)} methods score better on the holdout than their backtest average")
                   + ", so the level of the numbers moves with the period. The comparison is about the order "
                   "and the gaps between methods on the same days.")
    bt_bias = b.get("slice_total_bias_pct", {}).get(name)
    if slice_bias[name] < 0:
        bias_text = ("The model under-forecasts the slice total"
                     + (f", as in the backtest ({bt_bias:+.1f}%)" if bt_bias is not None and bt_bias < 0 else "")
                     + ". Per series it is more accurate, but F5 should size safety stock from the residuals, "
                     "not assume an unbiased forecast.")
    else:
        bias_text = "The model does not under-forecast the slice total on the holdout."

    return {
        "easier_text": easier_text, "bias_text": bias_text,
        "headline": headline, "best_text": best_text,
        "first_d": hr["first_d"], "last_d": hr["last_d"], "first_date": hr["first_date"], "last_date": hr["last_date"],
        "n_days": hr["last_d"] - hr["first_d"] + 1, "origin": s["forecast_origin"],
        "commit": h["commit"][:7], "commit_full": h["commit"], "started": h["started_utc"],
        "run_minutes": h["run_minutes"], "training_rows": f"{s['training_rows']:,}",
        "chosen": final["chosen_setting"], "leaves": p["num_leaves"], "lr": p["learning_rate"],
        "min_leaf": p["min_data_in_leaf"], "rounds": final["rounds"],
        "n_series": f"{cov['n_series']:,}", "n_scored": f"{cov['n_scored']:,}", "scored_days": f"{cov['scored_days']:,}",
        "classes": ", ".join(f"{c} {n:,}" for c, n in cov["classes"].items() if n),
        "overall_rows": overall_rows, "rank_text": (
            f"The model ranks first of {len(order)} methods on the holdout." if rank[0] == name
            else f"The model ranks {rank.index(name) + 1} of {len(order)} methods on the holdout."),
        "gap_sn": f"{abs(gap_sn):.1f}", "model_label": LABEL[name], "ref_label": LABEL[ref], "ref_short": SHORT[ref],
        "class_rows": class_rows, "class_text": class_text, "dept_rows": dept_rows,
        "win_rows": win_rows, "win_class_rows": win_class_rows,
        "week_rows": week_rows, "model_slice_bias": f"{slice_bias[name]:+.1f}", "ref_slice_bias": f"{slice_bias[ref]:+.1f}",
        "model_week_range": f"{wm.min():+.1f}% to {wm.max():+.1f}%", "ref_week_range": f"{wr_.min():+.1f}% to {wr_.max():+.1f}%",
        "under_weeks": under, "n_weeks": len(weeks),
        "f5_file": h["f5_forecasts"]["file"], "f5_rows": f"{h['f5_forecasts']['rows']:,}",
    }


def main() -> dict:
    h = json.loads((RESULTS / "f4_holdout.json").read_text(encoding="utf-8"))
    b = json.loads((RESULTS / "f4_backtest.json").read_text(encoding="utf-8"))
    f1 = json.loads((RESULTS / "f1_data_report.json").read_text(encoding="utf-8"))
    v = values(h, b, f1)
    plots.f4_holdout_vs_backtest(h, b, FIGURE)
    DOC.write_text(TEMPLATE.read_text(encoding="utf-8").format(**v), encoding="utf-8")
    return v


if __name__ == "__main__":
    main()
    print(f"wrote {DOC}")
