"""F2: demand patterns on the training days of the slice (the holdout is never read).

Writes results/f2_demand_patterns.json, small aggregated CSVs in results/, figures in docs/figures/
and docs/F2_demand_patterns.md. Per-series and per-cut tables stay in data/processed/ (gitignored,
since they are close to the raw data the Kaggle terms say not to redistribute).

    python -m forecast.run_f2
"""
from __future__ import annotations

import json

from forecast import config, plots
from forecast import demand_patterns as dp

ROOT = config.ROOT
RESULTS = ROOT / "results"
REPORT = RESULTS / "f2_demand_patterns.json"
FIGURES = ROOT / "docs" / "figures"
TEMPLATE = ROOT / "docs" / "templates" / "F2_demand_patterns.md"
DOC = ROOT / "docs" / "F2_demand_patterns.md"


def _records(df):
    return json.loads(df.to_json(orient="records"))


def main() -> dict:
    data_cfg = config.load("data")
    cfg = config.load("demand_patterns")
    holdout_days = config.load("holdout")["holdout_days"]
    processed = ROOT / data_cfg["processed_dir"]
    parquet = processed / data_cfg["slice_parquet"]

    con, last_train = dp.open_training_days(parquet, holdout_days, data_cfg["duckdb"]["memory_limit"],
                                            data_cfg["duckdb"]["threads"])
    span = con.execute("SELECT MIN(d_num), MAX(d_num), MIN(date), MAX(date), COUNT(*), "
                       "COUNT(DISTINCT id), SUM(units) FROM train_days").fetchone()

    classes = dp.classify(dp.series_stats(con), cfg["classification"])
    daily = dp.daily_totals(con)
    daily_dept = dp.daily_totals(con, by_dept=True)
    weekly = dp.weekly_series(con)
    cuts, controls = dp.price_windows(weekly, cfg["price_cuts"])
    price_vs_sales = dp.price_changes_by_sales(con)
    con.close()

    summary = dp.class_summary(classes)
    by_dept = (classes.groupby(["dept_id", "demand_class"]).size().unstack(fill_value=0)
               .reindex(columns=dp.CLASSES, fill_value=0))
    weekday = dp.calendar_profile(daily, "weekday")
    month = dp.month_profile(daily)
    snap = dp.snap_effect(daily)
    snap_dept = dp.snap_by_dept(daily_dept)
    occ, by_event = dp.event_effects(daily, cfg["calendar_effects"])
    by_type = (occ.groupby("event_type")["lift_pct"].agg(["size", "mean"])
               .rename(columns={"size": "n", "mean": "mean_lift_pct"}).reset_index())
    price = dp.price_cut_summary(cuts, controls, cfg["price_cuts"])
    pw = cfg["price_cuts"]["profile_weeks"]
    edges = cfg["price_cuts"]["cut_size_buckets_pct"]
    profiles = {}
    for b, lo, hi in zip(price["buckets"], edges[:-1], edges[1:]):
        sel = cuts[(cuts["cut_pct"] >= lo - 1e-9) & ((cuts["cut_pct"] <= hi) if hi == edges[-1]
                                                     else (cuts["cut_pct"] < hi - 1e-9))]
        profiles[b["label"]] = _records(dp.cut_profile(weekly, sel, pw))
    profiles["no change"] = _records(dp.cut_profile(weekly, controls, pw))
    price["profiles"] = profiles
    price["price_change_vs_sales"] = price_vs_sales
    del weekly

    report = {
        "training": {
            "first_d": int(span[0]), "last_d": int(span[1]), "last_train_d": last_train,
            "first_date": span[2].isoformat(), "last_date": span[3].isoformat(),
            "holdout_days": holdout_days, "listed_series_days": int(span[4]),
            "n_series": int(span[5]), "units": int(span[6]),
        },
        "classes": summary,
        "classes_by_dept": [{"dept_id": d, **{c: int(n) for c, n in r.items()}} for d, r in by_dept.iterrows()],
        "classes_median": {
            "adi": float(classes["adi"].median()), "cv2": float(classes["cv2"].median()),
            "share_listed_days_with_sales": float(classes["nonzero_days"].sum() / classes["listed_days"].sum()),
        },
        "weekday_profile": _records(weekday),
        "month_profile": _records(month),
        "month_profile_years": dp.full_years(daily),
        "year_rates": dp.year_rates(daily),
        "snap": snap,
        "snap_by_dept": snap_dept,
        "events": {"n_event_days": int(daily["is_event"].sum()), "n_occurrences_used": len(occ),
                   "by_event": _records(by_event), "by_type": _records(by_type)},
        "price_cuts": price,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    by_event.to_csv(RESULTS / "f2_event_effects.csv", index=False, float_format="%.4f")
    occ.to_csv(RESULTS / "f2_event_occurrences.csv", index=False, float_format="%.4f")
    classes.to_parquet(processed / "f2_series_classes.parquet", index=False)
    cuts.to_parquet(processed / "f2_price_cuts.parquet", index=False)

    plots.demand_class_map(classes, summary, cfg["classification"], FIGURES / "f2_demand_classes.png")
    plots.weekday_month_profile(weekday, month, FIGURES / "f2_weekday_month.png")
    plots.snap_effect(snap, snap_dept, FIGURES / "f2_snap_effect.png")
    plots.event_effects(by_event, FIGURES / "f2_event_effects.png")
    plots.price_cut_effect(price, cfg["price_cuts"]["before_weeks"], FIGURES / "f2_price_cuts.png")

    if TEMPLATE.exists():
        render_doc(report, cfg)
    return report


MEANING = {
    "smooth": "sells most days, steady amounts",
    "erratic": "sells most days, amounts swing",
    "intermittent": "many days without a sale, steady amounts",
    "lumpy": "many days without a sale, amounts swing",
    "too_few_sales": "fewer than the minimum days with a sale",
}


def _pct(x, digits=1, sign=False) -> str:
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def render_doc(report: dict, cfg: dict) -> None:
    t = report["training"]
    cls = {c["demand_class"]: c for c in report["classes"]}
    class_rows = "\n".join(
        f"| {c['demand_class']} | {MEANING[c['demand_class']]} | {c['n_series']:,} | "
        f"{100 * c['share_series']:.1f}% | {100 * c['share_units']:.1f}% |" for c in report["classes"])
    dept_rows = "\n".join(
        f"| {d['dept_id']} | " + " | ".join(f"{d[c]:,}" for c in dp.CLASSES) + " |" for d in report["classes_by_dept"])

    wk = sorted(report["weekday_profile"], key=lambda r: -r["index"])
    weekday_text = ", ".join(f"{r['weekday']} {100 * (r['index'] - 1):+.0f}%" for r in wk) + " vs the average day"
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mo = sorted(report["month_profile"], key=lambda r: -r["index"])
    month_text = (f"highest {', '.join(f'{names[r['month'] - 1]} {100 * (r['index'] - 1):+.0f}%' for r in mo[:3])}; "
                  f"lowest {', '.join(f'{names[r['month'] - 1]} {100 * (r['index'] - 1):+.0f}%' for r in mo[-3:])}")
    year_rows = "\n".join(f"| {y['year']} | {y['days']} | {y['listed_series_per_day']:,.0f} | "
                          f"{y['units_per_day']:,.0f} | {y['rate']:.2f} |" for y in report["year_rates"])

    snap = report["snap"]
    wk_lifts = [w["lift_pct"] for w in snap["by_weekday"]]
    snap_dept_rows = "\n".join(f"| {d['dept_id']} | {d['lift_pct']:+.1f}% |" for d in report["snap_by_dept"])

    ev = report["events"]
    by_event = {e["event_name"]: e for e in ev["by_event"]}
    shown = ev["by_event"][:5] + ev["by_event"][-5:]
    event_rows = "\n".join(
        f"| {e['event_name']} | {e['event_type']} | {e['n']} | {e['mean_lift_pct']:+.1f}% | "
        f"{e['min_lift_pct']:+.1f}% | {e['max_lift_pct']:+.1f}% |" for e in shown)
    type_text = "; ".join(f"{r['event_type']} {r['mean_lift_pct']:+.1f}% (n={r['n']})" for r in ev["by_type"])

    pc = report["price_cuts"]
    price_rows = "\n".join(
        f"| {b['label']} | {b['n_cuts']:,} | {b['n_series']:,} | {b['pooled_lift_pct']:+.1f}% | "
        f"{b['median_ratio']:.2f} | {100 * b['share_price_held']:.0f}% |" for b in pc["buckets"] if b["n_cuts"])
    a = pc["all_cuts"]
    price_rows += (f"\n| all cuts | {a['n_cuts']:,} | {a['n_series']:,} | {a['pooled_lift_pct']:+.1f}% | "
                   f"{a['median_ratio']:.2f} | {100 * a['share_price_held']:.0f}% |")
    c = pc["control"]
    price_rows += (f"\n| no price change (reference) | {c['n_windows']:,} windows | {c['n_series']:,} | "
                   f"{c['pooled_lift_pct']:+.1f}% | {c['median_ratio']:.2f} | n/a |")

    before = cfg["price_cuts"]["before_weeks"]

    def indexed(label):
        prof = {r["week"]: r["units_per_day"] for r in pc["profiles"][label]}
        base = sum(prof[k] for k in range(-before, 0)) / before
        return {k: v / base for k, v in prof.items() if v is not None}

    labels = [b["label"] for b in pc["buckets"] if b["n_cuts"]]
    idx = {lab: indexed(lab) for lab in labels}
    later = range(2, cfg["price_cuts"]["profile_weeks"] + 1)

    def later_range(lab):
        vals = [idx[lab][k] for k in later if k in idx[lab]]
        return f"{min(vals):.2f} to {max(vals):.2f} of the before-level"

    vs = {r["zero_sales_week"]: r for r in pc["price_change_vs_sales"]}
    values = {
        "first_d": t["first_d"], "last_d": t["last_d"], "first_date": t["first_date"],
        "last_date": t["last_date"], "holdout_days": t["holdout_days"],
        "listed_series_days": f"{t['listed_series_days']:,}", "n_series": f"{t['n_series']:,}",
        "units": f"{t['units']:,}",
        "adi_cutoff": cfg["classification"]["adi_cutoff"], "cv2_cutoff": cfg["classification"]["cv2_cutoff"],
        "class_rows": class_rows, "dept_rows": dept_rows,
        "non_smooth_pct": f"{100 * (1 - cls['smooth']['share_series']):.0f}",
        "median_adi": f"{report['classes_median']['adi']:.1f}",
        "sell_day_pct": f"{100 * report['classes_median']['share_listed_days_with_sales']:.0f}",
        "smooth_series_pct": f"{100 * cls['smooth']['share_series']:.0f}",
        "smooth_units_pct": f"{100 * cls['smooth']['share_units']:.0f}",
        "weekday_text": weekday_text, "month_text": month_text,
        "month_years": ", ".join(str(y) for y in report["month_profile_years"]), "year_rows": year_rows,
        "snap_lift": f"{snap['lift_pct']:.1f}", "snap_rate": f"{snap['snap_rate']:.2f}",
        "non_snap_rate": f"{snap['non_snap_rate']:.2f}", "snap_days": f"{snap['snap_days']:,}",
        "non_snap_days": f"{snap['non_snap_days']:,}", "snap_dept_rows": snap_dept_rows,
        "snap_wk_min": f"{min(wk_lifts):+.1f}", "snap_wk_max": f"{max(wk_lifts):+.1f}",
        "event_weeks": cfg["calendar_effects"]["event_window_weeks"],
        "n_occurrences": ev["n_occurrences_used"], "n_event_days": ev["n_event_days"],
        "event_rows": event_rows, "type_text": type_text,
        "christmas": f"{by_event['Christmas']['mean_lift_pct']:+.0f}",
        "thanksgiving": f"{by_event['Thanksgiving']['mean_lift_pct']:+.0f}",
        "newyear": f"{by_event['NewYear']['mean_lift_pct']:+.0f}",
        "laborday": f"{by_event['LaborDay']['mean_lift_pct']:+.0f}",
        "presidentsday": f"{by_event['PresidentsDay']['mean_lift_pct']:+.0f}",
        "memorialday": f"{by_event['MemorialDay']['mean_lift_pct']:+.0f}",
        "min_cut": cfg["price_cuts"]["min_cut_pct"], "before_weeks": before, "price_rows": price_rows,
        "wkm1_text": ", ".join(f"{idx[lab][-1]:.2f} ({lab})" for lab in labels),
        "zero_week_changes": f"{vs[True]['n_changes']:,}", "zero_weeks": f"{vs[True]['n_weeks']:,}",
        "sale_week_changes": f"{vs[False]['n_changes']:,}", "sale_weeks": f"{vs[False]['n_weeks']:,}",
        "sale_week_change_pct": f"{100 * vs[False]['change_rate']:.1f}",
        "after_small": later_range(labels[0]),
        "after_text": "; ".join(f"{lab}: {later_range(lab)}" for lab in labels[1:]),
        "held_pct": f"{100 * a['share_price_held']:.0f}",
        "control_lift": f"{c['pooled_lift_pct']:+.1f}",
    }
    DOC.write_text(TEMPLATE.read_text(encoding="utf-8").format(**values), encoding="utf-8")


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("training", "classes", "snap", "price_cuts")}, indent=2))
