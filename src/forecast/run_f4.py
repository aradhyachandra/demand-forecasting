"""F4 part A: the global LightGBM model on the F3 backtest origins (the holdout is never read).

For each backtest origin, train every grid setting on rows from earlier origins (target days <= the
backtest origin), forecast the 28-day window, and score it with the F3 harness next to the baselines,
on exactly the same series and days. Pick the setting with the lowest WRMSSE averaged over the
origins, then freeze it in config/model_final.yaml for the one-shot holdout run (run_f4_holdout.py).

Writes results/f4_backtest.json, results/f4_grid.csv, results/f4_summary.csv, figures in
docs/figures/, docs/F4_model_backtest.md and config/model_final.yaml. Forecasts and residuals per
series, origin and horizon go to data/processed/ (gitignored) for F5.

    python -m forecast.run_f4
"""
from __future__ import annotations

import hashlib
import json
import resource
import sys
import time

import numpy as np
import pandas as pd

from forecast import backtest as bt
from forecast import config, plots, run_f3
from forecast import model as m

ROOT = config.ROOT
RESULTS = ROOT / "results"
REPORT = RESULTS / "f4_backtest.json"
FIGURES = ROOT / "docs" / "figures"
TEMPLATE = ROOT / "docs" / "templates" / "F4_model_backtest.md"
DOC = ROOT / "docs" / "F4_model_backtest.md"
FINAL = config.CONFIG_DIR / "model_final.yaml"
F3_REPORT = RESULTS / "f3_backtest.json"


def peak_rss_gb() -> float:
    """Peak resident memory of this process (ru_maxrss is bytes on macOS, KiB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r * 1024 / 1e9


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid_name(setting: dict, rounds: int) -> str:
    return f"{setting['name']}_r{rounds}"


def run_grid(data: m.Data, cfg: dict, bt_cfg: dict, origins: list[int]):
    """Train each setting once per backtest origin with max(rounds) trees; forecast after each
    checkpoint in rounds. Returns forecasts {origin: {grid name: series x horizon}}, gain importance
    {grid name: {origin: {feature: gain}}} and per-origin training-table facts."""
    names = m.feature_names(cfg)
    rounds = sorted(cfg["grid"]["rounds"])
    forecasts, gains, tables = {o: {} for o in origins}, {}, []
    for B in origins:
        t0 = time.time()
        X, y, meta = m.training_table(data, B, cfg, bt_cfg)
        tables.append({"origin": B, "rows": int(len(X)), "table_gb": float((X.nbytes + y.nbytes) / 1e9),
                       "first_training_origin": int(meta["origin"].min()),
                       "last_training_origin": int(meta["origin"].max()),
                       "first_target_d": int(meta["d_num"].min()), "last_target_d": int(meta["d_num"].max()),
                       "n_series": int(len(np.unique(meta["series"]))), "build_seconds": round(time.time() - t0, 1),
                       "fit_seconds": {}})
        del meta
        for s in cfg["grid"]["settings"]:
            t0 = time.time()
            booster = m.fit(X, y, names, m.lgb_params(cfg, s), max(rounds))
            fc = m.forecast(data, booster, B, cfg, bt_cfg, rounds)
            for r in rounds:
                forecasts[B][grid_name(s, r)] = fc[r]
                gain = booster.feature_importance(importance_type="gain", iteration=r)
                gains.setdefault(grid_name(s, r), {})[B] = dict(zip(booster.feature_name(), map(float, gain)))
            tables[-1]["fit_seconds"][s["name"]] = round(time.time() - t0, 1)
            print(f"  origin d_{B} {s['name']}: {tables[-1]['fit_seconds'][s['name']]} s", flush=True)
        del X, y
    return forecasts, gains, tables


def grid_table(per_series: pd.DataFrame, cfg: dict, origins: list[int]) -> list[dict]:
    rows = []
    for s in cfg["grid"]["settings"]:
        for r in sorted(cfg["grid"]["rounds"]):
            g = per_series[per_series["baseline"] == grid_name(s, r)]
            gm = bt.group_metrics(g)
            per_o = {f"wrmsse_d_{o}": bt.group_metrics(g[g["origin"] == o])["wrmsse"] for o in origins}
            rows.append({"name": grid_name(s, r), "setting": s["name"], "num_leaves": s["num_leaves"],
                         "learning_rate": s["learning_rate"], "min_data_in_leaf": s["min_data_in_leaf"],
                         "rounds": r, "wrmsse": gm["wrmsse"], **per_o, "mean_rmsse": gm["mean_rmsse"],
                         "mean_bias": gm["mean_bias"], "weekly_wape": gm["weekly_wape"]})
    return rows


def forecast_long(panel: bt.Panel, captured: dict, model_fc: dict, model_name: str, H: int,
                  classes: dict) -> pd.DataFrame:
    """One row per scored series, origin, horizon and method (model + F3 baselines): the forecast,
    the actual (NaN on unlisted days) and residual = actual - forecast. For F5 (safety stock)."""
    parts = []
    for origin, inp in captured.items():
        y = panel.units[:, panel.col(origin + 1): panel.col(origin + H) + 1]
        scored = inp["eligible"] & (~np.isnan(y)).any(axis=1)
        idx = np.flatnonzero(scored)
        methods = {**inp["forecasts"], model_name: model_fc[origin]}
        for name, f in methods.items():
            fs, ys = f[idx].astype(np.float32), y[idx]
            parts.append(pd.DataFrame({
                "id": np.repeat(panel.ids[idx], H), "dept_id": np.repeat(panel.dept[idx], H),
                "demand_class": np.repeat(classes[origin][idx], H),
                "origin": np.int16(origin), "h": np.tile(np.arange(1, H + 1, dtype=np.int8), len(idx)),
                "d_num": np.tile(np.arange(origin + 1, origin + H + 1, dtype=np.int16), len(idx)),
                "method": name, "forecast": fs.ravel(), "actual": ys.ravel(),
                "residual": (ys - fs).ravel()}))
    out = pd.concat(parts, ignore_index=True)
    for c in ("id", "dept_id", "demand_class", "method"):
        out[c] = out[c].astype("category")
    return out


def importance_summary(gains: dict, origins: list[int]) -> list[dict]:
    """Gain per feature as a share of the total gain at each origin, then the mean over origins."""
    shares = pd.DataFrame({o: pd.Series(gains[o]) / sum(gains[o].values()) for o in origins})
    out = shares.assign(mean_share=shares.mean(axis=1)).sort_values("mean_share", ascending=False)
    return [{"feature": f, "mean_share": float(r["mean_share"]),
             "per_origin": {f"d_{o}": float(r[o]) for o in origins}} for f, r in out.iterrows()]


def main() -> dict:
    t_start = time.time()
    data_cfg, bt_cfg, cfg = config.load("data"), config.load("backtest"), config.load("model")
    cls_cfg = config.load("demand_patterns")["classification"]
    holdout_days = config.load("holdout")["holdout_days"]
    state = config.load("scope")["state"]
    processed = ROOT / data_cfg["processed_dir"]
    H, name = bt_cfg["horizon_days"], cfg["model_name"]

    data = m.load_data(processed / data_cfg["slice_parquet"], holdout_days, cfg,
                       calendar_csv=config.raw_paths(data_cfg)["calendar"], state=state, cls_cfg=cls_cfg,
                       memory_limit=data_cfg["duckdb"]["memory_limit"], threads=data_cfg["duckdb"]["threads"])
    panel = data.panel
    origins = bt.origins(panel.last_train, H, bt_cfg["n_origins"])
    n_feat = len(m.feature_names(cfg))
    est_gb = m.estimate_table_gb(len(panel.ids), H, cfg["training_origins"]["n_origins"], n_feat)
    m.check_table_size(est_gb, cfg["max_feature_table_gb"])
    print(f"training table estimate {est_gb:.2f} GB per backtest origin", flush=True)

    t_grid = time.time()
    forecasts, gains, tables = run_grid(data, cfg, bt_cfg, origins)
    grid_seconds = time.time() - t_grid

    captured, classes = {}, {}

    def extra(origin, inp):
        captured[origin] = {"eligible": inp["eligible"], "forecasts": inp["forecasts"]}
        classes[origin] = inp["demand_class"]
        return forecasts[origin]

    per_series, coverage, weekly = bt.evaluate(panel, bt_cfg, cls_cfg, extra=extra)

    grid = grid_table(per_series, cfg, origins)
    win = min(grid, key=lambda g: g["wrmsse"])
    order = bt_cfg["baselines"] + [name]
    keep = per_series["baseline"].isin(bt_cfg["baselines"] + [win["name"]])
    df = per_series[keep].copy()
    df["baseline"] = df["baseline"].replace({win["name"]: name})
    counts = df.groupby("baseline").size()
    assert counts.nunique() == 1, "the model is not scored on the same series-origins as the baselines"
    wk = weekly[weekly["baseline"].isin(bt_cfg["baselines"] + [win["name"]])].copy()
    wk["baseline"] = wk["baseline"].replace({win["name"]: name})
    wk["error_pct"] = 100 * (wk["forecast"] / wk["actual"] - 1)

    summary = run_f3.summarize(df, {**bt_cfg, "baselines": order})
    o = summary["overall"]
    best_base = min(bt_cfg["baselines"], key=lambda b: o[b]["wrmsse"])
    pair = df[df["baseline"].isin([best_base, name])]
    wins_best = {"reference": best_base,
                 "overall": bt.win_counts(pair, best_base, bt_cfg["tie_tolerance"])[name],
                 "by_class": {c: bt.win_counts(g, best_base, bt_cfg["tie_tolerance"])[name]
                              for c, g in pair.groupby("demand_class")}}
    f3 = json.loads(F3_REPORT.read_text(encoding="utf-8"))
    f3_match = all(abs(f3["overall"][b]["wrmsse"] - o[b]["wrmsse"]) < 1e-9 for b in bt_cfg["baselines"])
    slice_bias = {b: float(100 * (g["forecast"].sum() / g["actual"].sum() - 1)) for b, g in wk.groupby("baseline")}

    long = forecast_long(panel, captured, {B: forecasts[B][win["name"]] for B in origins}, name, H, classes)
    long_path = processed / "f4_backtest_forecasts.parquet"
    long.to_parquet(long_path, index=False)
    n_long = len(long)
    del long

    report = {
        "setup": {
            "origins": origins, "horizon_days": H, "last_train_d": panel.last_train,
            "first_holdout_d": data.first_holdout, "baselines": bt_cfg["baselines"], "model": name,
            "reference_baseline": bt_cfg["reference_baseline"], "methods": order,
            "features": m.feature_names(cfg), "categorical": m.CATEGORICAL,
            "objective": cfg["objective"], "tweedie_variance_power": cfg["tweedie_variance_power"],
            "training_origins": cfg["training_origins"], "closed_events": cfg["closed_events"],
            "n_series_origin_pairs": int(df[["id", "origin"]].drop_duplicates().shape[0]),
            "f3_baselines_match_f3_results": bool(f3_match),
        },
        "training_tables": tables,
        "coverage": coverage,
        "grid": grid,
        "chosen": win,
        "best_baseline": {"name": best_base, "wrmsse": o[best_base]["wrmsse"]},
        **summary,
        "wins_vs_best_baseline": wins_best,
        "slice_total_bias_pct": slice_bias,
        "feature_importance_gain": importance_summary(gains[win["name"]], origins),
        "weekly_totals": json.loads(wk.to_json(orient="records")),
        "f5_forecasts": {"file": f"{data_cfg['processed_dir']}/{long_path.name}", "rows": int(n_long),
                         "methods": order, "residual": "actual - forecast; actual NaN on unlisted days"},
        "resources": {"training_table_estimate_gb": round(est_gb, 3),
                      "training_table_max_measured_gb": round(max(t["table_gb"] for t in tables), 3),
                      "peak_memory_gb": round(peak_rss_gb(), 2), "grid_minutes": round(grid_seconds / 60, 1),
                      "total_minutes": round((time.time() - t_start) / 60, 1)},
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(grid).to_csv(RESULTS / "f4_grid.csv", index=False, float_format="%.4f")
    rows = [{"group": "overall", "key": "all", "method": b, **mm} for b, mm in o.items()]
    rows += [{"group": g, "key": k, "method": b, **mm}
             for g in ("by_class", "by_dept", "by_origin") for k, mb in report[g].items() for b, mm in mb.items()]
    pd.DataFrame(rows).to_csv(RESULTS / "f4_summary.csv", index=False, float_format="%.4f")
    df.to_parquet(processed / "f4_series_metrics.parquet", index=False)

    plots.f4_grid(report, FIGURES / "f4_tuning_grid.png")
    plots.f4_error_by_class(report, FIGURES / "f4_error_by_class.png")
    plots.f4_feature_importance(report, FIGURES / "f4_feature_importance.png")

    write_final(report, cfg)
    if TEMPLATE.exists():
        render_doc(report, cfg)
    return report


def write_final(report: dict, cfg: dict) -> None:
    """Freeze the chosen setting and the exact recipe of the final run (part B) in config/."""
    w, s = report["chosen"], report["setup"]
    last_train, H = s["last_train_d"], s["horizon_days"]
    to = m.training_origins(last_train, cfg["training_origins"])
    text = f"""# F4 final model: frozen by `python -m forecast.run_f4` (part A). Do not edit by hand.
# Used once by `python -m forecast.run_f4_holdout` (part B), from a clean, committed tree.
# The setting was chosen on the 4 backtest origins only (lowest mean WRMSSE, results/f4_backtest.json);
# the holdout was not read, scored or tuned on.

chosen_setting: {w['setting']}
chosen_reason: lowest WRMSSE averaged over the backtest origins {', '.join(f'd_{o}' for o in s['origins'])} ({w['wrmsse']:.4f}); the whole grid is in results/f4_grid.csv
params:
  num_leaves: {w['num_leaves']}
  learning_rate: {w['learning_rate']}
  min_data_in_leaf: {w['min_data_in_leaf']}
rounds: {w['rounds']}

# Everything else (features, objective, fixed LightGBM parameters, training-origin spacing, closed
# days) comes from config/model.yaml, which must be unchanged since this freeze:
model_yaml_sha256: {sha256(config.CONFIG_DIR / 'model.yaml')}

final_run:
  forecast_origin: {last_train}        # the last training day; nothing after it is used as an input
  forecast_days: [{last_train + 1}, {last_train + H}]   # the holdout, d_{last_train + 1}..d_{last_train + H}
  training_origins: [{to[0]}, {to[-1]}]   # every {cfg['training_origins']['step_days']} days, {len(to)} origins; their target days end on d_{last_train}
  training_reason: the same recipe as in the backtest (one year of origins before the forecast origin), moved to the last training day so the model sees the most recent year
  baselines: F3 baselines from config/backtest.yaml, same origin, scored with the same metrics and weights
"""
    FINAL.write_text(text, encoding="utf-8")


LABEL = {**run_f3.LABEL, "lgbm": "LightGBM (global)"}
SHORT = {**plots.F3_SHORT, "lgbm": "LightGBM"}


def render_doc(report: dict, cfg: dict) -> None:
    s, o = report["setup"], report["overall"]
    order, name = s["methods"], s["model"]
    bb = report["best_baseline"]["name"]
    ref = s["reference_baseline"]
    w = report["chosen"]
    res = report["resources"]

    def pct(a, b):
        return 100 * (a / b - 1)

    gap = pct(o[name]["wrmsse"], o[bb]["wrmsse"])
    overall_rows = "\n".join(
        f"| {'**' + LABEL[b] + '**' if b == name else LABEL[b]} | {o[b]['wrmsse']:.3f} | {o[b]['mean_rmsse']:.3f} | "
        f"{o[b]['median_rmsse']:.3f} | {o[b]['mean_mae']:.3f} | {o[b]['mean_bias']:+.3f} | "
        f"{report['slice_total_bias_pct'][b]:+.1f}% | {100 * o[b]['weekly_wape']:.1f}% |" for b in order)

    grid_rows = "\n".join(
        f"| {g['setting']} | {g['num_leaves']} | {g['learning_rate']} | {g['min_data_in_leaf']} | {g['rounds']} | "
        + " | ".join(f"{g[f'wrmsse_d_{x}']:.3f}" for x in s["origins"])
        + (f" | **{g['wrmsse']:.3f}** |" if g["name"] == w["name"] else f" | {g['wrmsse']:.3f} |")
        for g in report["grid"])
    gw = [g["wrmsse"] for g in report["grid"]]

    bc = report["by_class"]
    class_rows = "\n".join(
        f"| {c} | {mm[name]['n']:,} | {mm[name]['wrmsse']:.3f} | {mm[bb]['wrmsse']:.3f} | {mm[ref]['wrmsse']:.3f} | "
        f"{pct(mm[name]['wrmsse'], mm[bb]['wrmsse']):+.1f}% | {mm[name]['mean_bias']:+.3f} | {mm[bb]['mean_bias']:+.3f} |"
        for c, mm in bc.items())
    class_better = [c for c, mm in bc.items() if mm[name]["wrmsse"] < mm[bb]["wrmsse"]]
    class_worse = [c for c in bc if c not in class_better]

    bo = report["by_origin"]
    origin_rows = "\n".join(
        f"| d_{k} | {mm[name]['wrmsse']:.3f} | {mm[bb]['wrmsse']:.3f} | {mm[ref]['wrmsse']:.3f} | "
        f"{pct(mm[name]['wrmsse'], mm[bb]['wrmsse']):+.1f}% | {run_f3.LABEL[run_f3.best({b: mm[b] for b in s['baselines']})[0]]} |"
        for k, mm in bo.items())
    o_better = sum(mm[name]["wrmsse"] < mm[bb]["wrmsse"] for mm in bo.values())
    o_gaps = [pct(mm[name]["wrmsse"], mm[bb]["wrmsse"]) for mm in bo.values()]
    win_o = [w[f"wrmsse_d_{x}"] for x in s["origins"]]
    best_per_origin = {x: min(report["grid"], key=lambda g: g[f"wrmsse_d_{x}"])["setting"] for x in s["origins"]}

    bd = report["by_dept"]
    dept_rows = "\n".join(
        f"| {k} | {mm[name]['n']:,} | {mm[name]['wrmsse']:.3f} | {mm[bb]['wrmsse']:.3f} | {mm[ref]['wrmsse']:.3f} | "
        f"{pct(mm[name]['wrmsse'], mm[bb]['wrmsse']):+.1f}% |" for k, mm in bd.items())

    wr, wb = report["wins_vs_reference"], report["wins_vs_best_baseline"]

    def wline(x):
        return (f"{x['pairs']['wins']:,} | {x['pairs']['ties']:,} | {x['pairs']['losses']:,} | "
                f"{100 * x['pairs']['share_wins']:.0f}% | {x['series']['wins']:,} of {x['series']['n']:,} "
                f"({100 * x['series']['share_wins']:.0f}%)")
    win_rows = (f"| vs {LABEL[ref]} | {wline(wr['overall'][name])} |\n"
                f"| vs {LABEL[bb]} (best F3 baseline) | {wline(wb['overall'])} |")
    win_class_rows = "\n".join(
        f"| {c} | {100 * wr['by_class'][c][name]['pairs']['share_wins']:.0f}% | "
        f"{100 * wb['by_class'][c]['pairs']['share_wins']:.0f}% |" for c in bc if c in wb["by_class"])

    imp = report["feature_importance_gain"]
    top = imp[:5]
    imp_rows = "\n".join(f"| {i + 1} | `{r['feature']}` | {100 * r['mean_share']:.1f}% | "
                         f"{100 * min(r['per_origin'].values()):.1f}% to {100 * max(r['per_origin'].values()):.1f}% |"
                         for i, r in enumerate(imp))
    price_share = sum(r["mean_share"] for r in imp if r["feature"].startswith("price"))
    cal_feats = {"wday", "month", "mday", "snap", "event_name", "event_type", "days_to_event"}
    cal_share = sum(r["mean_share"] for r in imp if r["feature"] in cal_feats)

    wk = pd.DataFrame(report["weekly_totals"])
    weekly_text = "; ".join(
        f"{SHORT[b]} {g['error_pct'].min():+.1f}% to {g['error_pct'].max():+.1f}% "
        f"(mean absolute {g['error_pct'].abs().mean():.1f}%)"
        for b, g in ((b, wk[wk["baseline"] == b]) for b in (name, bb, ref)))
    first_o = s["origins"][0]
    wk_first = {b: wk[(wk["baseline"] == b) & (wk["origin"] == first_o)]["error_pct"] for b in (name, bb)}

    t = report["training_tables"]
    fit_times = [v for x in t for v in x["fit_seconds"].values()]
    tr = s["training_origins"]

    headline = (f"The global LightGBM model has a WRMSSE of **{o[name]['wrmsse']:.3f}**, against "
                f"**{o[bb]['wrmsse']:.3f}** for the best F3 baseline ({LABEL[bb]}) and {o[ref]['wrmsse']:.3f} "
                f"for seasonal naive: {abs(gap):.1f}% {'lower' if gap < 0 else 'higher'} than the best baseline.")
    stability_text = (f"The model beats {LABEL[bb]} at {o_better} of {len(bo)} origins (gap "
                      f"{min(o_gaps):+.1f}% to {max(o_gaps):+.1f}%).")
    grid_text = (f"The {len(gw)} grid points range from {min(gw):.3f} to {max(gw):.3f} WRMSSE, so the choice "
                 f"of setting moves the result by {max(gw) - min(gw):.3f}. The winner scores {min(win_o):.3f} to "
                 f"{max(win_o):.3f} across the origins. The best setting per origin: "
                 + ", ".join(f"d_{x} {v}" for x, v in best_per_origin.items()) + ".")
    settings = cfg["grid"]["settings"]
    edges = []
    if w["rounds"] == min(g["rounds"] for g in report["grid"]):
        edges.append("the fewest trees tried")
    if w["num_leaves"] == min(x["num_leaves"] for x in settings):
        edges.append("the smallest trees tried")
    grid_text += (f" The winner sits at the edge of the grid ({' and '.join(edges)}), so a smaller model "
                  "might score a little better; that was not tried (see NOTES)." if edges else "")
    osm = res.get("os_measurement")
    mem_text = (f" Measured by the operating system (`/usr/bin/time -l`): maximum resident memory "
                f"{osm['max_resident_gb']:.2f} GB, macOS peak memory footprint {osm['peak_memory_footprint_gb']:.2f} GB "
                f"(it also counts memory LightGBM's threads allocated and kept between fits), {osm['swaps']} swaps."
                if osm else "")
    if not class_worse:
        class_text = f"The model is better than {SHORT[bb]} in every class."
    elif class_better:
        class_text = f"The model is better than {SHORT[bb]} for {', '.join(class_better)}, and not for {', '.join(class_worse)}."
    else:
        class_text = f"The model is not better than {SHORT[bb]} in any class."

    values = {
        "headline": headline, "origins": ", ".join(f"d_{x}" for x in s["origins"]), "horizon": s["horizon_days"],
        "n_origins": len(s["origins"]), "first_holdout": s["first_holdout_d"], "last_train": s["last_train_d"],
        "pairs": f"{s['n_series_origin_pairs']:,}", "n_features": len(s["features"]),
        "tr_step": tr["step_days"], "tr_n": tr["n_origins"], "tr_gap": tr["latest_gap_days"],
        "rows_range": f"{min(x['rows'] for x in t):,} to {max(x['rows'] for x in t):,}",
        "tweedie": s["tweedie_variance_power"], "closed": ", ".join(s["closed_events"]),
        "est_gb": f"{res['training_table_estimate_gb']:.2f}", "meas_gb": f"{res['training_table_max_measured_gb']:.2f}",
        "peak_gb": f"{res['peak_memory_gb']:.2f}", "grid_min": res["grid_minutes"], "total_min": res["total_minutes"],
        "fit_range": f"{min(fit_times):.0f} to {max(fit_times):.0f}",
        "mem_text": mem_text,
        "grid_rows": grid_rows, "grid_text": grid_text,
        "rounds_text": " / ".join(str(r) for r in sorted({g["rounds"] for g in report["grid"]})),
        "origin_header": " | ".join(f"d_{x}" for x in s["origins"]),
        "origin_sep": "|".join("---" for _ in s["origins"]),
        "chosen": w["setting"], "chosen_leaves": w["num_leaves"], "chosen_lr": w["learning_rate"],
        "chosen_min": w["min_data_in_leaf"], "chosen_rounds": w["rounds"], "chosen_w": f"{w['wrmsse']:.3f}",
        "overall_rows": overall_rows, "model_label": LABEL[name], "bb_label": LABEL[bb], "ref_label": LABEL[ref],
        "bb_short": SHORT[bb], "class_rows": class_rows, "class_text": class_text,
        "origin_rows": origin_rows, "stability_text": stability_text, "dept_rows": dept_rows,
        "win_rows": win_rows, "win_class_rows": win_class_rows,
        "model_bias": f"{o[name]['mean_bias']:+.3f}", "model_slice_bias": f"{report['slice_total_bias_pct'][name]:+.1f}",
        "bb_slice_bias": f"{report['slice_total_bias_pct'][bb]:+.1f}",
        "weekly_text": weekly_text, "n_weeks": int((wk["baseline"] == name).sum()),
        "first_origin": first_o,
        "first_week_model": f"{wk_first[name].min():+.1f}% to {wk_first[name].max():+.1f}%",
        "first_week_bb": f"{wk_first[bb].min():+.1f}% to {wk_first[bb].max():+.1f}%",
        "imp_rows": imp_rows, "top_text": ", ".join(f"`{r['feature']}` ({100 * r['mean_share']:.0f}%)" for r in top),
        "price_share": f"{100 * price_share:.1f}", "cal_share": f"{100 * cal_share:.1f}",
        "f3_match": "match" if s["f3_baselines_match_f3_results"] else "DO NOT match",
        "f5_rows": f"{report['f5_forecasts']['rows']:,}", "f5_file": report["f5_forecasts"]["file"],
    }
    DOC.write_text(TEMPLATE.read_text(encoding="utf-8").format(**values), encoding="utf-8")


def rerender() -> dict:
    """Figures, model_final.yaml and the doc again from the saved results (no retraining)."""
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    cfg = config.load("model")
    plots.f4_grid(report, FIGURES / "f4_tuning_grid.png")
    plots.f4_error_by_class(report, FIGURES / "f4_error_by_class.png")
    plots.f4_feature_importance(report, FIGURES / "f4_feature_importance.png")
    write_final(report, cfg)
    render_doc(report, cfg)
    return report


if __name__ == "__main__":
    r = rerender() if "--render" in sys.argv else main()
    print(json.dumps({k: r[k] for k in ("chosen", "best_baseline", "overall", "resources")}, indent=2))
