"""F5: inventory policy and simulation on top of the fixed F4 forecasts.

    python -m forecast.run_f5            # simulate, write results, figures and the doc
    python -m forecast.run_f5 --render   # re-render the doc and figures from results/f5_inventory.json

Reads (never changes) the F4 forecasts in data/processed/: the one holdout forecast from d_1913
(forecast and actual only; its residuals are never read) and the backtest forecasts, whose errors
size the safety stock. Prices and revenue weights come from the training panel at the origin.
Writes results/f5_inventory.json, small aggregated CSVs, figures and docs/F5_inventory.md.
Every assumption is in config/inventory.yaml. Outcomes are simulated under those assumptions.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from forecast import backtest as bt
from forecast import config, plots
from forecast import inventory as inv
from forecast.demand_patterns import CLASSES

ROOT = config.ROOT
RESULTS = ROOT / "results"
REPORT = RESULTS / "f5_inventory.json"
FIGURES = ROOT / "docs" / "figures"
TEMPLATE = ROOT / "docs" / "templates" / "F5_inventory.md"
DOC = ROOT / "docs" / "F5_inventory.md"
LABEL = {"seasonal_naive": "Seasonal naive", "tsb": "TSB", "lgbm": "LightGBM"}
ROLE = {"seasonal_naive": "today's simple rule", "tsb": "best baseline", "lgbm": "model"}
ASSUMPTIONS = {   # config key -> simulate() / measure argument
    "lead_time_days": "lead", "review_period_days": "review", "starting_inventory_factor": "start_factor",
    "unmet_demand": "unmet", "holding_cost_rate_per_year": "hold", "stockout_cost_share_of_price": "short",
}


# --- loading ------------------------------------------------------------------------------------

@dataclass
class Window:
    """One forecast origin: series x 28-day forecasts per method and the actual demand."""
    origin: int
    ids: np.ndarray
    dept: np.ndarray
    cls: np.ndarray
    forecasts: dict
    actual: np.ndarray
    price: np.ndarray = None
    weight: np.ndarray = None


def load_window(path: Path, methods: list[str], origin: int, horizon: int) -> Window:
    """Forecast and actual (never the residual column) for one origin, one row per series."""
    p = Path(path).as_posix()
    q = (f"SELECT id, dept_id, demand_class, h, method, forecast, actual FROM read_parquet('{p}') "
         f"WHERE origin = {int(origin)} AND method IN ({', '.join(repr(m) for m in methods)}) ORDER BY method, id, h")
    df = duckdb.sql(q).df()
    ids = np.sort(df["id"].unique())
    fc = {}
    for m, g in df.groupby("method"):
        assert len(g) == len(ids) * horizon and (g["id"].to_numpy()[::horizon] == ids).all(), \
            f"{m} at d_{origin}: not every series has {horizon} forecast days"
        fc[m] = g["forecast"].to_numpy(float).reshape(len(ids), horizon)
    g = df[df["method"] == methods[0]]
    actual = g["actual"].to_numpy(float).reshape(len(ids), horizon)
    for m in methods[1:]:
        assert np.array_equal(df[df["method"] == m]["actual"].to_numpy(float).reshape(len(ids), horizon), actual)
    first = g.iloc[::horizon]
    return Window(origin=int(origin), ids=ids, dept=first["dept_id"].to_numpy(), cls=first["demand_class"].to_numpy(),
                  forecasts=fc, actual=actual)


def load_backtest_errors(path: Path, methods: list[str], origins: list[int]) -> pd.DataFrame:
    p = Path(path).as_posix()
    q = (f"SELECT id, origin, h, method, residual FROM read_parquet('{p}') WHERE origin IN "
         f"({', '.join(map(str, origins))}) AND method IN ({', '.join(repr(m) for m in methods)})")
    return duckdb.sql(q).df()


def add_prices(w: Window, panel: bt.Panel, weight_days: int) -> None:
    """Last known price and revenue weight at the origin (F3 rules), from days <= origin only."""
    assert w.origin <= panel.last_train
    c = panel.col(w.origin) + 1
    pos = pd.Series(np.arange(len(panel.ids)), index=panel.ids).reindex(w.ids).to_numpy()
    assert not np.isnan(pos).any(), "a forecast series is missing from the panel"
    rows = pos.astype(int)
    w.price = bt.last_known_price(panel.price[rows, :c], panel.wday[:c])
    w.weight = bt.revenue_weight(panel.units[rows, :c], panel.price[rows, :c], panel.wday[:c], weight_days)
    assert not np.isnan(w.price).any(), "a series has no known price at the origin"


# --- running scenarios --------------------------------------------------------------------------

def central(cfg: dict) -> dict:
    return {"lead": cfg["lead_time_days"]["central"], "review": cfg["review_period_days"]["central"],
            "start_factor": cfg["starting_inventory_factor"]["central"], "unmet": cfg["unmet_demand"]["central"],
            "hold": cfg["holding_cost_rate_per_year"]["central"], "short": cfg["stockout_cost_share_of_price"]["central"],
            "sl": cfg["service_level"]["central"], "ss_method": cfg["safety_stock_method"]["central"]}


class Runner:
    """Simulates policies on one window; error spreads cached per (method, protection days)."""

    def __init__(self, w: Window, errors: pd.DataFrame, spread_origins: list[int], horizon: int, policies: list[str],
                 top_share: float):
        self.w, self.errors, self.origins, self.H, self.policies = w, errors, spread_origins, horizon, policies
        self.top_share = top_share
        self._spread = {}
        self.keep = np.ones(len(w.ids), dtype=bool)

    def spread(self, method: str, p: int):
        if (method, p) not in self._spread:
            self._spread[(method, p)] = inv.spread_from_backtest(
                self.errors, method, self.w.ids, p, self.origins, forecast_origin=self.w.origin, horizon=self.H)
        return self._spread[(method, p)]

    def restrict_to_series_with_spread(self, p: int) -> int:
        for m in self.policies:
            self.keep &= self.spread(m, p)[2] > 1
        return int((~self.keep).sum())

    def safety_stock(self, method: str, s: dict) -> np.ndarray:
        mean, sd, _ = self.spread(method, s["lead"] + s["review"])
        assert not np.isnan(mean[self.keep]).any(), f"{method}: a kept series has no backtest windows"
        return inv.safety_stock(mean[self.keep], sd[self.keep], inv.z_value(s["sl"]), s["ss_method"])

    def measures(self, method: str, s: dict) -> pd.DataFrame:
        k = self.keep
        sim = inv.simulate(self.w.forecasts[method][k], self.w.actual[k], self.safety_stock(method, s),
                           s["lead"], s["review"], s["start_factor"], s["unmet"])
        m = inv.series_measures(sim, self.w.price[k], s["hold"], s["short"])
        m["dept"], m["cls"] = self.w.dept[k], self.w.cls[k]
        top_n = max(1, int(round(self.top_share * k.sum())))
        rank = np.argsort(-self.w.weight[k], kind="stable")
        m["top_revenue"] = False
        m.iloc[rank[:top_n], m.columns.get_loc("top_revenue")] = True
        return m


def grouped(m: pd.DataFrame) -> dict:
    return {
        "overall": inv.summarize(m),
        "by_class": {c: inv.summarize(m[m["cls"] == c]) for c in CLASSES if (m["cls"] == c).any()},
        "by_dept": {d: inv.summarize(g) for d, g in m.groupby("dept")},
        "by_revenue": {"top": inv.summarize(m[m["top_revenue"]]), "rest": inv.summarize(m[~m["top_revenue"]])},
    }


def sweep(r: Runner, s: dict, levels: list[float], methods=None) -> pd.DataFrame:
    rows = []
    for method in methods or inv.SS_METHODS:
        for pol in r.policies:
            for sl in levels:
                row = inv.summarize(r.measures(pol, {**s, "sl": sl, "ss_method": method}))
                rows.append({"policy": pol, "ss_method": method, "service_level": sl, **row})
    return pd.DataFrame(rows)


def equal_fill(sw: pd.DataFrame, targets: list[float], ss_method: str, policies: list[str]) -> list[dict]:
    """Inventory money each policy needs for each target fill rate, read off its curve."""
    out = []
    for f in targets:
        need = {}
        for pol in policies:
            g = sw[(sw["policy"] == pol) & (sw["ss_method"] == ss_method)].sort_values("service_level")
            need[pol] = inv.inventory_at_fill_rate(g["fill_rate"], g["avg_inv_money"], f)
        row = {"fill_rate": f, "inventory_money": need}
        for ref in ("seasonal_naive", "tsb"):
            if need.get("lgbm") is not None and need.get(ref) is not None:
                row[f"lgbm_vs_{ref}_pct"] = 100 * (need["lgbm"] / need[ref] - 1)
        out.append(row)
    return out


def ranking(summary: dict) -> list[str]:
    return sorted(summary, key=lambda p: summary[p]["total_cost"])


def money(m_by_policy: dict, share: float) -> dict:
    """Where lost-sales cost and inventory money sit under each policy."""
    out = {}
    for pol, m in m_by_policy.items():
        tl, ti = m["lost_cost"].sum(), m["avg_inv_money"].sum()

        def shares(col):
            g = m.groupby(col)[["lost_cost", "avg_inv_money", "demand"]].sum()
            return {str(k): {"lost_cost_share": float(v["lost_cost"] / tl) if tl else None,
                             "inventory_share": float(v["avg_inv_money"] / ti) if ti else None,
                             "demand_share": float(v["demand"] / m["demand"].sum())} for k, v in g.iterrows()}
        out[pol] = {
            "lost_cost": float(tl), "inventory_money": float(ti),
            "top_share_of_lost_cost": inv.top_share(m["lost_cost"], share),
            "top_share_of_inventory_money": inv.top_share(m["avg_inv_money"], share),
            "top_share_of_demand": inv.top_share(m["demand"], share),
            "series_with_any_lost_sale": int((m["lost"] > 0).sum()),
            "by_class": shares("cls"), "by_dept": shares("dept"),
            "by_revenue": {k: v for k, v in zip(("rest", "top"), shares("top_revenue").values())},
        }
    return out


def safety_stock_summary(r: Runner, s: dict) -> dict:
    out = {}
    k = r.keep
    for pol in r.policies:
        mean, sd, n = r.spread(pol, s["lead"] + s["review"])
        ss = r.safety_stock(pol, s)
        fc = r.w.forecasts[pol][k][:, : s["lead"] + s["review"]].sum(axis=1)
        out[pol] = {"mean_window_error_total": float(mean[k].sum()), "sd_window_error_median": float(np.median(sd[k])),
                    "share_series_under_forecast": float((mean[k] > 0).mean()),
                    "safety_stock_units": float(ss.sum()), "safety_stock_money": float((ss * r.w.price[k]).sum()),
                    "first_protection_forecast_units": float(fc.sum()),
                    "windows_per_series_median": float(np.median(n[k]))}
    return out


def scenario_values(cfg: dict) -> dict:
    return {key: cfg[key]["scenarios"] for key in ASSUMPTIONS}


def run_window(r: Runner, cfg: dict, s0: dict, full: bool) -> dict:
    """Main scenario (both safety-stock methods), service-level sweep, equal-fill comparison and,
    if `full`, every other assumption over its range."""
    levels, targets = cfg["service_level"]["sweep"], cfg["compare_fill_rates"]
    main = {}
    for method in inv.SS_METHODS:
        main[method] = {pol: grouped(r.measures(pol, {**s0, "ss_method": method})) for pol in r.policies}
    sw = sweep(r, s0, levels)
    out = {"main": main, "sweep": sw,
           "equal_fill": {m: equal_fill(sw, targets, m, r.policies) for m in inv.SS_METHODS},
           "ranking_by_total_cost": ranking({p: main[s0["ss_method"]][p]["overall"] for p in r.policies}),
           "safety_stock": safety_stock_summary(r, s0),
           "money": money({p: r.measures(p, s0) for p in r.policies}, cfg["concentration_share"])}
    if full:
        rows, checks = [], []
        for key, arg in ASSUMPTIONS.items():
            for val in cfg[key]["scenarios"]:
                s = {**s0, arg: val}
                summ = {p: inv.summarize(r.measures(p, s)) for p in r.policies}
                ssw = sweep(r, s, levels, methods=[s0["ss_method"]])
                ef = equal_fill(ssw, targets, s0["ss_method"], r.policies)
                sch = inv.schedule(r.H, s["lead"], s["review"])
                for p in r.policies:
                    rows.append({"assumption": key, "value": val, "is_central": val == cfg[key]["central"],
                                 "policy": p, **summ[p]})
                checks.append({"assumption": key, "value": val, "is_central": val == cfg[key]["central"],
                               "reviews": len(sch.reviews), "measured_days": sch.last - sch.first + 1,
                               "ranking_by_total_cost": ranking(summ), "equal_fill": ef})
        out["sensitivity"] = pd.DataFrame(rows)
        out["sensitivity_checks"] = checks
    return out


def setup_info(r: Runner, s0: dict, excluded: int) -> dict:
    sch = inv.schedule(r.H, s0["lead"], s0["review"])
    o = r.w.origin
    return {"forecast_origin": o, "forecast_days": [o + 1, o + r.H], "spread_origins": r.origins,
            "n_series": int(len(r.w.ids)), "n_simulated": int(r.keep.sum()), "n_excluded_no_backtest": excluded,
            "reviews_d": [o + 1 + t for t in sch.reviews], "measured_d": [o + 1 + sch.first, o + 1 + sch.last],
            "measured_days": sch.last - sch.first + 1, "protection_days": sch.protection,
            "demand_measured_units": float(r.w.actual[r.keep][:, sch.first:sch.last + 1].sum())}


def main() -> dict:
    cfg = config.load("inventory")
    data_cfg, bt_cfg = config.load("data"), config.load("backtest")
    holdout_days = config.load("holdout")["holdout_days"]
    H, pols = bt_cfg["horizon_days"], cfg["policies"]
    processed = ROOT / data_cfg["processed_dir"]
    panel = bt.load_panel(processed / data_cfg["slice_parquet"], holdout_days, data_cfg["duckdb"]["memory_limit"],
                          data_cfg["duckdb"]["threads"])
    bt_origins = bt.origins(panel.last_train, H, bt_cfg["n_origins"])
    s0 = central(cfg)
    p0 = s0["lead"] + s0["review"]

    # holdout: one forecast from the last training day; spread from all 4 backtest origins
    hold = load_window(processed / "f4_holdout_forecasts.parquet", pols, panel.last_train, H)
    add_prices(hold, panel, bt_cfg["weight_days"])
    errors = load_backtest_errors(processed / "f4_backtest_forecasts.parquet", pols, bt_origins)
    rh = Runner(hold, errors, bt_origins, H, pols, cfg["top_revenue_share"])
    ex_h = rh.restrict_to_series_with_spread(p0)
    res_h = run_window(rh, cfg, s0, full=True)

    # robustness, entirely before the holdout: last backtest origin, spread from the earlier ones
    ro = int(cfg["robustness_origin"])
    early = [o for o in bt_origins if o + H <= ro]
    rob = load_window(processed / "f4_backtest_forecasts.parquet", pols, ro, H)
    add_prices(rob, panel, bt_cfg["weight_days"])
    rr = Runner(rob, errors, early, H, pols, cfg["top_revenue_share"])
    ex_r = rr.restrict_to_series_with_spread(p0)
    res_r = run_window(rr, cfg, s0, full=False)
    del panel

    def pack(res, r, ex):
        return {"setup": setup_info(r, s0, ex), "main": res["main"], "equal_fill": res["equal_fill"],
                "ranking_by_total_cost": res["ranking_by_total_cost"], "safety_stock": res["safety_stock"],
                "money": res["money"], "sweep": json.loads(res["sweep"].to_json(orient="records"))}

    report = {"policies": pols, "labels": {p: LABEL[p] for p in pols}, "central": s0,
              "assumptions": {k: {"central": cfg[k]["central"], "scenarios": cfg[k]["scenarios"]} for k in ASSUMPTIONS},
              "service_level_sweep": cfg["service_level"]["sweep"], "compare_fill_rates": cfg["compare_fill_rates"],
              "top_revenue_share": cfg["top_revenue_share"], "concentration_share": cfg["concentration_share"],
              "holdout": pack(res_h, rh, ex_h), "robustness": pack(res_r, rr, ex_r),
              "sensitivity": json.loads(res_h["sensitivity"].to_json(orient="records")),
              "sensitivity_checks": res_h["sensitivity_checks"]}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    res_h["sweep"].assign(window="holdout").pipe(
        lambda d: pd.concat([d, res_r["sweep"].assign(window=f"robustness_d{ro}")])).to_csv(
        RESULTS / "f5_service_level_sweep.csv", index=False)
    res_h["sensitivity"].to_csv(RESULTS / "f5_sensitivity.csv", index=False)
    group_rows = []
    for method, by_pol in res_h["main"].items():
        for pol, g in by_pol.items():
            for level in ("by_class", "by_dept", "by_revenue"):
                for name, v in g[level].items():
                    group_rows.append({"ss_method": method, "policy": pol, "group_type": level[3:], "group": name, **v})
    pd.DataFrame(group_rows).to_csv(RESULTS / "f5_main_by_group.csv", index=False)
    render(report)
    return report


# --- write-up -----------------------------------------------------------------------------------

def render(report: dict) -> None:
    plots.f5_tradeoff(report, FIGURES / "f5_tradeoff.png")
    plots.f5_cost_breakdown(report, FIGURES / "f5_cost_breakdown.png")
    plots.f5_fill_by_class(report, FIGURES / "f5_fill_by_class.png")
    plots.f5_sensitivity(report, FIGURES / "f5_sensitivity.png")
    from forecast import render_f5
    DOC.write_text(TEMPLATE.read_text(encoding="utf-8").format(**render_f5.values(report)), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="re-render from results/f5_inventory.json only")
    if ap.parse_args().render:
        render(json.loads(REPORT.read_text(encoding="utf-8")))
    else:
        r = main()
        print(json.dumps({p: {k: round(v, 4) for k, v in r["holdout"]["main"]["bias_adjusted"][p]["overall"].items()
                              if k in ("fill_rate", "avg_inv_money", "total_cost")} for p in r["policies"]}, indent=1))
