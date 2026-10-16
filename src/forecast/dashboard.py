"""Data functions for the dashboard (app/streamlit_app.py). No new analysis: the app only reads
results/ and data/processed/ and, for one item at a time, replays the tested F5 simulation.

- Product view: training history (data/processed/m5_slice.parquet), the one holdout forecast and
  actuals (f4_holdout_forecasts.parquet), and a band from that item's backtest errors
  (f4_backtest_forecasts.parquet, origins before the holdout only).
- Inventory view: selects rows of the pre-computed F5 scenarios (results/f5_*.csv, f5_inventory.json).
- Portfolio view: where lost sales and inventory sit (results/f5_inventory.json).
Every inventory number is a simulated outcome under stated assumptions, not an observed result.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from forecast import backtest as bt
from forecast import inventory as inv

PROCESSED_FILES = ("m5_slice.parquet", "f4_holdout_forecasts.parquet", "f4_backtest_forecasts.parquet")
CLASS_ORDER = ["smooth", "erratic", "intermittent", "lumpy"]


def missing_processed(processed: Path) -> list[str]:
    return [f for f in PROCESSED_FILES if not (Path(processed) / f).exists()]


def load_results(root: Path) -> dict:
    res = Path(root) / "results"
    return {
        "f5": json.loads((res / "f5_inventory.json").read_text(encoding="utf-8")),
        "f4_holdout": json.loads((res / "f4_holdout.json").read_text(encoding="utf-8")),
        "sweep": pd.read_csv(res / "f5_service_level_sweep.csv"),
        "sensitivity": pd.read_csv(res / "f5_sensitivity.csv"),
        "by_group": pd.read_csv(res / "f5_main_by_group.csv"),
    }


def _q(path: Path) -> str:
    return Path(path).as_posix().replace("'", "''")


# --- product view -----------------------------------------------------------------------------------

def series_index(slice_path: Path, holdout_path: Path) -> pd.DataFrame:
    """One row per forecast series: id, item, store, department and demand class at the origin."""
    return duckdb.sql(
        f"SELECT s.id, s.item_id, s.store_id, s.dept_id, h.demand_class FROM "
        f"(SELECT DISTINCT id, item_id, store_id, dept_id FROM read_parquet('{_q(slice_path)}')) s "
        f"JOIN (SELECT DISTINCT id, demand_class FROM read_parquet('{_q(holdout_path)}')) h USING (id) "
        f"ORDER BY s.id").df()


def item_history(slice_path: Path, sid: str, last_train: int, days: int) -> pd.DataFrame:
    """The last `days` training days (<= last_train) of one series; units are NaN when not listed."""
    df = duckdb.execute(
        f"SELECT d_num, date, CASE WHEN is_listed THEN units END AS units FROM read_parquet('{_q(slice_path)}') "
        f"WHERE id = ? AND d_num <= ? AND d_num > ? ORDER BY d_num", [sid, int(last_train), int(last_train) - int(days)]).df()
    df["units"] = df["units"].astype(float)
    return df


def item_forecasts(holdout_path: Path, sid: str, methods: list[str]) -> pd.DataFrame:
    """Holdout actuals and each method's forecast for one series, one row per day."""
    df = duckdb.execute(f"SELECT d_num, method, forecast, actual FROM read_parquet('{_q(holdout_path)}') "
                        f"WHERE id = ? ORDER BY d_num", [sid]).df()
    wide = df.pivot(index="d_num", columns="method", values="forecast")[list(methods)]
    actual = df.drop_duplicates("d_num").set_index("d_num")["actual"]
    return pd.concat([actual, wide], axis=1).reset_index().rename_axis(columns=None).astype(
        {m: float for m in ["actual", *methods]})


def backtest_residuals(backtest_path: Path, sid: str, method: str, origins: list[int]) -> np.ndarray:
    """Daily errors (actual - forecast) of one series and method at the given backtest origins."""
    return duckdb.execute(
        f"SELECT residual FROM read_parquet('{_q(backtest_path)}') WHERE id = ? AND method = ? AND origin IN "
        f"({', '.join(str(int(o)) for o in origins)})", [sid, method]).fetchnumpy()["residual"].astype(float)


def band_from_residuals(forecast: np.ndarray, residuals: np.ndarray, lo: float, hi: float):
    """forecast + the lo / hi quantiles of the item's backtest daily errors, floored at zero. A rough
    picture of how wrong this method was for this item before, not a calibrated interval."""
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    fc = np.asarray(forecast, dtype=float)
    if len(r) == 0:
        return np.full(len(fc), np.nan), np.full(len(fc), np.nan)
    return np.maximum(fc + np.quantile(r, lo), 0.0), np.maximum(fc + np.quantile(r, hi), 0.0)


def last_known_price(slice_path: Path, sid: str, origin: int) -> float:
    """F3's last-full-week price at the origin for one series (days <= origin only)."""
    df = duckdb.execute(f"SELECT d_num, wday, CASE WHEN is_listed THEN sell_price END AS p FROM "
                        f"read_parquet('{_q(slice_path)}') WHERE id = ? AND d_num <= ? ORDER BY d_num",
                        [sid, int(origin)]).df()
    return float(bt.last_known_price(df["p"].to_numpy(float)[None, :], df["wday"].to_numpy())[0])


def item_backtest_errors(backtest_path: Path, sid: str, method: str, origins: list[int]) -> pd.DataFrame:
    return duckdb.execute(
        f"SELECT id, origin, h, method, residual FROM read_parquet('{_q(backtest_path)}') WHERE id = ? AND method = ? "
        f"AND origin IN ({', '.join(str(int(o)) for o in origins)})", [sid, method]).df()


def item_simulation(forecast, actual, errors: pd.DataFrame, method: str, sid: str, price: float, lead: int,
                    review: int, service_level: float, ss_method: str, spread_origins: list[int],
                    forecast_origin: int, hold: float, short: float, start_factor: float = 1.0,
                    unmet: str = "lost_sales") -> dict:
    """Replay one item with the F5 functions: spread from backtest errors only (the leakage guard in
    inventory.spread_from_backtest applies), order-up-to simulation, measures over the measured days."""
    fc, dem = np.asarray(forecast, dtype=float)[None, :], np.asarray(actual, dtype=float)[None, :]
    H = fc.shape[1]
    mean, sd, n = inv.spread_from_backtest(errors, method, np.array([sid]), lead + review, spread_origins,
                                           forecast_origin, H)
    ss = inv.safety_stock(mean, sd, inv.z_value(service_level), ss_method)
    sim = inv.simulate(fc, dem, ss, lead, review, start_factor, unmet)
    m = inv.series_measures(sim, np.array([price]), hold, short).iloc[0]
    s = sim.schedule
    days = pd.DataFrame({"day": np.arange(1, H + 1), "demand": sim.demand[0], "served": sim.served[0],
                         "lost": sim.lost[0], "arrived": sim.arrived[0], "ordered": sim.placed[0],
                         "on_hand": sim.on_hand[0]})
    days["measured"] = (days["day"] - 1 >= s.first) & (days["day"] - 1 <= s.last)
    return {"safety_stock": float(ss[0]), "windows": int(n[0]), "reviews": [r + 1 for r in s.reviews],
            "measures": {**{k: float(m[k]) for k in ("demand", "served", "lost", "avg_inv_units", "avg_inv_money",
                                                    "holding_cost", "lost_cost", "total_cost")},
                         "fill_rate": float(m["served"] / m["demand"]) if m["demand"] > 0 else None},
            "days": days}


# --- inventory view ---------------------------------------------------------------------------------

def scenario_options(f5: dict) -> dict:
    return {k: v["scenarios"] for k, v in f5["assumptions"].items()}


def _order(df: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    return df.set_index("policy").loc[policies].reset_index()


def sweep_rows(sweep: pd.DataFrame, service_level: float, ss_method: str, window: str = "holdout",
               policies=("seasonal_naive", "tsb", "lgbm")) -> pd.DataFrame:
    g = sweep[(sweep["window"] == window) & (sweep["ss_method"] == ss_method)
              & np.isclose(sweep["service_level"], service_level)]
    if g.empty:
        raise KeyError(f"no saved sweep row for service level {service_level}")
    return _order(g, list(policies))


def scenario_rows(sens: pd.DataFrame, assumption: str, value, policies=("seasonal_naive", "tsb", "lgbm")):
    g = sens[(sens["assumption"] == assumption) & (sens["value"].astype(str) == str(value))]
    if g.empty:
        raise KeyError(f"no saved scenario {assumption} = {value}")
    return _order(g, list(policies))


def scenario_check(f5: dict, assumption: str, value) -> dict:
    return next(c for c in f5["sensitivity_checks"] if c["assumption"] == assumption and str(c["value"]) == str(value))


# --- portfolio view ---------------------------------------------------------------------------------

def portfolio_shares(f5: dict, policy: str, level: str) -> pd.DataFrame:
    """Share of demand units, lost-sales cost and inventory money per class or department."""
    d = f5["holdout"]["money"][policy][level]
    keys = [c for c in CLASS_ORDER if c in d] if level == "by_class" else sorted(d)
    return pd.DataFrame([{"group": k, **d[k]} for k in keys])
