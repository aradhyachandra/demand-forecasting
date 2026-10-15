"""F3 rolling-origin backtest: origins and windows, loading the training days as series x day
matrices, the inputs fixed at each origin (forecasts, scaling denominator, revenue weight, demand
class) and the metrics.

An origin T is the last day of known history. Everything computed "at T" (forecasts, scale, weight,
class) gets only the matrix columns for days <= T; the window T+1 .. T+horizon is used only to score.
The holdout is never loaded (sql/f3_days.sql reads d_num <= last training day only).
Settings come from config/backtest.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import baselines, prep
from forecast import demand_patterns as dp
from forecast.demand_patterns import _sql

FRIDAY = 7  # M5 wday: 1 = Saturday ... 7 = Friday, so a week ends on wday 7


# --- origins and windows ------------------------------------------------------------------------

def last_train_day(max_d: int, holdout_days: int) -> int:
    return int(max_d) - int(holdout_days)


def origins(last_train: int, horizon: int, n_origins: int) -> list[int]:
    """Back-to-back windows, the last one ending on the last training day. Oldest first."""
    return [last_train - horizon * k for k in range(n_origins, 0, -1)]


def window(origin: int, horizon: int) -> tuple[int, int]:
    """First and last scored day of the window after an origin."""
    return origin + 1, origin + horizon


# --- loading ------------------------------------------------------------------------------------

@dataclass
class Panel:
    """Training days as matrices: one row per series (ids), one column per day first_d..last_train.
    units and price are NaN on unlisted days."""
    ids: np.ndarray
    dept: np.ndarray
    first_d: int
    last_train: int
    units: np.ndarray   # float32, series x day
    price: np.ndarray   # float32, series x day
    wday: np.ndarray    # int8, per day

    def col(self, d: int) -> int:
        return int(d) - self.first_d


def load_panel(parquet: Path, holdout_days: int, memory_limit: str = "2GB", threads: int = 4) -> Panel:
    con = prep.connect(memory_limit, threads)
    path = Path(parquet).as_posix()
    max_d, min_d = con.execute(f"SELECT MAX(d_num), MIN(d_num) FROM read_parquet('{path}')").fetchone()
    last_train = last_train_day(max_d, holdout_days)
    fill = {"parquet": path, "last_train": last_train}
    series = con.execute(_sql("f3_series.sql", **fill)).df()
    cal = con.execute(_sql("f3_calendar.sql", **fill)).df()
    days = con.execute(_sql("f3_days.sql", **fill, first_holdout=last_train + 1)).fetchnumpy()
    con.close()

    n_days = last_train - int(min_d) + 1
    assert len(cal) == n_days and cal["d_num"].tolist() == list(range(int(min_d), last_train + 1))
    units = np.full((len(series), n_days), np.nan, dtype=np.float32)
    price = np.full((len(series), n_days), np.nan, dtype=np.float32)
    rows = np.asarray(days["s"], dtype=np.int64)
    cols = np.asarray(days["d_num"], dtype=np.int64) - int(min_d)
    listed = np.asarray(days["is_listed"], dtype=bool)
    units[rows[listed], cols[listed]] = np.asarray(days["units"], dtype=np.float32)[listed]
    price[rows[listed], cols[listed]] = np.asarray(days["sell_price"], dtype=np.float32)[listed]
    return Panel(ids=series["id"].to_numpy(), dept=series["dept_id"].to_numpy(), first_d=int(min_d),
                 last_train=last_train, units=units, price=price, wday=cal["wday"].to_numpy(np.int8))


# --- inputs fixed at the origin -----------------------------------------------------------------

def scale(hist: np.ndarray) -> np.ndarray:
    """RMSSE denominator: mean squared one-step naive error, over pairs of consecutive days that are
    both listed. NaN if there is no such pair; 0 for a constant series."""
    h = np.asarray(hist, dtype=float)
    d = np.diff(h, axis=1)
    ok = ~np.isnan(d)
    n = ok.sum(axis=1)
    return np.where(n > 0, np.where(ok, d * d, 0.0).sum(axis=1) / np.maximum(n, 1), np.nan)


def last_known_price(price_hist: np.ndarray, wday_hist: np.ndarray) -> np.ndarray:
    """Price on the last listed day of the last week (Saturday-Friday) that ended by the origin.
    The origin's own week is not finished, and its price could reveal sales after the origin."""
    fridays = np.flatnonzero(np.asarray(wday_hist) == FRIDAY)
    if len(fridays) == 0:
        return np.full(len(price_hist), np.nan)
    known = np.asarray(price_hist[:, : fridays[-1] + 1], dtype=float)
    idx = np.where(~np.isnan(known), np.arange(known.shape[1]), -1).max(axis=1)
    return np.where(idx >= 0, known[np.arange(len(known)), np.maximum(idx, 0)], np.nan)


def revenue_weight(units_hist, price_hist, wday_hist, weight_days: int) -> np.ndarray:
    """Units sold in the last weight_days up to the origin x last known price (M5 WRMSSE style).
    0 if nothing sold or no price is known yet."""
    units = np.nansum(np.asarray(units_hist[:, -weight_days:], dtype=float), axis=1)
    p = last_known_price(price_hist, wday_hist)
    return np.where(np.isnan(p), 0.0, units * np.nan_to_num(p))


def class_at_origin(hist: np.ndarray, cls_cfg: dict) -> np.ndarray:
    """F2 demand class (demand_patterns.classify) from listed days up to the origin only."""
    h = np.asarray(hist, dtype=float)
    listed = ~np.isnan(h)
    nz = listed & (h > 0)
    n_nz = nz.sum(axis=1)
    vals = np.where(nz, h, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_nz = np.where(n_nz > 0, np.nansum(vals, axis=1) / np.maximum(n_nz, 1), np.nan)
        dev2 = np.where(nz, (h - mean_nz[:, None]) ** 2, 0.0).sum(axis=1)
        ddof = cls_cfg["cv2_ddof"]
        var_nz = np.where(n_nz > ddof, dev2 / np.maximum(n_nz - ddof, 1), np.nan)
    stats = pd.DataFrame({"id": np.arange(len(h)), "listed_days": listed.sum(axis=1), "nonzero_days": n_nz,
                          "mean_nz": mean_nz, "var_nz": var_nz})
    return dp.classify(stats, cls_cfg)["demand_class"].to_numpy()


def origin_inputs(panel: Panel, origin: int, cfg: dict, cls_cfg: dict) -> dict:
    """Everything fixed at the origin, from days <= origin only."""
    c = panel.col(origin) + 1
    hist, price, wday = panel.units[:, :c], panel.price[:, :c], panel.wday[:c]
    listed_days = (~np.isnan(hist)).sum(axis=1)
    return {
        "listed_days": listed_days,
        "eligible": listed_days >= cfg["min_history_days"],
        "forecasts": baselines.forecast_all(hist, cfg["horizon_days"], cfg),
        "scale": scale(hist),
        "weight": revenue_weight(hist, price, wday, cfg["weight_days"]),
        "demand_class": class_at_origin(hist, cls_cfg),
    }


# --- metrics ------------------------------------------------------------------------------------

def score(forecast: np.ndarray, actual: np.ndarray, scale: np.ndarray, season: int) -> dict:
    """Per series, on listed window days only (actual NaN = unlisted). Error = forecast - actual, so
    a positive bias means over-forecasting. rmsse = sqrt(mse / scale), NaN when scale is 0 or NaN.
    Weekly totals: sums over listed days in each block of `season` days after the origin."""
    f = np.asarray(forecast, dtype=float)
    y = np.asarray(actual, dtype=float)
    ok = ~np.isnan(y)
    n = ok.sum(axis=1)
    e = np.where(ok, f - y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = np.where(n > 0, np.abs(e).sum(axis=1) / n, np.nan)
        mse = np.where(n > 0, (e * e).sum(axis=1) / n, np.nan)
        bias = np.where(n > 0, e.sum(axis=1) / n, np.nan)
        rmsse = np.where((scale > 0) & (n > 0), np.sqrt(mse / scale), np.nan)
    n_weeks = -(-f.shape[1] // season)
    pad = n_weeks * season - f.shape[1]
    fw = np.pad(np.where(ok, f, 0.0), ((0, 0), (0, pad))).reshape(len(f), n_weeks, season).sum(axis=2)
    yw = np.pad(np.where(ok, y, 0.0), ((0, 0), (0, pad))).reshape(len(f), n_weeks, season).sum(axis=2)
    return {"n_days": n, "mae": mae, "rmse": np.sqrt(mse), "bias": bias, "rmsse": rmsse,
            "wk_forecast": fw, "wk_actual": yw}


def evaluate(panel: Panel, cfg: dict, cls_cfg: dict) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Run every baseline at every origin. Returns per-series metrics (one row per series, origin
    and baseline; scored series only), coverage per origin, and slice weekly totals."""
    H, season = cfg["horizon_days"], cfg["season_length_days"]
    rows, coverage, weekly = [], [], []
    for origin in origins(panel.last_train, H, cfg["n_origins"]):
        first, last = window(origin, H)
        assert last <= panel.last_train, "a backtest window reaches past the training days"
        inp = origin_inputs(panel, origin, cfg, cls_cfg)
        y = panel.units[:, panel.col(first): panel.col(last) + 1]
        has_window = (~np.isnan(y)).any(axis=1)
        scored = inp["eligible"] & has_window
        zero_scale = scored & ~(inp["scale"] > 0)
        cls = inp["demand_class"][scored]
        coverage.append({
            "origin": origin, "first_scored_d": first, "last_scored_d": last,
            "n_series": int(len(panel.ids)), "n_scored": int(scored.sum()),
            "n_short_history": int((~inp["eligible"]).sum()),
            "n_no_window_days": int((inp["eligible"] & ~has_window).sum()),
            "n_zero_scale": int(zero_scale.sum()),
            "zero_scale_weight_share": float(inp["weight"][zero_scale].sum() / inp["weight"][scored].sum())
            if inp["weight"][scored].sum() > 0 else 0.0,
            "scored_days": int((~np.isnan(y[scored])).sum()),
            "classes": {c: int((cls == c).sum()) for c in dp.CLASSES},
        })
        base = pd.DataFrame({"id": panel.ids[scored], "dept_id": panel.dept[scored], "origin": origin,
                             "demand_class": cls, "scale": inp["scale"][scored],
                             "weight": inp["weight"][scored]})
        for name, f in inp["forecasts"].items():
            s = score(f[scored], y[scored], inp["scale"][scored], season)
            rows.append(base.assign(
                baseline=name, n_days=s["n_days"], mae=s["mae"], rmse=s["rmse"], bias=s["bias"],
                rmsse=s["rmsse"], wk_abs_err=np.abs(s["wk_forecast"] - s["wk_actual"]).sum(axis=1),
                wk_actual=s["wk_actual"].sum(axis=1)))
            for k in range(s["wk_forecast"].shape[1]):
                weekly.append({"origin": origin, "week": k + 1, "baseline": name,
                               "forecast": float(s["wk_forecast"][:, k].sum()),
                               "actual": float(s["wk_actual"][:, k].sum())})
    return pd.concat(rows, ignore_index=True), coverage, pd.DataFrame(weekly)


# --- aggregation --------------------------------------------------------------------------------

def group_metrics(df: pd.DataFrame) -> dict:
    """Aggregates over series-origin rows of one baseline. wrmsse: per origin, the revenue-weighted
    mean RMSSE over series with a non-zero scale; then the mean over origins (as the M5 WRMSSE, but
    averaged over the backtest windows). weekly_wape = sum |weekly error| / sum weekly actual."""
    scaled = df[df["rmsse"].notna()]
    per_origin = [float((g["weight"] * g["rmsse"]).sum() / g["weight"].sum())
                  for _, g in scaled.groupby("origin") if g["weight"].sum() > 0]
    act = df["wk_actual"].sum()
    return {
        "n": int(len(df)), "n_scaled": int(len(scaled)),
        "mean_rmsse": float(scaled["rmsse"].mean()) if len(scaled) else None,
        "median_rmsse": float(scaled["rmsse"].median()) if len(scaled) else None,
        "wrmsse": float(np.mean(per_origin)) if per_origin else None,
        "mean_mae": float(df["mae"].mean()), "mean_rmse": float(df["rmse"].mean()),
        "mean_bias": float(df["bias"].mean()),
        "weekly_wape": float(df["wk_abs_err"].sum() / act) if act > 0 else None,
    }


def by_baseline(df: pd.DataFrame, order: list[str]) -> dict:
    return {b: group_metrics(df[df["baseline"] == b]) for b in order if (df["baseline"] == b).any()}


def win_counts(df: pd.DataFrame, reference: str, tol: float) -> dict:
    """Per baseline vs the reference, on RMSE (same denominator as RMSSE per series, so the order is
    the same, and zero-scale series still count). pairs = series-origin pairs; series = mean RMSE
    over the origins at which the series was scored."""
    wide = df.pivot_table(index=["id", "origin"], columns="baseline", values="rmse", sort=False)
    series = wide.groupby(level="id").mean()

    def count(t, b):
        d = (t[b] - t[reference]).dropna()
        wins, ties = int((d < -tol).sum()), int((d.abs() <= tol).sum())
        return {"n": int(len(d)), "wins": wins, "ties": ties, "losses": int(len(d) - wins - ties),
                "share_wins": float(wins / len(d)) if len(d) else 0.0}

    return {b: {"pairs": count(wide, b), "series": count(series, b)} for b in wide.columns if b != reference}
