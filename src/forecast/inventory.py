"""F5 inventory layer: periodic-review order-up-to policy on top of the fixed F4 forecasts.

Pure functions on series x day matrices (numpy, vectorised across series; a loop over the <= 28 days).
Day 0 is the first forecast day (origin + 1). Settings and their reasons are in config/inventory.yaml.

Timing of one day: orders due arrive at the start of the day; on a review day an order of
S - (on hand + on order), rounded up to whole units, is placed and arrives `lead` days later (the same
day if lead = 0); then the day's demand is served from stock on hand. Unmet demand is lost (or, as a
sensitivity check, backordered).
S = forecast over the protection period (lead + review days, from the review day) + safety stock.
Safety stock = mean (main method only) + z x sd of the backtest forecast error summed over windows of
protection-period length.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

SS_METHODS = ("bias_adjusted", "z_sd_only")
UNMET = ("lost_sales", "backorders")


class LeakageError(RuntimeError):
    """A safety-stock spread would use errors from days at or after the forecast origin."""


# --- the end-of-horizon rule ---------------------------------------------------------------------

@dataclass(frozen=True)
class Schedule:
    protection: int       # lead + review days
    reviews: list[int]    # review days (0-based) whose protection period fits inside the forecast
    first: int            # first measured day = lead (the first day an order can arrive)
    last: int             # last measured day = last review + protection - 1


def schedule(horizon: int, lead: int, review: int) -> Schedule:
    """Reviews every `review` days from day 0, kept only while the whole protection period lies inside
    the forecast horizon (config end_of_horizon: reviews_that_fit). Days before `lead` are a warm-up."""
    p = int(lead) + int(review)
    if review < 1 or lead < 0 or p > horizon:
        raise ValueError(f"lead {lead} + review {review} does not fit in a {horizon}-day forecast")
    reviews = [t for t in range(0, horizon, review) if t + p - 1 <= horizon - 1]
    return Schedule(protection=p, reviews=reviews, first=int(lead), last=reviews[-1] + p - 1)


# --- safety stock --------------------------------------------------------------------------------

def z_value(service_level: float) -> float:
    return NormalDist().inv_cdf(float(service_level))


def window_errors(residuals: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """residuals: series x origin x horizon (actual - forecast; NaN = no row). Sums over every window of
    `window` consecutive horizon days inside one origin, pooled over origins. A window touching a NaN
    is dropped. Returns per-series mean, sd (ddof 1) and number of windows (NaN when < 2)."""
    r = np.asarray(residuals, dtype=float)
    n, k, h = r.shape
    def win(x):
        c = np.concatenate([np.zeros((n, k, 1)), np.cumsum(x, axis=2)], axis=2)
        return (c[:, :, window:] - c[:, :, : h - window + 1]).reshape(n, -1)
    sums = win(np.nan_to_num(r))
    ok = win(np.isnan(r).astype(float)) == 0
    cnt = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, np.where(ok, sums, 0).sum(axis=1) / cnt, np.nan)
        dev2 = np.where(ok, (sums - mean[:, None]) ** 2, 0).sum(axis=1)
        sd = np.where(cnt > 1, np.sqrt(dev2 / np.maximum(cnt - 1, 1)), np.nan)
    mean = np.where(cnt > 1, mean, np.nan)
    return mean, sd, cnt


def spread_from_backtest(backtest: pd.DataFrame, method: str, ids: np.ndarray, window: int,
                         origins: list[int], forecast_origin: int, horizon: int):
    """Summed-window error mean / sd / count per series in `ids` order, from the backtest rows of
    `method` at the listed origins only. Refuses any origin whose window reaches past the forecast
    origin, so errors on the forecast days themselves (the holdout) can never enter."""
    bad = [o for o in origins if int(o) + int(horizon) > int(forecast_origin)]
    if bad:
        raise LeakageError(f"origins {bad} have error days after the forecast origin d_{forecast_origin}")
    df = backtest[(backtest["method"] == method) & backtest["origin"].isin(origins)]
    pos = pd.Series(np.arange(len(ids)), index=pd.Index(ids))
    row = pos.reindex(df["id"].to_numpy()).to_numpy()
    keep = ~np.isnan(row)
    res = np.full((len(ids), len(origins), horizon), np.nan)
    o_idx = pd.Series(np.arange(len(origins)), index=origins).reindex(df["origin"].to_numpy()).to_numpy()
    res[row[keep].astype(int), o_idx[keep].astype(int), df["h"].to_numpy()[keep].astype(int) - 1] = \
        df["residual"].to_numpy(dtype=float)[keep]
    return window_errors(res, window)


def safety_stock(mean: np.ndarray, sd: np.ndarray, z: float, method: str) -> np.ndarray:
    if method == "bias_adjusted":
        return np.asarray(mean, dtype=float) + z * np.asarray(sd, dtype=float)
    if method == "z_sd_only":
        return z * np.asarray(sd, dtype=float)
    raise ValueError(f"unknown safety stock method {method!r}; expected one of {SS_METHODS}")


# --- simulation ----------------------------------------------------------------------------------

@dataclass
class Sim:
    schedule: Schedule
    start: np.ndarray     # on hand at the start of day 0
    placed: np.ndarray    # units ordered on each day
    arrived: np.ndarray   # units arriving at the start of each day
    served: np.ndarray    # demand served from stock on the day it occurred
    lost: np.ndarray      # demand not served on its day (lost, or backordered)
    on_hand: np.ndarray   # stock on hand at the end of each day (never negative)
    demand: np.ndarray


def _order_up_to(forecast: np.ndarray, ss: np.ndarray, t: int, p: int) -> np.ndarray:
    return np.maximum(forecast[:, t:t + p].sum(axis=1) + ss, 0.0)


def simulate(forecast: np.ndarray, demand: np.ndarray, ss: np.ndarray, lead: int, review: int,
             start_factor: float = 1.0, unmet: str = "lost_sales") -> Sim:
    """Replay `demand` (series x day) under the order-up-to policy built on `forecast` (same shape,
    one forecast made before day 0, never updated) with per-series safety stock `ss`."""
    if unmet not in UNMET:
        raise ValueError(f"unknown unmet-demand rule {unmet!r}; expected one of {UNMET}")
    fc, dem = np.asarray(forecast, dtype=float), np.asarray(demand, dtype=float)
    n, H = dem.shape
    sch = schedule(H, lead, review)
    p, reviews = sch.protection, set(sch.reviews)
    ss = np.asarray(ss, dtype=float)

    start = np.ceil(start_factor * _order_up_to(fc, ss, 0, p) - 1e-9)
    net = start.copy()                        # on hand (negative = backlog when backordering)
    due = np.zeros((n, H + lead + 1))         # arrivals by day
    placed, served, lost, on_hand = (np.zeros((n, H)) for _ in range(4))
    for t in range(H):
        net += due[:, t]
        if t in reviews:
            position = net + due[:, t + 1:].sum(axis=1)
            q = np.ceil(np.maximum(_order_up_to(fc, ss, t, p) - position, 0.0) - 1e-9)
            placed[:, t] = q
            due[:, t + lead] += q
            if lead == 0:
                net += q
        avail = np.maximum(net, 0.0)
        served[:, t] = np.minimum(avail, dem[:, t])
        lost[:, t] = dem[:, t] - served[:, t]
        net = net - dem[:, t] if unmet == "backorders" else net - served[:, t]
        on_hand[:, t] = np.maximum(net, 0.0)
    return Sim(schedule=sch, start=start, placed=placed, arrived=due[:, :H], served=served, lost=lost,
               on_hand=on_hand, demand=dem)


# --- measures ------------------------------------------------------------------------------------

def series_measures(sim: Sim, price: np.ndarray, holding_rate_per_year: float, stockout_share: float) -> pd.DataFrame:
    """Per-series totals over the measured days (warm-up and days after the last covered day excluded).
    Holding cost on stock on hand at the end of each day; lost-sales cost per unit not served on its day.
    A cycle = the days one review's order protects (review + lead .. review + protection - 1)."""
    s = sim.schedule
    sl = slice(s.first, s.last + 1)
    price = np.asarray(price, dtype=float)
    dem, served, lost, oh = sim.demand[:, sl], sim.served[:, sl], sim.lost[:, sl], sim.on_hand[:, sl]
    days = dem.shape[1]
    cycles_short = np.zeros(len(price), dtype=int)
    for r in s.reviews:
        cycles_short += (sim.lost[:, r + s.first: r + s.protection] > 0).any(axis=1)
    avg_units = oh.mean(axis=1)
    holding = oh.sum(axis=1) * price * holding_rate_per_year / 365.0
    lost_cost = lost.sum(axis=1) * price * stockout_share
    return pd.DataFrame({
        "demand": dem.sum(axis=1), "served": served.sum(axis=1), "lost": lost.sum(axis=1),
        "days": days, "stockout_days": (lost > 0).sum(axis=1),
        "avg_inv_units": avg_units, "avg_inv_money": avg_units * price,
        "holding_cost": holding, "lost_cost": lost_cost, "total_cost": holding + lost_cost,
        "cycles": len(s.reviews), "cycles_short": cycles_short,
    })


def summarize(m: pd.DataFrame) -> dict:
    """Totals over a group of series: fill rate = served / demand units; cycle service level = share
    of (series, cycle) pairs without a lost unit."""
    t = m.sum(numeric_only=True)
    return {
        "n_series": int(len(m)),
        "demand": float(t["demand"]), "served": float(t["served"]), "lost": float(t["lost"]),
        "fill_rate": float(t["served"] / t["demand"]) if t["demand"] > 0 else None,
        "stockout_days": int(t["stockout_days"]),
        "stockout_day_share": float(t["stockout_days"] / t["days"]) if t["days"] > 0 else None,
        "cycle_service_level": float(1 - t["cycles_short"] / t["cycles"]) if t["cycles"] > 0 else None,
        "avg_inv_units": float(t["avg_inv_units"]), "avg_inv_money": float(t["avg_inv_money"]),
        "holding_cost": float(t["holding_cost"]), "lost_cost": float(t["lost_cost"]),
        "total_cost": float(t["total_cost"]),
    }


def inventory_at_fill_rate(fill_rates, money, target: float):
    """Inventory money a policy needs for `target` fill rate, read off its service-level curve by
    linear interpolation; None when the curve does not reach the target."""
    f, m = np.asarray(fill_rates, dtype=float), np.asarray(money, dtype=float)
    order = np.argsort(f)
    f, m = f[order], m[order]
    if target < f[0] - 1e-12 or target > f[-1] + 1e-12:
        return None
    return float(np.interp(target, f, m))


def top_share(values: np.ndarray, share: float):
    """Share of the total held by the top `share` of series (at least one series)."""
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    total = v.sum()
    if total <= 0:
        return None
    k = max(1, int(round(share * len(v))))
    return float(v[:k].sum() / total)
