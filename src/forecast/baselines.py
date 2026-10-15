"""F3 baselines. Each is a pure function: the daily history up to the forecast origin goes in, a flat or
weekly-repeating forecast for the next `horizon` days comes out.

`hist` is one series (1-D) or many (2-D, one row per series, one column per day, the last column is
the origin). NaN means the item was not listed that day ("not on sale", F1): those days are skipped,
never treated as zero sales. A series never listed gets NaN. The recursive methods loop over days and
work on all series at once, so the whole slice runs in seconds.
"""
from __future__ import annotations

import numpy as np


def _rows(hist) -> tuple[np.ndarray, bool]:
    h = np.asarray(hist, dtype=float)
    return (h[None, :], True) if h.ndim == 1 else (h, False)


def _flat(level: np.ndarray, horizon: int, one: bool) -> np.ndarray:
    out = np.repeat(level[:, None], horizon, axis=1)
    return out[0] if one else out


def _last_listed(h: np.ndarray) -> np.ndarray:
    idx = np.where(~np.isnan(h), np.arange(h.shape[1]), -1).max(axis=1)
    return np.where(idx >= 0, h[np.arange(len(h)), np.maximum(idx, 0)], np.nan)


def naive(hist, horizon: int) -> np.ndarray:
    """The last listed day's value."""
    h, one = _rows(hist)
    return _flat(_last_listed(h), horizon, one)


def seasonal_naive(hist, horizon: int, season: int = 7) -> np.ndarray:
    """Same weekday last week. If that day was unlisted, the same weekday one week earlier, and so on;
    if the weekday was never listed, the naive value."""
    h, one = _rows(hist)
    n = h.shape[1]
    base = np.full((len(h), season), np.nan)
    for j in range(season):
        cols = np.arange(n - season + j, -1, -season)
        cols = cols[cols >= 0]
        if len(cols) == 0:
            continue
        sub = h[:, cols]  # most recent first
        ok = ~np.isnan(sub)
        first = ok.argmax(axis=1)
        base[:, j] = np.where(ok.any(axis=1), sub[np.arange(len(h)), first], np.nan)
    base = np.where(np.isnan(base), _last_listed(h)[:, None], base)
    out = base[:, np.arange(horizon) % season]
    return out[0] if one else out


def moving_average(hist, horizon: int, window: int = 28) -> np.ndarray:
    """Mean of the last `window` listed days (fewer if the series has fewer)."""
    h, one = _rows(hist)
    listed = ~np.isnan(h)
    from_end = np.cumsum(listed[:, ::-1], axis=1)[:, ::-1]  # listed days from here to the origin
    use = listed & (from_end <= window)
    n = use.sum(axis=1)
    total = np.where(use, h, 0.0).sum(axis=1)
    return _flat(np.where(n > 0, total / np.maximum(n, 1), np.nan), horizon, one)


def ses(hist, horizon: int, alpha: float) -> np.ndarray:
    """Simple exponential smoothing; the level starts at the first listed value."""
    h, one = _rows(hist)
    level = np.full(len(h), np.nan)
    for t in range(h.shape[1]):
        y = h[:, t]
        ok = ~np.isnan(y)
        start = ok & np.isnan(level)
        level[start] = y[start]
        upd = ok & ~start
        level[upd] += alpha * (y[upd] - level[upd])
    return _flat(level, horizon, one)


def _croston_parts(h: np.ndarray, alpha: float):
    """Smoothed demand size z and interval p, updated only on days with a sale. The first sale sets
    z to its size and p to the number of listed days up to and including it."""
    n = len(h)
    z, p = np.zeros(n), np.ones(n)
    q = np.zeros(n)  # listed days since the last sale (or since listing)
    seen = np.zeros(n, dtype=bool)
    listed_any = np.zeros(n, dtype=bool)
    for t in range(h.shape[1]):
        y = h[:, t]
        ok = ~np.isnan(y)
        listed_any |= ok
        q += ok
        demand = ok & (y > 0)
        first = demand & ~seen
        later = demand & seen
        z[first], p[first] = y[first], q[first]
        z[later] += alpha * (y[later] - z[later])
        p[later] += alpha * (q[later] - p[later])
        q[demand] = 0
        seen |= demand
    return z, p, seen, listed_any


def croston(hist, horizon: int, alpha: float) -> np.ndarray:
    """Croston: demand size / interval. No sale yet -> 0."""
    h, one = _rows(hist)
    z, p, seen, listed = _croston_parts(h, alpha)
    level = np.where(seen, z / p, np.where(listed, 0.0, np.nan))
    return _flat(level, horizon, one)


def sba(hist, horizon: int, alpha: float) -> np.ndarray:
    """Syntetos-Boylan approximation: Croston x (1 - alpha / 2), which removes Croston's upward bias."""
    return (1 - alpha / 2) * croston(hist, horizon, alpha)


def tsb(hist, horizon: int, alpha_demand: float, alpha_probability: float) -> np.ndarray:
    """Teunter-Syntetos-Babai: demand probability (updated every listed day) x demand size (updated on
    sale days). The first sale sets the size and a probability of 1 / listed days up to it."""
    h, one = _rows(hist)
    n = len(h)
    z, prob, q = np.zeros(n), np.zeros(n), np.zeros(n)
    seen = np.zeros(n, dtype=bool)
    listed_any = np.zeros(n, dtype=bool)
    for t in range(h.shape[1]):
        y = h[:, t]
        ok = ~np.isnan(y)
        listed_any |= ok
        q += ok
        demand = ok & (y > 0)
        first = demand & ~seen
        later = ok & seen
        z[first], prob[first] = y[first], 1.0 / q[first]
        prob[later] += alpha_probability * (demand[later] - prob[later])
        upd = demand & seen
        z[upd] += alpha_demand * (y[upd] - z[upd])
        seen |= demand
    level = np.where(seen, prob * z, np.where(listed_any, 0.0, np.nan))
    return _flat(level, horizon, one)


def forecast_all(hist, horizon: int, cfg: dict) -> dict[str, np.ndarray]:
    """Every baseline named in cfg["baselines"], with the settings from config/backtest.yaml."""
    a = cfg["croston_alpha"]
    run = {
        "naive": lambda h: naive(h, horizon),
        "seasonal_naive": lambda h: seasonal_naive(h, horizon, cfg["season_length_days"]),
        "moving_average": lambda h: moving_average(h, horizon, cfg["moving_average_days"]),
        "ses": lambda h: ses(h, horizon, cfg["ses_alpha"]),
        "croston": lambda h: croston(h, horizon, a),
        "sba": lambda h: sba(h, horizon, a),
        "tsb": lambda h: tsb(h, horizon, cfg["tsb_alpha_demand"], cfg["tsb_alpha_probability"]),
    }
    return {name: run[name](hist) for name in cfg["baselines"]}
