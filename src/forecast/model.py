"""F4 global LightGBM: features as of a forecast origin, training and prediction rows, the holdout
guard, training and forecasting.

One row = (series, origin t, target day t + h), h = 1..horizon. Features use only the panel columns
for days <= t (units, prices, listing) plus calendar facts of the target day, which are published in
advance. One model per backtest origin B, trained on rows from earlier origins whose target days are
all <= B (direct multi-horizon: h is a feature, nothing is fed back recursively).

Guard: any origin or target day at or after the first holdout day raises HoldoutError unless the
caller passes allow_holdout=True (only the one-shot holdout script does). Settings come from
config/model.yaml; the backtest origins, eligibility, metrics and weights are F3's (backtest.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecast import backtest as bt
from forecast import config, prep

FRIDAY = bt.FRIDAY
CALENDAR_FEATURES = ["h", "wday", "month", "mday", "snap", "event_name", "event_type", "days_to_event"]
OTHER_FEATURES = ["zero_share", "days_since_sale", "listed_days", "adi", "cv2", "price_last", "price_ratio",
                  "item_id", "dept_id", "store_id"]
CATEGORICAL = ["event_name", "event_type", "item_id", "dept_id", "store_id"]


class HoldoutError(RuntimeError):
    """A day at or after the first holdout day was requested without the explicit holdout flag."""


def check_days(last_day: int, first_holdout: int, allow_holdout: bool = False) -> None:
    if int(last_day) >= int(first_holdout) and not allow_holdout:
        raise HoldoutError(f"d_{last_day} is in the holdout (starts d_{first_holdout}); "
                           "only the one-shot holdout run may pass allow_holdout=True")


def feature_names(cfg: dict) -> list[str]:
    f = cfg["features"]
    return (CALENDAR_FEATURES + [f"mean_{k}" for k in f["mean_days"]]
            + [f"wday_mean_{w}" for w in f["weekday_mean_weeks"]] + OTHER_FEATURES)


# --- data ---------------------------------------------------------------------------------------

@dataclass
class Data:
    panel: bt.Panel          # F3 matrices; ends before the holdout unless loaded with allow_holdout
    cal: pd.DataFrame        # calendar features per day (index d_num)
    first_holdout: int
    item: np.ndarray         # integer codes per series (panel row order)
    dept: np.ndarray
    store: np.ndarray
    cls_cfg: dict            # F2 class settings (ADI / CV² definitions)


def calendar_from_parquet(parquet: Path, last_day: int) -> pd.DataFrame:
    """Calendar columns of the slice (fixtures), days <= last_day only."""
    con = prep.connect("512MB", 1)
    df = con.execute(f"""
        SELECT DISTINCT d_num, date, wday, month, snap, event_name_1, event_type_1
        FROM read_parquet('{Path(parquet).as_posix()}') WHERE d_num <= {int(last_day)} ORDER BY d_num""").df()
    con.close()
    return df


def calendar_from_csv(path: Path, state: str) -> pd.DataFrame:
    """The raw M5 calendar (no sales in it): every date with its events and the state's SNAP flag."""
    con = prep.connect("512MB", 1)
    df = con.execute(f"""
        SELECT CAST(substr(d, 3) AS INTEGER) AS d_num, CAST(date AS DATE) AS date, wday, month,
               snap_{state} AS snap, event_name_1, event_type_1
        FROM read_csv('{Path(path).as_posix()}', header = true, all_varchar = true,
                      nullstr = '') ORDER BY d_num""").df()
    con.close()
    for c in ("wday", "month", "snap"):
        df[c] = df[c].astype(int)
    return df


def calendar_table(cal: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per day: weekday, month, day of month, SNAP, event name / type codes (0 = none), the signed
    distance to the nearest event day (positive = ahead, ties count as ahead, cut at +-cap) and
    whether the stores are closed (config closed_events)."""
    cal = cal.sort_values("d_num").reset_index(drop=True)
    names = sorted(cal["event_name_1"].dropna().unique())
    types = sorted(cal["event_type_1"].dropna().unique())
    cap = cfg["calendar"]["event_distance_cap_days"]
    d = cal["d_num"].to_numpy()
    ev = d[cal["event_name_1"].notna().to_numpy()]
    nxt = np.searchsorted(ev, d, side="left")
    ahead = np.where(nxt < len(ev), ev[np.minimum(nxt, len(ev) - 1)] - d, np.inf) if len(ev) else np.full(len(d), np.inf)
    prv = np.searchsorted(ev, d, side="right") - 1
    behind = np.where(prv >= 0, d - ev[np.maximum(prv, 0)], np.inf) if len(ev) else np.full(len(d), np.inf)
    dist = np.where(ahead <= behind, ahead, -behind)
    out = pd.DataFrame({
        "wday": cal["wday"].to_numpy(), "month": cal["month"].to_numpy(),
        "mday": pd.to_datetime(cal["date"]).dt.day.to_numpy(), "snap": cal["snap"].to_numpy(),
        "event_name": cal["event_name_1"].map({n: i + 1 for i, n in enumerate(names)}).fillna(0).to_numpy(),
        "event_type": cal["event_type_1"].map({n: i + 1 for i, n in enumerate(types)}).fillna(0).to_numpy(),
        "days_to_event": np.clip(np.where(np.isinf(dist), cap, dist), -cap, cap),
        "closed": cal["event_name_1"].isin(cfg["closed_events"]).to_numpy(),
    }, index=pd.Index(d, name="d_num"))
    return out


def _codes(values) -> np.ndarray:
    return np.unique(np.asarray(values), return_inverse=True)[1].astype(np.int32)


def load_data(parquet: Path, holdout_days: int, cfg: dict, allow_holdout: bool = False,
              calendar_csv: Path | None = None, state: str = "CA", cls_cfg: dict | None = None,
              memory_limit: str = "2GB", threads: int = 4) -> Data:
    """The F3 panel (training days only, unless allow_holdout) plus calendar features and codes."""
    if allow_holdout:
        panel = bt.load_panel(parquet, 0, memory_limit, threads)
        first_holdout = panel.last_train - int(holdout_days) + 1
    else:
        panel = bt.load_panel(parquet, holdout_days, memory_limit, threads)
        first_holdout = panel.last_train + 1
    raw = calendar_from_csv(calendar_csv, state) if calendar_csv else calendar_from_parquet(parquet, panel.last_train)
    parts = [i.split("_") for i in panel.ids]   # e.g. FOODS_3_090_CA_1_evaluation
    return Data(panel=panel, cal=calendar_table(raw, cfg), first_holdout=first_holdout,
                item=_codes(["_".join(p[:3]) for p in parts]), dept=_codes(panel.dept),
                store=_codes(["_".join(p[3:5]) for p in parts]),
                cls_cfg=cls_cfg or config.load("demand_patterns")["classification"])


# --- features as of the origin ------------------------------------------------------------------

def _mean_listed(block: np.ndarray) -> np.ndarray:
    n = (~np.isnan(block)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.nansum(block, axis=1) / np.maximum(n, 1), np.nan)


def origin_features(data: Data, t: int, cfg: dict, allow_holdout: bool = False) -> dict:
    """Per-series features from days <= t only. Closed days (Christmas) count as unlisted. Weekday
    means are series x 7 arrays, column = M5 wday - 1 (0 = Saturday)."""
    check_days(t, data.first_holdout, allow_holdout)
    p, f = data.panel, cfg["features"]
    c = p.col(t) + 1
    raw = p.units[:, :c]
    hist = raw.astype(float)
    days = np.arange(p.first_d, t + 1)
    closed = data.cal["closed"].reindex(days, fill_value=False).to_numpy()
    hist[:, closed] = np.nan
    wday = p.wday[:c]
    out = {"eligible_days": (~np.isnan(raw)).sum(axis=1)}   # F3 eligibility counts every listed day

    for k in f["mean_days"]:
        out[f"mean_{k}"] = _mean_listed(hist[:, -k:])
    for w in f["weekday_mean_weeks"]:
        block, wd = hist[:, -7 * w:], wday[-7 * w:]
        out[f"wday_mean_{w}"] = np.stack([_mean_listed(block[:, wd == j]) for j in range(1, 8)], axis=1)

    block = hist[:, -f["zero_share_days"]:]
    n = (~np.isnan(block)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["zero_share"] = np.where(n > 0, (block == 0).sum(axis=1) / np.maximum(n, 1), np.nan)
    sold = np.where(hist > 0, np.arange(c), -1).max(axis=1)
    out["days_since_sale"] = np.where(sold >= 0, (c - 1) - sold, np.nan).astype(float)
    out["listed_days"] = (~np.isnan(hist)).sum(axis=1).astype(float)
    stats = bt.intermittency_stats(hist, data.cls_cfg)
    out["adi"], out["cv2"] = stats["adi"].to_numpy(float), stats["cv2"].to_numpy(float)

    price = p.price[:, :c]
    out["price_last"] = bt.last_known_price(price, wday)
    fridays = np.flatnonzero(wday == FRIDAY)
    if len(fridays):
        lf = fridays[-1]
        ref = _mean_listed(np.asarray(price[:, max(0, lf + 1 - f["price_ref_days"]): lf + 1], dtype=float))
    else:
        ref = np.full(len(price), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["price_ratio"] = out["price_last"] / ref
    return out


# --- rows ---------------------------------------------------------------------------------------

def rows_at(data: Data, t: int, horizon: int, cfg: dict, bt_cfg: dict, with_target: bool,
            allow_holdout: bool = False) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Rows for origin t, one per eligible series (F3 rule: listed days >= min_history_days) and
    h = 1..horizon. with_target: training rows with the actual units, leaving out unlisted and
    closed target days; otherwise every target day (prediction rows, no target).
    Returns X (float32, columns = feature_names(cfg)), y and meta (series row, origin, d_num)."""
    check_days(t + horizon, data.first_holdout, allow_holdout)
    f = origin_features(data, t, cfg, allow_holdout)
    s_idx = np.flatnonzero(f["eligible_days"] >= bt_cfg["min_history_days"])
    days = np.arange(t + 1, t + horizon + 1)
    cal = data.cal.loc[days]
    S, H = len(s_idx), horizon
    keep = np.ones((S, H), dtype=bool)
    y = None
    if with_target:
        p = data.panel
        assert t + horizon <= p.last_train, "training targets beyond the loaded days"
        y_mat = p.units[s_idx, p.col(t + 1): p.col(t + horizon) + 1]
        keep = ~np.isnan(y_mat) & ~cal["closed"].to_numpy()[None, :]
        y = y_mat[keep].astype(np.float32)
    keep = keep.ravel()

    def per_day(v):          # same value for every series
        return np.broadcast_to(np.asarray(v, dtype=float)[None, :], (S, H)).ravel()[keep]

    def per_series(v):       # same value for every h
        return np.repeat(np.asarray(v, dtype=float)[s_idx], H)[keep]

    cols = {"h": per_day(np.arange(1, H + 1))}
    for k in ("wday", "month", "mday", "snap", "event_name", "event_type", "days_to_event"):
        cols[k] = per_day(cal[k].to_numpy())
    for k in cfg["features"]["mean_days"]:
        cols[f"mean_{k}"] = per_series(f[f"mean_{k}"])
    tw = cal["wday"].to_numpy() - 1
    for w in cfg["features"]["weekday_mean_weeks"]:
        cols[f"wday_mean_{w}"] = f[f"wday_mean_{w}"][s_idx][:, tw].ravel()[keep]
    for k in ("zero_share", "days_since_sale", "listed_days", "adi", "cv2", "price_last", "price_ratio"):
        cols[k] = per_series(f[k])
    cols["item_id"], cols["dept_id"], cols["store_id"] = (per_series(data.item), per_series(data.dept),
                                                          per_series(data.store))
    names = feature_names(cfg)
    X = np.empty((int(keep.sum()), len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        X[:, j] = cols[name]
    meta = {"series": np.repeat(s_idx, H)[keep].astype(np.int32),
            "origin": np.full(int(keep.sum()), t, dtype=np.int32),
            "d_num": np.tile(days, S)[keep].astype(np.int32)}
    return X, y, meta


def training_origins(backtest_origin: int, tcfg: dict) -> list[int]:
    """Oldest first; the newest is backtest_origin - latest_gap_days."""
    newest = backtest_origin - tcfg["latest_gap_days"]
    return [newest - tcfg["step_days"] * k for k in range(tcfg["n_origins"] - 1, -1, -1)]


def estimate_table_gb(n_series: int, horizon: int, n_origins: int, n_features: int) -> float:
    """Upper bound of the training table: every series listed on every target day, float32
    features plus the target."""
    return n_series * horizon * n_origins * (n_features + 1) * 4 / 1e9


def check_table_size(gb: float, max_gb: float) -> None:
    if gb > max_gb:
        raise MemoryError(f"training table estimate {gb:.2f} GB is above the configured {max_gb} GB")


def training_table(data: Data, backtest_origin: int, cfg: dict, bt_cfg: dict,
                   allow_holdout: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Training rows for one backtest origin B: every configured training origin, all target days <= B."""
    H, tcfg = bt_cfg["horizon_days"], cfg["training_origins"]
    assert tcfg["latest_gap_days"] >= H, "latest_gap_days < horizon would put targets after B"
    check_days(backtest_origin, data.first_holdout, allow_holdout)
    origins = training_origins(backtest_origin, tcfg)
    n_feat = len(feature_names(cfg))
    check_table_size(estimate_table_gb(len(data.panel.ids), H, len(origins), n_feat), cfg["max_feature_table_gb"])
    cap = len(data.panel.ids) * H * len(origins)
    X = np.empty((cap, n_feat), dtype=np.float32)
    y = np.empty(cap, dtype=np.float32)
    meta = {k: np.empty(cap, dtype=np.int32) for k in ("series", "origin", "d_num")}
    n = 0
    for t in origins:
        Xo, yo, mo = rows_at(data, t, H, cfg, bt_cfg, with_target=True, allow_holdout=allow_holdout)
        X[n: n + len(Xo)], y[n: n + len(Xo)] = Xo, yo
        for k in meta:
            meta[k][n: n + len(Xo)] = mo[k]
        n += len(Xo)
    meta = {k: v[:n] for k, v in meta.items()}
    assert n == 0 or meta["d_num"].max() <= backtest_origin, "a training target is after the backtest origin"
    return X[:n], y[:n], meta


# --- model --------------------------------------------------------------------------------------

def lgb_params(cfg: dict, setting: dict) -> dict:
    params = {"objective": cfg["objective"], **cfg["fixed_params"],
              **{k: v for k, v in setting.items() if k != "name"}}
    if cfg["objective"] == "tweedie":
        params["tweedie_variance_power"] = cfg["tweedie_variance_power"]
    return params


def fit(X: np.ndarray, y: np.ndarray, names: list[str], params: dict, rounds: int) -> lgb.Booster:
    ds = lgb.Dataset(X, y, feature_name=names, categorical_feature=[c for c in CATEGORICAL if c in names],
                     free_raw_data=True, params={"verbosity": params.get("verbosity", -1)})
    return lgb.train(params, ds, num_boost_round=rounds)


def forecast(data: Data, booster: lgb.Booster, t: int, cfg: dict, bt_cfg: dict, rounds: list[int],
             allow_holdout: bool = False) -> dict[int, np.ndarray]:
    """Series x horizon forecasts from origin t after each number of trees in `rounds` (panel row
    order; NaN for series not eligible at t). Closed target days are forecast as 0."""
    H = bt_cfg["horizon_days"]
    X, _, meta = rows_at(data, t, H, cfg, bt_cfg, with_target=False, allow_holdout=allow_holdout)
    closed = data.cal.loc[np.arange(t + 1, t + H + 1), "closed"].to_numpy()
    out = {}
    for r in rounds:
        f = np.full((len(data.panel.ids), H), np.nan)
        pred = np.clip(booster.predict(X, num_iteration=r), 0, None)
        f[meta["series"], meta["d_num"] - t - 1] = pred
        f[:, closed] = np.where(np.isnan(f[:, closed]), np.nan, 0.0)
        out[r] = f
    return out


def importance(booster: lgb.Booster) -> dict[str, float]:
    return dict(zip(booster.feature_name(), map(float, booster.feature_importance(importance_type="gain"))))
