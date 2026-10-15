"""F2 demand patterns: demand classes (Syntetos-Boylan-Croston), weekday / month / SNAP profiles,
event effects and price-cut effects.

Everything reads the slice through the `train_days` view (sql/f2_training_days.sql): training days
only (the holdout is never read) and only days the item was listed. Settings come from
config/demand_patterns.yaml. Calendar figures are "units per listed series-day" (total units / series
on sale), so they do not move just because more items were listed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forecast import config, prep

CLASSES = ["smooth", "erratic", "intermittent", "lumpy", "too_few_sales"]
WEEKDAYS = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _sql(name: str, **values) -> str:
    sql = (config.SQL_DIR / name).read_text(encoding="utf-8")
    for key, value in values.items():
        sql = sql.replace("{" + key + "}", str(value))
    return sql


# --- reading ------------------------------------------------------------------------------------

def open_training_days(parquet: Path, holdout_days: int, memory_limit: str = "2GB", threads: int = 4):
    """Open DuckDB with the `train_days` view. The last training day is the last day in the file
    minus holdout_days. Returns (connection, last training d_num)."""
    con = prep.connect(memory_limit, threads)
    path = Path(parquet).as_posix()
    last_day = con.execute(f"SELECT MAX(d_num) FROM read_parquet('{path}')").fetchone()[0]
    last_train = int(last_day) - int(holdout_days)
    con.execute(_sql("f2_training_days.sql", parquet=path, last_train=last_train))
    return con, last_train


def series_stats(con) -> pd.DataFrame:
    return con.execute(_sql("f2_series_stats.sql")).df()


def daily_totals(con, by_dept: bool = False) -> pd.DataFrame:
    """One row per training day (per department if by_dept), with is_event = any event that day."""
    sql = _sql("f2_daily_totals.sql",
               dept_col="dept_id," if by_dept else "",
               dept_group=", dept_id" if by_dept else "")
    df = con.execute(sql).df()
    df["date"] = pd.to_datetime(df["date"])
    df["is_event"] = df["event_name_1"].notna() | df["event_name_2"].notna()
    return df


def weekly_series(con) -> pd.DataFrame:
    return con.execute(_sql("f2_weekly_series.sql")).df()


# --- demand classes -----------------------------------------------------------------------------

def classify(stats: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add ADI (listed days / days with sales), CV² (sample variance / mean², non-zero days) and the
    demand class. Values on a cut-off count as high. Fewer than min_nonzero_days -> too_few_sales."""
    out = stats.copy()
    nz = out["nonzero_days"].astype(float)
    out["adi"] = np.where(nz > 0, out["listed_days"] / nz.where(nz > 0), np.nan)
    enough = out["nonzero_days"] >= cfg["min_nonzero_days"]
    out["cv2"] = np.where(enough, out["var_nz"] / out["mean_nz"] ** 2, np.nan)
    high_adi = out["adi"] >= cfg["adi_cutoff"]
    high_cv2 = out["cv2"] >= cfg["cv2_cutoff"]
    out["demand_class"] = np.select(
        [~enough, ~high_adi & ~high_cv2, ~high_adi & high_cv2, high_adi & ~high_cv2],
        ["too_few_sales", "smooth", "erratic", "intermittent"], default="lumpy")
    return out


def class_summary(classes: pd.DataFrame) -> list[dict]:
    """Series and units per class (share of series and of training units)."""
    g = classes.groupby("demand_class").agg(n_series=("id", "size"), units=("units", "sum"))
    g = g.reindex(CLASSES, fill_value=0)
    return [{"demand_class": c, "n_series": int(r.n_series), "units": int(r.units),
             "share_series": float(r.n_series / g["n_series"].sum()),
             "share_units": float(r.units / g["units"].sum())} for c, r in g.iterrows()]


# --- weekday, month, SNAP -----------------------------------------------------------------------

def _rate(df: pd.DataFrame) -> float:
    days = df["listed_series"].sum()
    return float(df["units"].sum() / days) if days else float("nan")


def calendar_profile(daily: pd.DataFrame, key: str) -> pd.DataFrame:
    """Units per listed series-day by `key` (e.g. weekday) on non-event days; index = rate relative to
    the rate over all non-event days."""
    plain = daily[~daily["is_event"]]
    g = plain.groupby(key).agg(units=("units", "sum"), series_days=("listed_series", "sum"))
    g["rate"] = g["units"] / g["series_days"]
    g["index"] = g["rate"] / _rate(plain)
    if key == "weekday":
        g = g.reindex([w for w in WEEKDAYS if w in g.index])
    return g.reset_index()


def full_years(daily: pd.DataFrame) -> list[int]:
    """Calendar years covered from 1 January to 31 December."""
    span = daily.groupby("year")["date"].agg(["min", "max"])
    full = span[(span["min"].dt.strftime("%m-%d") == "01-01") & (span["max"].dt.strftime("%m-%d") == "12-31")]
    return [int(y) for y in full.index]


def month_profile(daily: pd.DataFrame) -> pd.DataFrame:
    """Month index = month rate / that year's rate, averaged over full years only, so sales growth
    and the partial first and last years do not leak into the seasonal shape. Non-event days."""
    years = full_years(daily)
    plain = daily[~daily["is_event"] & daily["year"].isin(years)]
    g = plain.groupby(["year", "month"]).agg(units=("units", "sum"), series_days=("listed_series", "sum"))
    y = plain.groupby("year").agg(units=("units", "sum"), series_days=("listed_series", "sum"))
    ratio = (g["units"] / g["series_days"]) / (y["units"] / y["series_days"]).reindex(
        g.index.get_level_values("year")).to_numpy()
    out = ratio.groupby("month").agg(["mean", "min", "max"]).rename(columns={"mean": "index"})
    out["n_years"] = len(years)
    return out.reset_index()


def year_rates(daily: pd.DataFrame) -> list[dict]:
    """Per year (all training days): units per listed series-day, total units per day and series on sale per day."""
    g = daily.groupby("year").agg(units=("units", "sum"), series_days=("listed_series", "sum"),
                                  days=("d_num", "size"))
    return [{"year": int(y), "days": int(r.days), "rate": float(r.units / r.series_days),
             "units_per_day": float(r.units / r.days), "listed_series_per_day": float(r.series_days / r.days)}
            for y, r in g.iterrows()]


def snap_effect(daily: pd.DataFrame) -> dict:
    """SNAP vs non-SNAP days (non-event days): rates and lift, overall and per weekday."""
    plain = daily[~daily["is_event"]]

    def lift(df):
        s, n = _rate(df[df["snap"] == 1]), _rate(df[df["snap"] == 0])
        return {"snap_rate": s, "non_snap_rate": n, "lift_pct": 100 * (s / n - 1)}

    by_weekday = [{"weekday": w, **lift(plain[plain["weekday"] == w])}
                  for w in WEEKDAYS if w in set(plain["weekday"])]
    return {**lift(plain), "snap_days": int((plain["snap"] == 1).sum()),
            "non_snap_days": int((plain["snap"] == 0).sum()), "by_weekday": by_weekday}


def snap_by_dept(daily_dept: pd.DataFrame) -> list[dict]:
    return [{"dept_id": d, **{k: v for k, v in snap_effect(g).items() if k != "by_weekday"}}
            for d, g in daily_dept.groupby("dept_id")]


# --- events -------------------------------------------------------------------------------------

def event_comparison_days(daily: pd.DataFrame, d_num: int, cfg: dict) -> list[int]:
    """Same weekday within +-event_window_weeks, present in the (training) data, not an event day
    and, if event_match_snap, with the same SNAP status as the event day."""
    day = daily.set_index("d_num")
    offsets = [7 * k for k in range(1, cfg["event_window_weeks"] + 1)]
    candidates = sorted(d_num + s * o for o in offsets for s in (-1, 1))
    keep = []
    for c in candidates:
        if c not in day.index:
            continue
        if cfg["event_comparison_excludes_event_days"] and day.at[c, "is_event"]:
            continue
        if cfg["event_match_snap"] and day.at[c, "snap"] != day.at[d_num, "snap"]:
            continue
        keep.append(int(c))
    return keep


def event_effects(daily: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per event occurrence: rate on the day vs pooled rate on its comparison days. Per event name:
    number of occurrences used and mean / min / max lift. A day with two events counts for both."""
    day = daily.set_index("d_num")
    rows = []
    for d, r in day[day["is_event"]].iterrows():
        comp = event_comparison_days(daily, d, cfg)
        if not comp:
            continue
        rate = r["units"] / r["listed_series"]
        comp_rate = day.loc[comp, "units"].sum() / day.loc[comp, "listed_series"].sum()
        for name, etype in [(r["event_name_1"], r["event_type_1"]), (r["event_name_2"], r["event_type_2"])]:
            if isinstance(name, str):
                rows.append({"d_num": int(d), "date": r["date"].date().isoformat(), "event_name": name,
                             "event_type": etype, "snap": int(r["snap"]), "rate": float(rate),
                             "comparison_rate": float(comp_rate), "n_comparison_days": len(comp),
                             "lift_pct": float(100 * (rate / comp_rate - 1))})
    occ = pd.DataFrame(rows, columns=["d_num", "date", "event_name", "event_type", "snap", "rate",
                                      "comparison_rate", "n_comparison_days", "lift_pct"])
    by_event = (occ.groupby(["event_name", "event_type"])["lift_pct"]
                .agg(n=("size"), mean_lift_pct="mean", min_lift_pct="min", max_lift_pct="max")
                .reset_index().sort_values("mean_lift_pct", ascending=False, ignore_index=True))
    return occ, by_event


# --- price cuts ---------------------------------------------------------------------------------

def price_windows(weekly: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find price cuts and no-change control windows.

    A window around week t: before = weeks t-B..t-1 at one unchanged price p0; effect = weeks
    t..t+A-1. Cut: price in week t at least min_cut_pct below p0. Control: every week of the window
    at p0. All weeks must exist consecutively (and, if full_weeks_only, have 7 listed training days).
    """
    B, A = cfg["before_weeks"], cfg["after_weeks"]
    w = weekly.sort_values(["id", "wk"]).reset_index(drop=True)
    g = w.groupby("id", sort=False)
    offsets = list(range(-B, A))
    shifted = {k: g[["wk", "price", "price_min", "units", "n_days"]].shift(-k) for k in offsets if k != 0}
    shifted[0] = w[["wk", "price", "price_min", "units", "n_days"]]

    ok = pd.Series(True, index=w.index)
    for k in offsets:
        ok &= shifted[k]["wk"] == w["wk"] + k
        if cfg["full_weeks_only"]:
            ok &= shifted[k]["n_days"] == 7
        ok &= shifted[k]["price"] == shifted[k]["price_min"]  # one price within each week
    p0 = shifted[-1]["price"]
    for k in range(-B, -1):
        ok &= shifted[k]["price"] == p0

    cut_pct = 100 * (1 - w["price"] / p0)
    is_cut = ok & (cut_pct >= cfg["min_cut_pct"] - 1e-9)
    is_control = ok.copy()
    for k in range(0, A):
        is_control &= shifted[k]["price"] == p0

    base_units = sum(shifted[k]["units"] for k in range(-B, 0))
    effect_units = sum(shifted[k]["units"] for k in range(0, A))
    out = pd.DataFrame({
        "id": w["id"], "dept_id": w["dept_id"], "wk": w["wk"], "price_before": p0, "price_cut": w["price"],
        "cut_pct": cut_pct, "base_units": base_units, "effect_units": effect_units,
        "base_rate": base_units / (7 * B), "effect_rate": effect_units / (7 * A),
        "price_held": shifted[A - 1]["price"] <= w["price"] if A > 1 else True,
    })
    cuts = out[is_cut].reset_index(drop=True)
    controls = out[is_control].drop(columns=["price_cut", "cut_pct", "price_held"]).reset_index(drop=True)
    for df in (cuts, controls):
        df[["base_units", "effect_units", "wk"]] = df[["base_units", "effect_units", "wk"]].astype("int64")
    return cuts, controls


def cut_profile(weekly: pd.DataFrame, windows: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Pooled units per listed day and mean price relative to the pre-cut price, for each week from
    -weeks to +weeks around the windows' cut week (week 0). Only weeks present in the training data
    count; n_windows says how many windows had that week."""
    w = weekly[["id", "wk", "price", "units", "n_days"]]
    base = windows[["id", "wk", "price_before"]].rename(columns={"wk": "wk0"})
    rows = []
    for k in range(-weeks, weeks + 1):
        m = base.assign(wk=base["wk0"] + k).merge(w, on=["id", "wk"], how="inner")
        days = m["n_days"].sum()
        rows.append({"week": k, "n_windows": len(m),
                     "units_per_day": float(m["units"].sum() / days) if days else None,
                     "price_vs_before": float((m["price"] / m["price_before"]).mean()) if len(m) else None})
    return pd.DataFrame(rows)


def price_changes_by_sales(con) -> list[dict]:
    """Full weeks with and without sales: how many there are and in how many the price changed."""
    df = con.execute(_sql("f2_price_change_vs_sales.sql")).df()
    return [{"zero_sales_week": bool(r.zero_sales_week), "n_weeks": int(r.n_weeks),
             "n_changes": int(r.n_changes), "change_rate": float(r.n_changes / r.n_weeks)}
            for r in df.itertuples()]


def _pooled(df: pd.DataFrame, cfg: dict) -> dict:
    """Pooled lift = (effect units per effect day) / (baseline units per baseline day) - 1 over all
    windows; median_ratio = median per-window effect/base rate among windows with baseline sales."""
    n = len(df)
    if n == 0:
        return {"n": 0, "n_series": 0, "pooled_lift_pct": None, "median_ratio": None}
    eff = df["effect_units"].sum() / (n * 7 * cfg["after_weeks"])
    base = df["base_units"].sum() / (n * 7 * cfg["before_weeks"])
    has_base = df[df["base_rate"] > 0]
    return {"n": n, "n_series": int(df["id"].nunique()),
            "pooled_lift_pct": float(100 * (eff / base - 1)) if base > 0 else None,
            "median_ratio": float((has_base["effect_rate"] / has_base["base_rate"]).median())
            if len(has_base) else None}


def price_cut_summary(cuts: pd.DataFrame, controls: pd.DataFrame, cfg: dict) -> dict:
    edges = cfg["cut_size_buckets_pct"]
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi == edges[-1]
        sel = cuts[(cuts["cut_pct"] >= lo - 1e-9) & ((cuts["cut_pct"] <= hi) if last else (cuts["cut_pct"] < hi - 1e-9))]
        p = _pooled(sel, cfg)
        buckets.append({"label": f"{lo}%+" if last else f"{lo}-{hi}%", "n_cuts": p.pop("n"), **p,
                        "share_price_held": float(sel["price_held"].mean()) if len(sel) else None})
    allc = _pooled(cuts, cfg)
    ctrl = _pooled(controls, cfg)
    return {"buckets": buckets,
            "all_cuts": {"n_cuts": allc.pop("n"), **allc,
                         "share_price_held": float(cuts["price_held"].mean()) if len(cuts) else None},
            "control": {"n_windows": ctrl.pop("n"), **ctrl}}
