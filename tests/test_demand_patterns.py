"""F2 demand patterns, checked on small hand-made day tables written to parquet (tests/conftest.py)."""
import math

import numpy as np
import pandas as pd
import pytest

from forecast import config
from forecast import demand_patterns as dp

A = "FOODS_1_001_CA_1_evaluation"
B = "FOODS_1_002_CA_1_evaluation"
C = "FOODS_2_001_CA_2_evaluation"

CLS = {"adi_cutoff": 1.32, "cv2_cutoff": 0.49, "cv2_ddof": 1, "min_nonzero_days": 2}
CAL = {"event_window_weeks": 4, "event_match_snap": True, "event_comparison_excludes_event_days": True}
PRICE = {"min_cut_pct": 5, "before_weeks": 4, "after_weeks": 2,
         "cut_size_buckets_pct": [5, 10, 20, 100], "full_weeks_only": True}


def open_days(path, holdout_days=0):
    return dp.open_training_days(path, holdout_days, memory_limit="512MB", threads=1)


def test_real_config_has_the_settings_f2_needs():
    cfg = config.load("demand_patterns")
    assert set(CLS) <= set(cfg["classification"])
    assert set(CAL) <= set(cfg["calendar_effects"])
    assert set(PRICE) <= set(cfg["price_cuts"])
    assert config.load("holdout")["holdout_days"] == 28


# --- holdout and listing ------------------------------------------------------------------------

def test_holdout_days_are_never_read(write_days):
    # 21 days, last 7 are the holdout and carry huge sales that must not show up anywhere.
    path = write_days({A: [1] * 14 + [1000] * 7, B: [0, 2] * 7 + [1000] * 7})
    con, last_train = open_days(path, holdout_days=7)
    assert last_train == 14
    assert con.execute("SELECT MAX(d_num) FROM train_days").fetchone()[0] == 14
    stats = dp.series_stats(con).set_index("id")
    assert stats.loc[A, "units"] == 14 and stats.loc[B, "units"] == 14
    daily = dp.daily_totals(con)
    assert daily["d_num"].max() == 14 and daily["units"].max() < 1000
    weekly = dp.weekly_series(con)
    assert weekly["units"].max() < 1000


def test_holdout_cut_off_comes_from_the_last_day_in_the_file(write_days):
    path = write_days({A: [1] * 30})
    _, last_train = open_days(path, holdout_days=28)
    assert last_train == 2


def test_unlisted_days_are_dropped(write_days):
    path = write_days({A: [None, None, None, 2, 0, 1, 0], B: [1, 1, 1, 1, 1, 1, 1]})
    con, _ = open_days(path)
    stats = dp.series_stats(con).set_index("id")
    assert stats.loc[A, "listed_days"] == 4
    assert stats.loc[A, "nonzero_days"] == 2
    daily = dp.daily_totals(con).set_index("d_num")
    assert daily.loc[1, "listed_series"] == 1 and daily.loc[4, "listed_series"] == 2


# --- demand classes -----------------------------------------------------------------------------

def classes_of(rows):
    stats = pd.DataFrame(rows, columns=["id", "listed_days", "nonzero_days", "mean_nz", "var_nz"])
    return dp.classify(stats, CLS).set_index("id")


def test_classification_quadrants():
    out = classes_of([
        ("smooth", 10, 10, 5.0, 1.0),        # ADI 1.0, CV² 0.04
        ("erratic", 10, 10, 2.0, 4.0),       # ADI 1.0, CV² 1.0
        ("intermittent", 30, 10, 5.0, 1.0),  # ADI 3.0, CV² 0.04
        ("lumpy", 30, 10, 2.0, 4.0),         # ADI 3.0, CV² 1.0
    ])
    assert out["demand_class"].to_dict() == {
        "smooth": "smooth", "erratic": "erratic", "intermittent": "intermittent", "lumpy": "lumpy"}
    assert out.loc["lumpy", "adi"] == 3.0 and out.loc["lumpy", "cv2"] == 1.0


def test_values_on_a_cut_off_count_as_high():
    out = classes_of([
        ("adi_on_cutoff", 33, 25, 5.0, 1.0),   # ADI exactly 1.32
        ("cv2_on_cutoff", 10, 10, 10.0, 49.0),  # CV² exactly 0.49
        ("just_below", 131, 100, 10.0, 48.9),   # ADI 1.31, CV² 0.489
    ])
    assert out.loc["adi_on_cutoff", "demand_class"] == "intermittent"
    assert out.loc["cv2_on_cutoff", "demand_class"] == "erratic"
    assert out.loc["just_below", "demand_class"] == "smooth"


def test_too_few_sales():
    out = classes_of([("one_sale", 50, 1, 3.0, np.nan), ("no_sale", 50, 0, np.nan, np.nan)])
    assert set(out["demand_class"]) == {"too_few_sales"}
    assert out.loc["one_sale", "adi"] == 50.0
    assert math.isnan(out.loc["no_sale", "adi"]) and math.isnan(out.loc["no_sale", "cv2"])


def test_cv2_uses_sample_variance_of_nonzero_days(write_days):
    # Listed days: 0, 1, 0, 3 -> non-zero sales [1, 3], mean 2, sample variance 2, CV² 0.5; ADI 4/2.
    path = write_days({A: [None, 0, 1, 0, 3]})
    con, _ = open_days(path)
    out = dp.classify(dp.series_stats(con), CLS).set_index("id")
    assert out.loc[A, "adi"] == 2.0
    assert out.loc[A, "cv2"] == pytest.approx(0.5)
    assert out.loc[A, "demand_class"] == "lumpy"


# --- weekday, month and SNAP --------------------------------------------------------------------

def test_weekday_profile_is_units_per_listed_series_day_without_event_days(write_days):
    # 14 days from Saturday. A sells 2 on Saturdays, 1 otherwise; B is listed only in week 2 and
    # sells 1 every day. Day 8 (a Saturday) is an event day with huge sales and must be ignored.
    a = [2, 1, 1, 1, 1, 1, 1, 100, 1, 1, 1, 1, 1, 1]
    b = [None] * 7 + [1] * 7
    path = write_days({A: a, B: b}, events={8: "Big"})
    con, _ = open_days(path)
    prof = dp.calendar_profile(dp.daily_totals(con), "weekday").set_index("weekday")
    assert prof.loc["Saturday", "series_days"] == 1  # day 1 only (A); day 8 is an event day
    assert prof.loc["Saturday", "rate"] == 2.0
    assert prof.loc["Sunday", "series_days"] == 3 and prof.loc["Sunday", "rate"] == 1.0
    overall = (2 + 12 + 6) / (7 + 12)  # non-event units / non-event listed series-days
    assert prof.loc["Saturday", "index"] == pytest.approx(2.0 / overall)


def test_month_profile_is_relative_to_each_full_year():
    # Two full years; year 2 sells twice as much, and March is always 50% above the rest.
    dates = pd.date_range("2012-01-01", "2013-12-31")
    rate = np.where(dates.year == 2013, 2.0, 1.0) * np.where(dates.month == 3, 1.5, 1.0)
    daily = pd.DataFrame({"date": dates, "year": dates.year, "month": dates.month,
                          "units": rate * 10, "listed_series": 10, "is_event": False})
    # a partial year that would distort the profile if it were used
    extra = pd.DataFrame({"date": [pd.Timestamp("2014-01-01")], "year": [2014], "month": [1],
                          "units": [1e6], "listed_series": [10], "is_event": [False]})
    both = pd.concat([daily, extra], ignore_index=True)
    assert dp.full_years(both) == [2012, 2013]
    prof = dp.month_profile(both).set_index("month")
    year_rate = (1.5 * 31 + 1.0 * (365 - 31)) / 365  # 2013 is not a leap year; 2012 is
    assert prof.loc[3, "index"] == pytest.approx(1.5 / year_rate, rel=2e-3)
    assert prof.loc[1, "index"] == pytest.approx(1.0 / year_rate, rel=2e-3)


def test_snap_effect(write_days):
    # Starts 2011-02-05 (a Saturday): days 1-6 are SNAP (5th-10th), days 7-14 are not.
    a = [3] * 6 + [1] * 8
    path = write_days({A: a}, start="2011-02-05")
    con, _ = open_days(path)
    snap = dp.snap_effect(dp.daily_totals(con))
    assert snap["snap_rate"] == 3.0 and snap["non_snap_rate"] == 1.0
    assert snap["lift_pct"] == pytest.approx(200.0)
    by_day = {r["weekday"]: r for r in snap["by_weekday"]}
    assert by_day["Saturday"]["lift_pct"] == pytest.approx(200.0)


# --- events -------------------------------------------------------------------------------------

def event_daily(n_days, events, snap_days=(), units=None):
    d = np.arange(1, n_days + 1)
    units = units if units is not None else np.full(n_days, 10.0)
    return pd.DataFrame({
        "d_num": d, "date": pd.Timestamp("2012-01-07") + pd.to_timedelta(d - 1, "D"),
        "wday": (d - 1) % 7 + 1, "snap": [int(x in snap_days) for x in d],
        "event_name_1": [events.get(x) for x in d], "event_name_2": None,
        "event_type_1": ["Test" if x in events else None for x in d],
        "event_type_2": None, "units": units, "listed_series": 10,
        "is_event": [x in events for x in d],
    })


def test_event_comparison_days_same_weekday_within_window():
    daily = event_daily(60, {30: "E"})
    assert dp.event_comparison_days(daily, 30, CAL) == [2, 9, 16, 23, 37, 44, 51, 58]


def test_event_comparison_skips_other_events_other_snap_and_missing_days():
    # Day 16 is another event, day 44 is a SNAP day (event day is not), data end at day 50.
    daily = event_daily(50, {30: "E", 16: "F"}, snap_days={44})
    assert dp.event_comparison_days(daily, 30, CAL) == [2, 9, 23, 37]


def test_event_effect_vs_comparison_days():
    units = np.full(60, 10.0)
    units[29] = 15.0   # event day 30: +50% vs the comparison days
    occ, by_event = dp.event_effects(event_daily(60, {30: "E"}, units=units), CAL)
    assert len(occ) == 1 and occ.iloc[0]["n_comparison_days"] == 8
    assert occ.iloc[0]["lift_pct"] == pytest.approx(50.0)
    assert by_event.set_index("event_name").loc["E", "mean_lift_pct"] == pytest.approx(50.0)


def test_event_comparison_never_reaches_into_the_holdout(write_days):
    # 42 days; holdout = last 7 (days 36-42). Event on day 29: comparison days 1, 8, 15, 22 and 36
    # would be in the window, but 36 is holdout.
    path = write_days({A: [1] * 42}, events={29: "E"})
    con, _ = open_days(path, holdout_days=7)
    daily = dp.daily_totals(con)
    assert dp.event_comparison_days(daily, 29, {**CAL, "event_match_snap": False}) == [1, 8, 15, 22]


# --- price cuts ---------------------------------------------------------------------------------

def price_series(weekly_prices, weekly_units):
    prices = [p for p in weekly_prices for _ in range(7)]
    units = [u for u in weekly_units for _ in range(7)]
    return prices, units


def test_price_cut_detected_with_before_and_after_rates(write_days):
    # 4 weeks at 2.00 selling 1/day, then a 25% cut to 1.50 for 2 weeks selling 3/day.
    p, u = price_series([2.0] * 4 + [1.5] * 2, [1] * 4 + [3] * 2)
    con, _ = open_days(write_days({A: u}, prices_by_series={A: p}))
    cuts, controls = dp.price_windows(dp.weekly_series(con), PRICE)
    assert len(cuts) == 1 and len(controls) == 0
    c = cuts.iloc[0]
    assert c["cut_pct"] == pytest.approx(25.0)
    assert c["base_rate"] == 1.0 and c["effect_rate"] == 3.0
    assert bool(c["price_held"])


def test_small_drops_changed_baselines_and_partial_weeks_are_not_cuts(write_days):
    small, u = price_series([2.0] * 4 + [1.92] * 2, [1] * 6)             # 4% drop
    moved, _ = price_series([2.1, 2.0, 2.0, 2.0, 1.5, 1.5], [1] * 6)       # baseline not one price
    partial, _ = price_series([2.0] * 4 + [1.5] * 2, [1] * 6)
    con, _ = open_days(write_days(
        {A: u, B: u, C: u[:-1] + [None]},  # C: last day of the after window not listed
        prices_by_series={A: small, B: moved, C: partial}))
    cuts, _ = dp.price_windows(dp.weekly_series(con), PRICE)
    assert cuts.empty


def test_cut_week_split_by_the_holdout_is_not_used(write_days):
    p, u = price_series([2.0] * 4 + [1.5] * 2, [1] * 6)
    con, _ = open_days(write_days({A: u}, prices_by_series={A: p}), holdout_days=3)
    cuts, _ = dp.price_windows(dp.weekly_series(con), PRICE)
    assert cuts.empty


def test_control_windows_have_no_price_change(write_days):
    p, u = price_series([2.0] * 7, [1] * 7)
    con, _ = open_days(write_days({A: u}, prices_by_series={A: p}))
    cuts, controls = dp.price_windows(dp.weekly_series(con), PRICE)
    assert cuts.empty and len(controls) == 2  # windows ending in weeks 6 and 7


def test_price_cut_summary_pools_by_cut_size():
    cuts = pd.DataFrame({
        "id": ["a", "b", "c"], "cut_pct": [6.0, 25.0, 30.0],
        "base_units": [28, 28, 0], "effect_units": [28, 42, 14],
        "base_rate": [1.0, 1.0, 0.0], "effect_rate": [2.0, 3.0, 1.0],
        "price_held": [True, False, True],
    })
    controls = pd.DataFrame({"id": ["a"], "base_units": [28], "effect_units": [14],
                             "base_rate": [1.0], "effect_rate": [1.0]})
    s = dp.price_cut_summary(cuts, controls, PRICE)
    small, big = s["buckets"][0], s["buckets"][2]
    assert small["label"] == "5-10%" and small["n_cuts"] == 1
    assert small["pooled_lift_pct"] == pytest.approx(100.0)
    assert big["label"] == "20%+" and big["n_cuts"] == 2
    # pooled over the 2 cuts: (42 + 14) units in 28 effect days vs 28 units in 56 baseline days
    assert big["pooled_lift_pct"] == pytest.approx(((42 + 14) / 28) / (28 / 56) * 100 - 100)
    assert big["median_ratio"] == pytest.approx(3.0)  # cut c has no baseline sales: left out
    assert big["share_price_held"] == pytest.approx(0.5)
    assert s["buckets"][1]["n_cuts"] == 0
    assert s["control"]["pooled_lift_pct"] == pytest.approx(0.0)


def test_cut_profile_week_by_week_around_the_cut(write_days):
    # 4 weeks at 2.00 selling 1/day, cut to 1.00 for 2 weeks selling 3/day, then 2/day at 1.00.
    p, u = price_series([2.0] * 4 + [1.0] * 3, [1] * 4 + [3, 3, 2])
    con, _ = open_days(write_days({A: u}, prices_by_series={A: p}))
    weekly = dp.weekly_series(con)
    cuts, _ = dp.price_windows(weekly, PRICE)
    prof = dp.cut_profile(weekly, cuts, weeks=5).set_index("week")
    assert prof.loc[-4, "units_per_day"] == 1.0 and prof.loc[0, "units_per_day"] == 3.0
    assert prof.loc[2, "units_per_day"] == 2.0
    assert prof.loc[2, "price_vs_before"] == 0.5
    assert prof.loc[-4, "n_windows"] == 1 and prof.loc[3, "n_windows"] == 0  # no week 3 in the data
    assert -5 not in prof.index or prof.loc[-5, "n_windows"] == 0


def test_price_changes_by_week_with_or_without_sales(write_days):
    # A: price changes in week 2 (has sales) and week 3 (no sales); B never changes.
    p = [2.0] * 7 + [1.5] * 7 + [1.0] * 7
    u = [1] * 14 + [0] * 7
    con, _ = open_days(write_days({A: u, B: u}, prices_by_series={A: p}))
    out = {r["zero_sales_week"]: r for r in dp.price_changes_by_sales(con)}
    assert out[False]["n_weeks"] == 2 and out[False]["n_changes"] == 1
    assert out[True]["n_weeks"] == 2 and out[True]["n_changes"] == 1
