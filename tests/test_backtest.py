"""F3 backtest harness: origins and windows, the holdout cut-off, metrics, revenue weights, classes at
the origin, eligibility and leakage. Hand-made day tables written to parquet (tests/conftest.py)."""
import math

import numpy as np
import pandas as pd
import pytest

from forecast import backtest as bt
from forecast import config

A = "FOODS_1_001_CA_1_evaluation"
B = "FOODS_1_002_CA_1_evaluation"
C = "FOODS_2_001_CA_2_evaluation"
nan = np.nan

CLS = {"adi_cutoff": 1.32, "cv2_cutoff": 0.49, "cv2_ddof": 1, "min_nonzero_days": 2}
CFG = {"horizon_days": 7, "n_origins": 2, "min_history_days": 14, "season_length_days": 7,
       "moving_average_days": 7, "ses_alpha": 0.1, "croston_alpha": 0.1, "tsb_alpha_demand": 0.1,
       "tsb_alpha_probability": 0.1, "weight_days": 7, "tie_tolerance": 1e-9,
       "baselines": ["naive", "seasonal_naive", "moving_average", "ses", "croston", "sba", "tsb"],
       "reference_baseline": "seasonal_naive"}


def panel_of(write_days, units, holdout_days=0, **kwargs):
    return bt.load_panel(write_days(units, **kwargs), holdout_days, memory_limit="512MB", threads=1)


# --- origins, windows, holdout ------------------------------------------------------------------

def test_real_config_gives_the_four_origins_before_the_holdout():
    cfg = config.load("backtest")
    last_train = bt.last_train_day(1941, config.load("holdout")["holdout_days"])
    assert last_train == 1913
    org = bt.origins(last_train, cfg["horizon_days"], cfg["n_origins"])
    assert org == [1801, 1829, 1857, 1885]
    wins = [bt.window(o, cfg["horizon_days"]) for o in org]
    assert wins[0] == (1802, 1829) and wins[-1] == (1886, 1913)
    assert max(last for _, last in wins) < 1914
    assert set(CFG) <= set(cfg)


def test_windows_are_back_to_back_and_end_on_the_last_training_day():
    org = bt.origins(100, 10, 3)
    assert org == [70, 80, 90]
    assert [bt.window(o, 10) for o in org] == [(71, 80), (81, 90), (91, 100)]


def test_holdout_days_are_never_loaded(write_days):
    p = panel_of(write_days, {A: [1] * 35 + [1000] * 7, B: [0, 2] * 21}, holdout_days=7)
    assert p.last_train == 35 and p.units.shape == (2, 35)
    assert np.nanmax(p.units) < 1000


def test_panel_marks_unlisted_days_as_missing(write_days):
    p = panel_of(write_days, {A: [None, None, 2, 0], B: [1, 1, 1, 1]})
    assert list(p.ids) == [A, B] and list(p.dept) == ["FOODS_1", "FOODS_1"]
    assert np.isnan(p.units[0, :2]).all() and p.units[0, 2:].tolist() == [2, 0]
    assert np.isnan(p.price[0, :2]).all() and p.price[1].tolist() == [2.0] * 4
    assert p.wday.tolist() == [1, 2, 3, 4]


# --- metrics ------------------------------------------------------------------------------------

def test_scale_is_the_one_step_naive_error_on_listed_days():
    hist = np.array([[nan, nan, 5, 1, 3], [1, nan, 4, 6, 6], [2, 2, 2, 2, 2], [nan, nan, nan, nan, 7]])
    # row 0: (1-5)^2, (3-1)^2 -> 10; row 1: pairs across the unlisted day are skipped -> (6-4)^2, 0 -> 2
    s = bt.scale(hist)
    assert s[:3].tolist() == [10, 2, 0]
    assert math.isnan(s[3])  # one listed day: no pair


def test_scores_skip_unlisted_window_days():
    f = np.array([[1, 1, 1, 1.0]])
    y = np.array([[0, nan, 3, 1.0]])  # errors (forecast - actual): 1, -, -2, 0
    s = bt.score(f, y, scale=np.array([0.5]), season=2)
    assert s["n_days"].tolist() == [3]
    assert s["mae"][0] == pytest.approx(1.0)
    assert s["rmse"][0] == pytest.approx(math.sqrt(5 / 3))
    assert s["bias"][0] == pytest.approx(-1 / 3)
    assert s["rmsse"][0] == pytest.approx(math.sqrt((5 / 3) / 0.5))
    # weekly (2-day) totals on listed days: week 1 f=1 y=0, week 2 f=2 y=4
    assert s["wk_forecast"].tolist() == [[1, 2]] and s["wk_actual"].tolist() == [[0, 4]]


def test_zero_scale_gives_no_scaled_error_but_keeps_the_plain_errors():
    s = bt.score(np.array([[1, 1.0]]), np.array([[2, 2.0]]), scale=np.array([0.0]), season=2)
    assert math.isnan(s["rmsse"][0])
    assert s["mae"][0] == 1 and s["rmse"][0] == 1 and s["bias"][0] == -1


def test_window_with_no_listed_day_scores_nothing():
    s = bt.score(np.array([[1, 1.0]]), np.array([[nan, nan]]), scale=np.array([1.0]), season=2)
    assert s["n_days"][0] == 0 and math.isnan(s["mae"][0]) and math.isnan(s["rmsse"][0])


# --- weights and classes at the origin ----------------------------------------------------------

def test_revenue_weight_uses_the_price_of_the_last_full_week(write_days):
    # Day 1 is a Saturday, so weeks are days 1-7, 8-14, 15-21. Origin = day 16 (a Sunday): the week of
    # days 15-21 is not over, so its price (9.0) is not known yet; the last known price is 2.0.
    price = [2.0] * 14 + [9.0] * 7
    p = panel_of(write_days, {A: [1] * 21}, prices_by_series={A: price})
    w = bt.revenue_weight(p.units[:, :16], p.price[:, :16], p.wday[:16], weight_days=7)
    assert w.tolist() == [7 * 2.0]


def test_last_known_price_skips_unlisted_days_and_needs_a_finished_week():
    price = np.array([[2, 2, 2, 2, 2, 2, 3, 5.0], [nan] * 7 + [4.0]])
    wday = np.array([1, 2, 3, 4, 5, 6, 7, 1])
    assert bt.last_known_price(price, wday)[0] == 3.0
    assert math.isnan(bt.last_known_price(price, wday)[1])


def test_demand_class_uses_only_days_up_to_the_origin():
    steady = [5.0] * 20
    later = [0] * 10 + [50] + [0] * 10 + [1]  # after day 20: rare, very uneven sales -> lumpy overall
    hist = np.array([steady + later, [nan] * 10 + [0] * 32, [0, 3] * 21])
    at20 = bt.class_at_origin(hist[:, :20], CLS)
    assert at20.tolist() == ["smooth", "too_few_sales", "intermittent"]
    assert bt.class_at_origin(hist, CLS)[0] == "lumpy"


# --- eligibility and leakage --------------------------------------------------------------------

def edge_case_table():
    rng = np.random.default_rng(0)
    return {
        A: [int(v) for v in rng.poisson(2.0, 35)],
        B: [None] * 14 + [int(v) for v in rng.poisson(1.0, 21)],   # launched day 15: 7 listed days at day 21
        C: [0] * 35,                                                # no sales at all
    }


def test_short_history_series_are_counted_not_dropped(write_days):
    p = panel_of(write_days, edge_case_table())
    inp = bt.origin_inputs(p, 21, CFG, CLS)
    assert inp["eligible"].tolist() == [True, False, True]
    assert inp["listed_days"].tolist() == [21, 7, 21]
    per_series, coverage, _ = bt.evaluate(p, CFG, CLS)
    c21 = next(c for c in coverage if c["origin"] == 21)
    assert c21["n_series"] == 3 and c21["n_scored"] == 2 and c21["n_short_history"] == 1
    assert set(per_series[per_series["origin"] == 21]["id"]) == {A, C}


def test_series_with_no_sales_forecasts_zero_and_has_zero_scale(write_days):
    p = panel_of(write_days, edge_case_table())
    inp = bt.origin_inputs(p, 21, CFG, CLS)
    assert inp["scale"][2] == 0 and inp["weight"][2] == 0
    for name, f in inp["forecasts"].items():
        assert (f[2] == 0).all(), name
    per_series, coverage, _ = bt.evaluate(p, CFG, CLS)
    c = per_series[(per_series["id"] == C) & (per_series["origin"] == 21)]
    assert c["rmsse"].isna().all() and (c["mae"] == 0).all()
    assert next(x for x in coverage if x["origin"] == 21)["n_zero_scale"] == 1


def test_series_unlisted_for_part_of_the_window_is_scored_on_listed_days(write_days):
    units = {A: [1] * 21 + [None] * 3 + [4] * 4, B: [2] * 28}
    p = panel_of(write_days, units)
    per_series, _, _ = bt.evaluate(p, {**CFG, "n_origins": 1}, CLS)
    a = per_series[(per_series["id"] == A) & (per_series["baseline"] == "naive")].iloc[0]
    assert a["n_days"] == 4 and a["mae"] == pytest.approx(3.0)


def test_changing_anything_after_the_origin_changes_nothing_at_the_origin(write_days, tmp_path):
    rng = np.random.default_rng(1)
    base = {A: [int(v) for v in rng.poisson(2.0, 42)],
            B: [None] * 5 + [int(v) for v in rng.poisson(0.3, 37)],
            C: [int(v) for v in rng.poisson(5.0, 42)]}
    prices = {A: [2.0] * 42, B: [1.5] * 42, C: [3.0] * 42}
    origin = 23
    p1 = panel_of(write_days, base, prices_by_series=prices)

    changed = {k: v[:origin] + [None if i % 3 == 0 else 99 for i in range(42 - origin)] for k, v in base.items()}
    changed_prices = {k: v[:origin] + [0.01] * (42 - origin) for k, v in prices.items()}
    path = tmp_path / "changed.parquet"
    from tests.conftest import day_table
    day_table(changed, changed_prices).to_parquet(path, index=False)
    p2 = bt.load_panel(path, 0, memory_limit="512MB", threads=1)
    assert not np.array_equal(np.nan_to_num(p1.units), np.nan_to_num(p2.units))

    i1, i2 = bt.origin_inputs(p1, origin, CFG, CLS), bt.origin_inputs(p2, origin, CFG, CLS)
    for key in ("scale", "weight", "listed_days", "eligible", "demand_class"):
        np.testing.assert_array_equal(i1[key], i2[key], err_msg=key)
    for name in CFG["baselines"]:
        np.testing.assert_array_equal(i1["forecasts"][name], i2["forecasts"][name], err_msg=name)


def test_evaluate_scores_only_windows_inside_the_training_days(write_days):
    p = panel_of(write_days, {A: [1, 2, 3] * 14, B: [0, 0, 4] * 14}, holdout_days=7)
    per_series, coverage, weekly = bt.evaluate(p, CFG, CLS)
    assert [c["origin"] for c in coverage] == [21, 28]
    assert max(c["last_scored_d"] for c in coverage) == 35 < 36
    assert len(per_series) == 2 * 2 * len(CFG["baselines"])
    assert set(weekly["week"]) == {1}


# --- aggregation and win counts -----------------------------------------------------------------

def test_group_metrics_weighted_scaled_error_averages_over_origins():
    df = pd.DataFrame({
        "origin": [1, 1, 2, 2], "rmsse": [1.0, 2.0, 0.5, nan], "weight": [3.0, 1.0, 1.0, 5.0],
        "mae": [1.0, 2.0, 3.0, 4.0], "rmse": [1.0, 2.0, 3.0, 4.0], "bias": [1.0, -1.0, 0.0, 2.0],
        "wk_abs_err": [1.0, 1.0, 1.0, 1.0], "wk_actual": [2.0, 2.0, 2.0, 2.0]})
    m = bt.group_metrics(df)
    assert m["n"] == 4 and m["n_scaled"] == 3
    assert m["wrmsse"] == pytest.approx((1.25 + 0.5) / 2)  # origin 1: (3*1 + 1*2)/4; origin 2: zero-scale row left out
    assert m["mean_rmsse"] == pytest.approx(3.5 / 3)
    assert m["mean_bias"] == pytest.approx(0.5)
    assert m["weekly_wape"] == pytest.approx(0.5)


def test_win_counts_against_the_reference_per_pair_and_per_series():
    df = pd.DataFrame({
        "id": ["a", "a", "b", "b"] * 2, "origin": [1, 2, 1, 2] * 2,
        "baseline": ["seasonal_naive"] * 4 + ["ma"] * 4,
        "rmse": [2.0, 2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.5]})
    w = bt.win_counts(df, "seasonal_naive", tol=1e-9)
    assert w["ma"]["pairs"] == {"n": 4, "wins": 1, "ties": 1, "losses": 2, "share_wins": 0.25}
    # per series (mean RMSE over origins): a 2.0 vs 2.0 tie; b 0.5 vs 0.75 loss
    assert w["ma"]["series"] == {"n": 2, "wins": 0, "ties": 1, "losses": 1, "share_wins": 0.0}
    assert "seasonal_naive" not in w
