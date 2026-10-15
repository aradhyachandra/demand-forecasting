"""F4 model features, rows and guards on hand-made day tables (tests/conftest.py): the holdout guard,
hand-computed feature values, leakage, training targets never after the backtest origin, unlisted and
closed target days, the calendar, the size estimate and a tiny end-to-end fit."""
import math

import numpy as np
import pytest

from forecast import backtest as bt
from forecast import config
from forecast import model as m

A = "FOODS_1_001_CA_1_evaluation"
B = "FOODS_1_002_CA_1_evaluation"
C = "FOODS_2_001_CA_2_evaluation"
nan = np.nan

BT = {"horizon_days": 7, "n_origins": 2, "min_history_days": 14, "season_length_days": 7,
      "moving_average_days": 7, "ses_alpha": 0.1, "croston_alpha": 0.1, "tsb_alpha_demand": 0.1,
      "tsb_alpha_probability": 0.1, "weight_days": 7, "tie_tolerance": 1e-9,
      "baselines": ["naive", "seasonal_naive", "moving_average", "ses", "croston", "sba", "tsb"],
      "reference_baseline": "seasonal_naive"}
CLS = {"adi_cutoff": 1.32, "cv2_cutoff": 0.49, "cv2_ddof": 1, "min_nonzero_days": 2}
MCFG = {
    "training_origins": {"step_days": 7, "n_origins": 2, "latest_gap_days": 7},
    "max_feature_table_gb": 2.0,
    "features": {"mean_days": [7, 14], "weekday_mean_weeks": [1, 4], "zero_share_days": 14,
                 "price_ref_days": 14},
    "calendar": {"event_distance_cap_days": 5},
    "closed_events": ["Christmas"],
    "objective": "tweedie", "tweedie_variance_power": 1.1,
    "fixed_params": {"seed": 1, "deterministic": True, "force_col_wise": True, "num_threads": 1,
                     "verbosity": -1},
}

# Series A: day 1 is a Saturday. Origin 23 (a Sunday). Units on days 1..23, then 7 target days.
A_UNITS = [None, None, 0, 1, 2, 0, 3,   1, 0, 0, 4, 0, 2, 1,   0, 5, 0, 0, 2, 0, 0,   6, 0]
A_PRICE = [2.0] * 14 + [1.0] * 7 + [1.5] * 9   # week 3 (d15-21) at 1.0; the unfinished week 4 at 1.5


NAMES = m.feature_names(MCFG)


def data_of(write_days, units, holdout_days=0, allow_holdout=False, **kwargs):
    return m.load_data(write_days(units, **kwargs), holdout_days, MCFG, allow_holdout=allow_holdout,
                       memory_limit="512MB", threads=1)


def fixture_a(write_days, tail=None, **kwargs):
    tail = tail if tail is not None else [1, 2, 3, 4, 5, 6, 7]
    return data_of(write_days, {A: A_UNITS + tail, B: [0] * 30}, prices_by_series={A: A_PRICE}, **kwargs)


def col(X, name):
    return X[:, NAMES.index(name)]


# --- the holdout guard --------------------------------------------------------------------------

def test_guard_refuses_holdout_days_without_the_flag():
    m.check_days(1913, first_holdout=1914)
    with pytest.raises(m.HoldoutError):
        m.check_days(1914, first_holdout=1914)
    m.check_days(1941, first_holdout=1914, allow_holdout=True)


def test_data_without_flag_stops_before_the_holdout(write_days):
    d = data_of(write_days, {A: [1] * 30 + [1000] * 7}, holdout_days=7)
    assert d.first_holdout == 31 and d.panel.last_train == 30
    assert np.nanmax(d.panel.units) < 1000


def test_data_with_flag_keeps_the_boundary(write_days):
    d = data_of(write_days, {A: [1] * 30 + [1000] * 7}, holdout_days=7, allow_holdout=True)
    assert d.first_holdout == 31 and d.panel.last_train == 37


def test_rows_reaching_into_the_holdout_are_refused(write_days):
    d = data_of(write_days, {A: [1] * 30 + [5] * 7}, holdout_days=7, allow_holdout=True)
    with pytest.raises(m.HoldoutError):
        m.rows_at(d, 30, 7, MCFG, BT, with_target=False)       # targets d31..d37
    with pytest.raises(m.HoldoutError):
        m.training_table(d, 37, MCFG, BT)                       # training targets up to d37
    X, _, _ = m.rows_at(d, 30, 7, MCFG, BT, with_target=False, allow_holdout=True)
    assert len(X) == 7
    m.rows_at(d, 23, 7, MCFG, BT, with_target=True)              # targets d24..d30: fine


# --- hand-computed features ---------------------------------------------------------------------

def test_series_features_by_hand(write_days):
    d = fixture_a(write_days)
    f = m.origin_features(d, 23, MCFG)
    a, b = 0, 1
    assert f["mean_7"][a] == pytest.approx(8 / 7)          # d17..23: 0,0,2,0,0,6,0
    assert f["mean_14"][a] == pytest.approx(20 / 14)        # d10..23
    assert f["zero_share"][a] == pytest.approx(8 / 14)
    assert f["days_since_sale"][a] == 1                     # last sale d22
    assert f["listed_days"][a] == 21
    nz = np.array([1, 2, 3, 1, 4, 2, 1, 5, 2, 6], dtype=float)
    assert f["adi"][a] == pytest.approx(21 / 10)
    assert f["cv2"][a] == pytest.approx(nz.var(ddof=1) / nz.mean() ** 2)
    # never sold: means 0, no "days since", ADI and CV² undefined
    assert f["mean_7"][b] == 0 and math.isnan(f["days_since_sale"][b])
    assert math.isnan(f["adi"][b]) and math.isnan(f["cv2"][b])


def test_last_known_price_ignores_the_unfinished_week(write_days):
    d = fixture_a(write_days)
    f = m.origin_features(d, 23, MCFG)
    assert f["price_last"][0] == pytest.approx(1.0)        # week d15..21, not the 1.5 of d22..23
    assert f["price_ratio"][0] == pytest.approx(1.0 / 1.5)  # mean listed price d8..21 = 1.5


def test_weekday_means_skip_unlisted_days(write_days):
    d = fixture_a(write_days)
    f = m.origin_features(d, 23, MCFG)
    wd1, wd4 = f["wday_mean_1"][0], f["wday_mean_4"][0]   # indexed by M5 wday - 1 (0 = Saturday)
    assert wd1[0] == 6 and wd1[1] == 0                       # last Saturday d22, last Sunday d23
    assert wd4[0] == pytest.approx(7 / 3)                    # d1 unlisted, d8 1, d15 0, d22 6
    assert wd4[1] == pytest.approx(5 / 3)                    # d2 unlisted, d9 0, d16 5, d23 0
    assert wd4[2] == 0                                       # Mondays d3, d10, d17


def test_rows_line_up_target_day_features(write_days):
    d = fixture_a(write_days)
    X, y, meta = m.rows_at(d, 23, 7, MCFG, BT, with_target=True)
    a = meta["series"] == 0
    assert meta["d_num"][a].tolist() == list(range(24, 31))
    assert col(X, "h")[a].tolist() == list(range(1, 8))
    assert col(X, "wday")[a].tolist() == [3, 4, 5, 6, 7, 1, 2]
    assert col(X, "wday_mean_1")[a].tolist() == [0, 0, 2, 0, 0, 6, 0]   # d17..d23, Monday..Sunday
    assert y[a].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert np.all(col(X, "mean_7")[a] == np.float32(8 / 7))
    assert X.dtype == np.float32


# --- leakage ------------------------------------------------------------------------------------

def test_changing_anything_after_the_origin_leaves_features_unchanged(write_days, tmp_path):
    base = fixture_a(write_days)
    X0, _, _ = m.rows_at(base, 23, 7, MCFG, BT, with_target=False)
    f0 = m.origin_features(base, 23, MCFG)
    changed = base
    changed.panel.units[:, 23:] = 99.0                   # units after d23
    changed.panel.price[:, 23:] = 0.01                   # prices after d23
    changed.panel.units[0, 25:27] = nan                  # listing after d23
    changed.panel.price[0, 25:27] = nan
    X1, _, _ = m.rows_at(changed, 23, 7, MCFG, BT, with_target=False)
    f1 = m.origin_features(changed, 23, MCFG)
    np.testing.assert_array_equal(X0, X1)
    for k in f0:
        np.testing.assert_array_equal(f0[k], f1[k])


def test_changing_the_prices_of_the_origin_week_leaves_features_unchanged(write_days):
    d0 = fixture_a(write_days)
    X0, _, _ = m.rows_at(d0, 23, 7, MCFG, BT, with_target=False)
    d0.panel.price[:, 21:23] = 9.0   # d22, d23: inside the unfinished week of the origin
    X1, _, _ = m.rows_at(d0, 23, 7, MCFG, BT, with_target=False)
    np.testing.assert_array_equal(X0, X1)


# --- training rows ------------------------------------------------------------------------------

def test_training_origins_end_one_horizon_before_the_backtest_origin():
    assert m.training_origins(100, MCFG["training_origins"]) == [86, 93]
    cfg = config.load("model")
    to = m.training_origins(1801, cfg["training_origins"])
    assert max(to) == 1801 - 28 and len(to) == cfg["training_origins"]["n_origins"]
    assert {(1801 - t) % 7 for t in to} == {0}           # same weekday as the backtest origin


def test_training_targets_never_after_the_backtest_origin(write_days):
    d = data_of(write_days, {A: [1, 0, 2] * 14, B: [0, 3] * 21})
    X, y, meta = m.training_table(d, 35, MCFG, BT)
    assert meta["d_num"].max() == 35
    assert set(meta["origin"]) == {21, 28}
    assert np.all(meta["d_num"] - meta["origin"] == col(X, "h"))


def test_unlisted_and_closed_target_days_are_left_out(write_days):
    units = {A: [1] * 23 + [2, None, None, 2, 2, 2, 2], B: [1] * 30}
    d = data_of(write_days, units, events={27: "Christmas"})
    X, y, meta = m.rows_at(d, 23, 7, MCFG, BT, with_target=True)
    assert sorted(meta["d_num"][meta["series"] == 0].tolist()) == [24, 28, 29, 30]
    assert 27 not in meta["d_num"][meta["series"] == 1].tolist()
    assert not np.isnan(y).any()


def test_short_history_series_get_no_training_rows(write_days):
    d = data_of(write_days, {A: [1] * 30, B: [None] * 15 + [1] * 15})   # B listed 8 days at d23
    _, _, meta = m.rows_at(d, 23, 7, MCFG, BT, with_target=True)
    assert set(meta["series"]) == {0}


def test_closed_history_days_are_ignored_in_features(write_days):
    d = data_of(write_days, {A: [1] * 19 + [100] + [1] * 10}, events={20: "Christmas"})
    f = m.origin_features(d, 23, MCFG)
    assert f["mean_7"][0] == 1.0
    assert f["listed_days"][0] == 22


# --- calendar -----------------------------------------------------------------------------------

def test_calendar_features(write_days):
    d = data_of(write_days, {A: [1] * 30}, events={10: "SuperBowl", 20: "Christmas"})
    cal = d.cal
    assert cal.loc[1, "wday"] == 1 and cal.loc[1, "month"] == 1 and cal.loc[1, "mday"] == 29
    assert cal.loc[4, "snap"] == 1 and cal.loc[1, "snap"] == 0      # 2011-02-01 vs 2011-01-29
    assert cal.loc[10, "event_name"] > 0 and cal.loc[11, "event_name"] == 0
    assert cal.loc[8, "days_to_event"] == 2 and cal.loc[12, "days_to_event"] == -2
    assert cal.loc[15, "days_to_event"] == 5                        # 5 ahead of d20 beats 5 behind d10
    assert cal.loc[1, "days_to_event"] == 5                         # capped (9 days to d10)
    assert bool(cal.loc[20, "closed"]) and not bool(cal.loc[10, "closed"])


def test_calendar_from_csv_matches_the_parquet_layout(tmp_path):
    csv = tmp_path / "calendar.csv"
    csv.write_text("date,wm_yr_wk,weekday,wday,month,year,d,event_name_1,event_type_1,event_name_2,"
                   "event_type_2,snap_CA,snap_TX,snap_WI\n"
                   "2011-01-29,11101,Saturday,1,1,2011,d_1,,,,,0,1,1\n"
                   "2011-01-30,11101,Sunday,2,1,2011,d_2,SuperBowl,Sporting,,,1,0,0\n")
    cal = m.calendar_from_csv(csv, "CA")
    assert cal["d_num"].tolist() == [1, 2] and cal["snap"].tolist() == [0, 1]
    assert cal["event_name_1"].tolist()[1] == "SuperBowl"


# --- size estimate and forecasts ----------------------------------------------------------------

def test_size_estimate_and_cap():
    gb = m.estimate_table_gb(n_series=5748, horizon=28, n_origins=26, n_features=24)
    assert gb == pytest.approx(5748 * 28 * 26 * 25 * 4 / 1e9)
    with pytest.raises(MemoryError):
        m.check_table_size(10.0, 2.0)
    m.check_table_size(1.0, 2.0)


def test_fit_and_forecast_end_to_end(write_days):
    rng = np.random.default_rng(0)
    units = {A: rng.poisson(3, 70).tolist(), B: rng.poisson(1, 70).tolist(),
             C: [None] * 10 + rng.poisson(5, 60).tolist()}
    d = data_of(write_days, units, events={60: "Christmas"})
    X, y, _ = m.training_table(d, 56, MCFG, BT)
    params = m.lgb_params(MCFG, {"num_leaves": 4, "learning_rate": 0.1, "min_data_in_leaf": 5})
    booster = m.fit(X, y, NAMES, params, rounds=20)
    f = m.forecast(d, booster, 56, MCFG, BT, rounds=[10, 20])
    assert set(f) == {10, 20}
    assert f[20].shape == (3, 7) and np.all(f[20] >= 0)
    assert np.all(f[20][:, 3] == 0)                    # d60 is Christmas: forecast 0
    imp = m.importance(booster)
    assert set(imp) == set(NAMES)


def test_extra_forecasts_are_scored_on_the_f3_series(write_days):
    d = data_of(write_days, {A: [1, 2] * 21, B: [None] * 30 + [1] * 12, C: [0, 3] * 21})
    seen = []

    def extra(origin, inp):
        seen.append(origin)
        return {"const": np.full((3, 7), 1.0)}

    per_series, _, _ = bt.evaluate(d.panel, BT, CLS, extra=extra)
    assert seen == [28, 35]
    a = per_series[per_series["baseline"] == "naive"][["id", "origin"]]
    b = per_series[per_series["baseline"] == "const"][["id", "origin"]]
    assert a.reset_index(drop=True).equals(b.reset_index(drop=True))
