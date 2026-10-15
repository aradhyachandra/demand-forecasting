"""F3 baselines on tiny hand-computed series. NaN = the item was not listed that day."""
import numpy as np
import pytest

from forecast import baselines as bl

nan = np.nan


def test_naive_repeats_the_last_listed_value():
    assert bl.naive(np.array([3, 5, 2.0]), 4).tolist() == [2, 2, 2, 2]
    assert bl.naive(np.array([3, 5, nan]), 2).tolist() == [5, 5]


def test_seasonal_naive_repeats_the_same_weekday_last_week():
    hist = np.arange(1, 15, dtype=float)  # days 1..14, origin = day 14
    assert bl.seasonal_naive(hist, 10, season=7).tolist() == [8, 9, 10, 11, 12, 13, 14, 8, 9, 10]


def test_seasonal_naive_goes_back_another_week_if_that_day_was_unlisted():
    hist = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, nan, 11, 12, 13, 14.0])
    assert bl.seasonal_naive(hist, 7, season=7).tolist() == [8, 9, 3, 11, 12, 13, 14]


def test_moving_average_uses_the_last_listed_days():
    hist = np.array([1, 2, 3, nan, 6.0])
    assert bl.moving_average(hist, 2, window=3).tolist() == pytest.approx([11 / 3, 11 / 3])


def test_ses_starts_at_the_first_listed_value():
    # level 2 -> 2 + 0.5*(4-2) = 3 -> 3 + 0.5*(0-3) = 1.5
    assert bl.ses(np.array([nan, 2, 4, 0.0]), 3, alpha=0.5).tolist() == [1.5, 1.5, 1.5]


def test_croston_and_sba_hand_computed():
    hist = np.array([0, 0, 3, 0, 1.0])
    # first demand (3) after 3 periods: z=3, p=3; next demand (1) after 2: z=2, p=2.5 -> 0.8
    assert bl.croston(hist, 2, alpha=0.5).tolist() == pytest.approx([0.8, 0.8])
    assert bl.sba(hist, 2, alpha=0.5).tolist() == pytest.approx([0.6, 0.6])  # x (1 - 0.5/2)


def test_croston_with_no_demand_yet_forecasts_zero():
    assert bl.croston(np.zeros(5), 3, alpha=0.1).tolist() == [0, 0, 0]
    assert bl.sba(np.zeros(5), 3, alpha=0.1).tolist() == [0, 0, 0]
    assert bl.tsb(np.zeros(5), 3, alpha_demand=0.1, alpha_probability=0.1).tolist() == [0, 0, 0]


def test_croston_with_a_single_demand_uses_its_size_and_interval():
    hist = np.array([0, 4, 0, 0.0])  # z=4, p=2 (two periods up to and including the demand)
    assert bl.croston(hist, 1, alpha=0.5).tolist() == [2.0]
    assert bl.sba(hist, 1, alpha=0.5).tolist() == [1.5]


def test_croston_counts_only_listed_days_in_the_interval():
    hist = np.array([nan, nan, 0, 3.0])  # two listed periods to the first demand
    assert bl.croston(hist, 1, alpha=0.5).tolist() == [1.5]


def test_tsb_hand_computed():
    hist = np.array([0, 0, 3, 0, 1.0])
    # first demand: z=3, prob=1/3; then y=0: prob=1/6; then y=1: prob=7/12, z=2 -> 7/6
    assert bl.tsb(hist, 1, alpha_demand=0.5, alpha_probability=0.5).tolist() == pytest.approx([7 / 6])


def test_baselines_work_row_by_row_on_a_matrix():
    hist = np.array([[0, 0, 3, 0, 1.0], [2, 2, 2, 2, 2.0], [nan, nan, nan, 1, 0]])
    out = bl.croston(hist, 2, alpha=0.5)
    assert out.shape == (3, 2)
    assert out[:, 0] == pytest.approx([0.8, 2.0, 1.0])
    for i in range(3):
        assert bl.croston(hist[i], 2, alpha=0.5) == pytest.approx(out[i])


def test_series_never_listed_gets_no_forecast():
    hist = np.array([nan, nan, nan])
    for f in (bl.naive(hist, 2), bl.moving_average(hist, 2, window=3), bl.ses(hist, 2, alpha=0.1),
              bl.croston(hist, 2, alpha=0.1), bl.seasonal_naive(hist, 2, season=7)):
        assert np.isnan(f).all()


def test_forecast_all_runs_every_configured_baseline():
    cfg = {"baselines": ["naive", "seasonal_naive", "moving_average", "ses", "croston", "sba", "tsb"],
           "season_length_days": 7, "moving_average_days": 3, "ses_alpha": 0.5, "croston_alpha": 0.5,
           "tsb_alpha_demand": 0.5, "tsb_alpha_probability": 0.5}
    out = bl.forecast_all(np.array([[0, 0, 3, 0, 1, 2, 0, 1.0]]), 4, cfg)
    assert list(out) == cfg["baselines"]
    assert all(f.shape == (1, 4) for f in out.values())
