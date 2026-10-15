"""F5 inventory policy and simulation on tiny hand-made series, every expected value worked out by hand.

Day indices are 0-based in code (day 0 = the first forecast day, d_origin + 1)."""
import numpy as np
import pandas as pd
import pytest

from forecast import inventory as inv


def one(x):
    return np.asarray([x], dtype=float)


# --- the end-of-horizon rule ---------------------------------------------------------------------

def test_schedule_keeps_only_reviews_whose_protection_period_fits():
    # H 10, lead 2, review 3: protection 5 days. Review 0 covers 0-4, review 3 covers 3-7, review 6
    # would cover 6-10 (day 10 is past the forecast) -> dropped. Measured: day 2 (first arrival) to 7.
    s = inv.schedule(horizon=10, lead=2, review=3)
    assert s.reviews == [0, 3]
    assert (s.first, s.last) == (2, 7)
    assert s.protection == 5


def test_schedule_on_the_real_central_settings():
    # 28 days, lead 2, review 7 (config/inventory.yaml): reviews on days 1, 8, 15; measured days 3-23
    s = inv.schedule(28, 2, 7)
    assert [r + 1 for r in s.reviews] == [1, 8, 15]
    assert (s.first + 1, s.last + 1) == (3, 23)


def test_schedule_last_review_ends_exactly_on_the_horizon():
    s = inv.schedule(horizon=10, lead=1, review=3)   # P = 4: reviews 0, 3, 6 (6 + 3 = 9 = last day)
    assert s.reviews == [0, 3, 6] and s.last == 9


def test_schedule_refuses_a_protection_period_longer_than_the_forecast():
    with pytest.raises(ValueError):
        inv.schedule(horizon=10, lead=6, review=5)


def test_no_order_is_placed_on_a_review_that_does_not_fit():
    fc = np.ones((1, 10))
    sim = inv.simulate(fc, fc.copy(), np.zeros(1), lead=2, review=3)
    assert np.flatnonzero(sim.placed[0] > 0).tolist() == [3]   # day 0 orders 0 (start = S), day 6 not reviewed
    assert sim.placed[0, 6] == 0


# --- order-up-to and arrival timing --------------------------------------------------------------

def test_order_up_to_and_arrival_after_the_lead_time():
    # forecast = demand = 1 a day, no safety stock, lead 2, review 3 -> S = 5.
    # Start: on hand 5. Day 0: IP 5, order 0. End of days 0,1,2: 4,3,2.
    # Day 3 review: IP 2 -> order 3, arrives at the start of day 5. End of 3,4: 1,0. Day 5: 0+3-1 = 2.
    fc = np.ones((1, 10))
    sim = inv.simulate(fc, fc.copy(), np.zeros(1), lead=2, review=3)
    assert sim.start[0] == 5
    assert sim.placed[0, 3] == 3 and sim.arrived[0, 5] == 3 and sim.arrived[0].sum() == 3
    assert sim.on_hand[0].tolist() == [4, 3, 2, 1, 0, 2, 1, 0, 0, 0]
    assert sim.lost[0].tolist() == [0] * 8 + [1, 1]   # days 8-9 are after the last covered day


def test_zero_lead_time_order_arrives_the_same_day():
    fc = np.ones((1, 6))
    dem = np.array([[2, 2, 2, 2, 2, 2]], dtype=float)
    sim = inv.simulate(fc, dem, np.zeros(1), lead=0, review=2)   # S = 2
    # day 0: start 2, order 0, serve 2 -> 0. Day 2: order 2, arrives now, serve 2.
    assert sim.arrived[0, 2] == 2 and sim.on_hand[0, 2] == 0
    assert sim.lost[0].tolist() == [0, 2, 0, 2, 0, 2]   # reviews 0, 2, 4 cover only the review day


def test_order_counts_stock_on_order_in_the_inventory_position():
    # lead 3, review 1: an order placed on day 1 is still on order on day 2, so day 2 does not order it again
    fc = np.ones((1, 8))
    dem = np.zeros((1, 8))
    dem[0, 0] = 1   # one unit sold on day 0 only
    sim = inv.simulate(fc, dem, np.zeros(1), lead=3, review=1)   # S = 4, start 4
    assert sim.placed[0].tolist()[:5] == [0, 1, 0, 0, 0]


def test_order_is_rounded_up_to_whole_units_and_never_negative():
    fc = np.full((1, 6), 0.4)
    sim = inv.simulate(fc, np.zeros((1, 6)), np.zeros(1), lead=1, review=2, start_factor=0.0)
    assert sim.start[0] == 0
    assert sim.placed[0, 0] == 2   # S = 3 x 0.4 = 1.2 -> 2 units
    # a big negative safety stock: S floors at 0, no order, nothing negative
    sim = inv.simulate(fc, np.ones((1, 6)), np.array([-10.0]), lead=1, review=2)
    assert sim.start[0] == 0 and sim.placed.sum() == 0 and (sim.on_hand >= 0).all()


def test_starting_inventory_factor():
    fc = np.ones((1, 10))
    sim = inv.simulate(fc, fc.copy(), np.array([0.5]), lead=2, review=3, start_factor=1.5)
    assert sim.start[0] == np.ceil(1.5 * 5.5)


def test_series_are_simulated_independently():
    fc = np.vstack([np.ones(10), 2 * np.ones(10)])
    both = inv.simulate(fc, fc.copy(), np.zeros(2), 2, 3)
    second = inv.simulate(fc[1:], fc[1:].copy(), np.zeros(1), 2, 3)
    assert np.array_equal(both.on_hand[1], second.on_hand[0])


# --- lost sales and backorders -------------------------------------------------------------------

def test_lost_sales_are_lost_not_carried():
    fc = np.ones((1, 7))
    dem = np.array([[1, 10, 1, 1, 1, 1, 1]], dtype=float)
    sim = inv.simulate(fc, dem, np.zeros(1), lead=1, review=3)   # S = 4, start 4, reviews 0 and 3
    # day 0: 4 -> 3; day 1: demand 10, serve 3, lose 7 -> 0; day 2: lose 1; day 3: order 4 (IP 0),
    # lose 1; day 4: the 4 arrive, serve 1 -> 3
    assert sim.served[0].tolist() == [1, 3, 0, 0, 1, 1, 1]
    assert sim.lost[0].tolist() == [0, 7, 1, 1, 0, 0, 0]
    assert sim.placed[0, 3] == 4 and sim.on_hand[0, 4] == 3


def test_backorders_are_served_from_the_next_delivery():
    fc = np.ones((1, 7))
    dem = np.array([[1, 10, 1, 1, 1, 1, 1]], dtype=float)
    sim = inv.simulate(fc, dem, np.zeros(1), lead=1, review=3, unmet="backorders")
    # net stock at the end of days 0-2: 3, -7, -8. Day 3 (before its demand) orders S - IP = 4 - (-8)
    # = 12, then net -9. Day 4: -9 + 12 = 3 (backlog cleared first), serve 1 -> 2
    assert sim.placed[0, 3] == 12
    assert sim.lost[0].tolist() == [0, 7, 1, 1, 0, 0, 0]   # short on the day it was demanded
    assert sim.on_hand[0].tolist() == [3, 0, 0, 0, 2, 1, 0]
    assert sim.served[0].tolist() == [1, 3, 0, 0, 1, 1, 1]


def test_unknown_unmet_demand_rule_is_refused():
    with pytest.raises(ValueError):
        inv.simulate(np.ones((1, 4)), np.ones((1, 4)), np.zeros(1), 1, 1, unmet="maybe")


# --- measures and costs --------------------------------------------------------------------------

def test_fill_rate_stockout_days_and_costs_by_hand():
    fc = np.ones((1, 6))
    dem = np.array([[1, 10, 1, 1, 1, 1]], dtype=float)
    sim = inv.simulate(fc, dem, np.zeros(1), lead=1, review=3)
    # schedule(6, 1, 3): P 4, reviews [0] only (review 3 would cover 3-6) -> measured days 1..3
    assert (sim.schedule.first, sim.schedule.last) == (1, 3)
    price = one(3.65)
    m = inv.series_measures(sim, price, holding_rate_per_year=0.1, stockout_share=0.5)
    r = m.iloc[0]
    assert r["demand"] == 12 and r["served"] == 3 and r["lost"] == 9          # days 1-3: 10+1+1, 3+0+0
    assert r["stockout_days"] == 3 and r["days"] == 3
    assert r["avg_inv_units"] == 0                                             # on hand 0, 0, 0 at the end of days 1-3
    assert r["lost_cost"] == pytest.approx(9 * 3.65 * 0.5)
    assert r["holding_cost"] == 0
    assert r["cycles"] == 1 and r["cycles_short"] == 1                        # the one cycle (days 1-3) ran short


def test_holding_cost_per_unit_day():
    fc = np.ones((1, 10))
    sim = inv.simulate(fc, fc.copy(), np.zeros(1), lead=2, review=3)
    # measured days 2..7: on hand 2,1,0,2,1,0 -> 6 unit-days, average 1 unit
    m = inv.series_measures(sim, one(3.65), holding_rate_per_year=0.1, stockout_share=0.5).iloc[0]
    assert m["avg_inv_units"] == pytest.approx(1.0)
    assert m["avg_inv_money"] == pytest.approx(3.65)
    assert m["holding_cost"] == pytest.approx(6 * 3.65 * 0.1 / 365)
    assert m["lost"] == 0 and m["total_cost"] == pytest.approx(m["holding_cost"])
    assert m["cycles"] == 2 and m["cycles_short"] == 0


def test_summary_adds_series_up():
    df = pd.DataFrame({"demand": [10, 30], "served": [9, 30], "lost": [1, 0], "days": [5, 5],
                       "stockout_days": [1, 0], "avg_inv_units": [2.0, 4.0], "avg_inv_money": [4.0, 8.0],
                       "holding_cost": [0.1, 0.2], "lost_cost": [1.0, 0.0], "total_cost": [1.1, 0.2],
                       "cycles": [2, 2], "cycles_short": [1, 0]})
    s = inv.summarize(df)
    assert s["fill_rate"] == pytest.approx(39 / 40)
    assert s["stockout_days"] == 1 and s["stockout_day_share"] == pytest.approx(0.1)
    assert s["avg_inv_money"] == pytest.approx(12.0) and s["total_cost"] == pytest.approx(1.3)
    assert s["cycle_service_level"] == pytest.approx(0.75)
    assert s["n_series"] == 2


# --- safety stock from summed-window backtest errors ---------------------------------------------

def test_window_errors_by_hand():
    # one series, one origin, residuals 1,0,2,1, window 2 -> sums 1, 2, 3: mean 2, sd 1
    mean, sd, n = inv.window_errors(np.array([[[1, 0, 2, 1]]], dtype=float), 2)
    assert mean[0] == pytest.approx(2) and sd[0] == pytest.approx(1) and n[0] == 3


def test_zero_errors_give_zero_safety_stock():
    mean, sd, _ = inv.window_errors(np.zeros((1, 3, 28)), 9)
    for method in ("bias_adjusted", "z_sd_only"):
        assert inv.safety_stock(mean, sd, inv.z_value(0.99), method)[0] == 0


def test_bias_only_moves_the_main_method():
    # the forecast is 0.5 a day too low on every day of every origin: summed error 0.5 x 9, no spread
    mean, sd, _ = inv.window_errors(np.full((1, 4, 28), 0.5), 9)
    assert mean[0] == pytest.approx(4.5) and sd[0] == pytest.approx(0)
    assert inv.safety_stock(mean, sd, 1.645, "bias_adjusted")[0] == pytest.approx(4.5)
    assert inv.safety_stock(mean, sd, 1.645, "z_sd_only")[0] == 0
    # over-forecasting lowers the main safety stock
    mean, sd, _ = inv.window_errors(np.full((1, 4, 28), -0.5), 9)
    assert inv.safety_stock(mean, sd, 1.645, "bias_adjusted")[0] == pytest.approx(-4.5)


def test_correlated_errors_are_wider_than_the_square_root_rule():
    # each origin is off by a constant level (+1 or -1 a day): perfectly correlated within a window
    res = np.stack([np.full(28, 1.0), np.full(28, -1.0), np.full(28, 1.0), np.full(28, -1.0)])[None]
    _, sd, _ = inv.window_errors(res, 9)
    daily_sd = res.reshape(-1).std(ddof=1)
    assert sd[0] > daily_sd * np.sqrt(9) * 2.5   # sums are +-9, sd about 9; sqrt rule gives about 3


def test_missing_days_drop_only_the_windows_that_touch_them():
    res = np.array([[[1, np.nan, 2, 1, 1]]])
    mean, _, n = inv.window_errors(res, 2)   # windows (1,nan) and (nan,2) dropped -> 3, 2
    assert n[0] == 2 and mean[0] == pytest.approx(2.5)


def test_series_without_backtest_windows_get_nan():
    mean, sd, n = inv.window_errors(np.full((1, 2, 28), np.nan), 9)
    assert n[0] == 0 and np.isnan(mean[0]) and np.isnan(sd[0])


def test_z_value():
    assert inv.z_value(0.5) == pytest.approx(0)
    assert inv.z_value(0.95) == pytest.approx(1.6449, abs=1e-4)


def test_unknown_safety_stock_method_is_refused():
    with pytest.raises(ValueError):
        inv.safety_stock(np.zeros(1), np.zeros(1), 1.0, "guess")


# --- leakage: holdout errors never enter the safety stock ----------------------------------------

def backtest_frame(origins, ids=("a", "b"), H=28, value=None):
    rows = []
    for o in origins:
        for k, sid in enumerate(ids):
            for h in range(1, H + 1):
                r = value if value is not None else float((h * (k + 1) + o) % 5 - 2)
                rows.append({"id": sid, "origin": o, "h": h, "method": "tsb", "residual": r})
    return pd.DataFrame(rows)


def test_spread_ignores_rows_from_origins_not_allowed():
    ids = np.array(["a", "b"])
    clean = backtest_frame([1801, 1829, 1857, 1885])
    poisoned = pd.concat([clean, backtest_frame([1913], value=1e6)])   # holdout errors, huge
    a = inv.spread_from_backtest(clean, "tsb", ids, 9, [1801, 1829, 1857, 1885], forecast_origin=1913, horizon=28)
    b = inv.spread_from_backtest(poisoned, "tsb", ids, 9, [1801, 1829, 1857, 1885], forecast_origin=1913, horizon=28)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_spread_refuses_an_origin_whose_window_reaches_the_forecast_days():
    ids = np.array(["a", "b"])
    df = backtest_frame([1801, 1829, 1857, 1885, 1913])
    with pytest.raises(inv.LeakageError):
        inv.spread_from_backtest(df, "tsb", ids, 9, [1885, 1913], forecast_origin=1913, horizon=28)
    with pytest.raises(inv.LeakageError):   # robustness run at d_1885 may not use origin 1885 itself
        inv.spread_from_backtest(df, "tsb", ids, 9, [1857, 1885], forecast_origin=1885, horizon=28)


def test_spread_aligns_to_the_requested_ids_and_marks_missing_series():
    ids = np.array(["b", "zz", "a"])
    df = backtest_frame([1801], value=1.0)
    mean, sd, n = inv.spread_from_backtest(df, "tsb", ids, 9, [1801], forecast_origin=1913, horizon=28)
    assert mean[0] == pytest.approx(9) and mean[2] == pytest.approx(9)
    assert n[1] == 0 and np.isnan(mean[1])


# --- reading results -----------------------------------------------------------------------------

def test_inventory_at_fill_rate_interpolates_on_the_curve():
    fill = [0.80, 0.90, 0.96]
    money = [100.0, 150.0, 300.0]
    assert inv.inventory_at_fill_rate(fill, money, 0.93) == pytest.approx(225.0)
    assert inv.inventory_at_fill_rate(fill, money, 0.90) == pytest.approx(150.0)
    assert inv.inventory_at_fill_rate(fill, money, 0.99) is None   # curve does not reach it
    assert inv.inventory_at_fill_rate(fill, money, 0.70) is None


def test_top_share():
    v = np.array([50.0] + [5.0] * 9 + [0.0] * 10)   # 20 series, top 10% = 2 series: 50 + 5 of 95
    assert inv.top_share(v, 0.10) == pytest.approx(55 / 95)
    assert inv.top_share(np.zeros(4), 0.5) is None
