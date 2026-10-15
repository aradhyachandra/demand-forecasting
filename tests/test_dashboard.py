"""Dashboard data functions (src/forecast/dashboard.py) on hand-made files, the committed results, and
the app itself (streamlit AppTest) without data/processed."""
import numpy as np
import pandas as pd
import pytest

from forecast import dashboard as dash
from forecast import inventory as inv
from forecast.config import ROOT

APP = ROOT / "app" / "streamlit_app.py"


def forecast_rows(path, ids=("FOODS_1_001_CA_1_evaluation",), origins=(1913,), H=28, methods=("tsb", "lgbm"),
                  residual=None, fc=None):
    rows = []
    for o in origins:
        for sid in ids:
            for m in methods:
                for h in range(1, H + 1):
                    f = fc(m, h) if fc else 1.0
                    a = float(h % 3)
                    r = residual(o, h) if residual else a - f
                    rows.append({"id": sid, "dept_id": "FOODS_1", "demand_class": "smooth", "origin": o, "h": h,
                                 "d_num": o + h, "method": m, "forecast": f, "actual": a, "residual": r})
    pd.DataFrame(rows).to_parquet(path, index=False)


# --- missing data ---------------------------------------------------------------------------------

def test_missing_processed_files_are_listed(tmp_path):
    assert dash.missing_processed(tmp_path) == list(dash.PROCESSED_FILES)
    for f in dash.PROCESSED_FILES:
        (tmp_path / f).write_bytes(b"")
    assert dash.missing_processed(tmp_path) == []


# --- product view ---------------------------------------------------------------------------------

def test_band_from_residuals_by_hand():
    fc = np.array([1.0, 2.0, 0.2])
    res = np.array([-1.0, 0.0, 0.0, 0.0, 3.0])   # 0% quantile -1, 100% quantile 3
    lo, hi = dash.band_from_residuals(fc, res, 0.0, 1.0)
    assert lo.tolist() == [0.0, 1.0, 0.0]        # clipped at zero
    assert hi.tolist() == [4.0, 5.0, 3.2]


def test_band_from_residuals_ignores_missing_and_handles_none():
    lo, hi = dash.band_from_residuals(np.array([1.0]), np.array([np.nan, 1.0, 1.0]), 0.1, 0.9)
    assert lo.tolist() == [2.0] and hi.tolist() == [2.0]
    lo, hi = dash.band_from_residuals(np.array([1.0]), np.array([np.nan]), 0.1, 0.9)
    assert np.isnan(lo).all() and np.isnan(hi).all()


def test_item_forecasts_wide_and_backtest_residuals(tmp_path):
    sid = "FOODS_1_001_CA_1_evaluation"
    hold, back = tmp_path / "h.parquet", tmp_path / "b.parquet"
    forecast_rows(hold, fc=lambda m, h: 2.0 if m == "lgbm" else 1.0)
    forecast_rows(back, origins=(1801, 1829), residual=lambda o, h: float(o == 1829))
    wide = dash.item_forecasts(hold, sid, ["tsb", "lgbm"])
    assert list(wide.columns) == ["d_num", "actual", "tsb", "lgbm"]
    assert len(wide) == 28 and (wide["lgbm"] == 2.0).all() and wide["d_num"].iloc[0] == 1914
    res = dash.backtest_residuals(back, sid, "lgbm", [1801, 1829])
    assert sorted(res.tolist()) == [0.0] * 28 + [1.0] * 28
    assert dash.backtest_residuals(back, sid, "lgbm", [1801]).tolist() == [0.0] * 28


def test_item_history_stops_at_the_last_training_day(write_days):
    sid = "FOODS_1_001_CA_1"
    path = write_days({sid: [None, 1, 2, 3, 4, 5]})
    h = dash.item_history(path, sid, last_train=4, days=10)
    assert h["d_num"].tolist() == [1, 2, 3, 4]
    assert h["units"].isna().tolist() == [True, False, False, False]   # not listed = no value, not zero
    assert dash.item_history(path, sid, last_train=4, days=2)["d_num"].tolist() == [3, 4]


def test_series_index(tmp_path, write_days):
    ids = ["FOODS_1_001_CA_1", "FOODS_1_002_CA_2"]
    slice_path = write_days({i: [1, 2] for i in ids})
    hold = tmp_path / "h.parquet"
    forecast_rows(hold, ids=ids)
    idx = dash.series_index(slice_path, hold)
    assert idx["id"].tolist() == ids
    assert idx["store_id"].tolist() == ["CA_1", "CA_2"] and idx["item_id"].tolist() == ["FOODS_1_001", "FOODS_1_002"]
    assert idx["demand_class"].tolist() == ["smooth", "smooth"]


# --- inventory view: pre-computed scenarios ---------------------------------------------------------

def test_sweep_and_scenario_rows_come_from_the_saved_files():
    r = dash.load_results(ROOT)
    f5 = r["f5"]
    rows = dash.sweep_rows(r["sweep"], 0.95, "bias_adjusted")
    assert rows["policy"].tolist() == f5["policies"]
    for p in f5["policies"]:
        saved = f5["holdout"]["main"]["bias_adjusted"][p]["overall"]
        got = rows.set_index("policy").loc[p]
        assert got["fill_rate"] == pytest.approx(saved["fill_rate"])
        assert got["total_cost"] == pytest.approx(saved["total_cost"])
    s = dash.scenario_rows(r["sensitivity"], "stockout_cost_share_of_price", 1.0)
    assert len(s) == 3 and s["policy"].tolist() == f5["policies"]
    ch = dash.scenario_check(f5, "stockout_cost_share_of_price", 1.0)
    assert ch["ranking_by_total_cost"] == s.sort_values("total_cost")["policy"].tolist()
    with pytest.raises(KeyError):
        dash.scenario_rows(r["sensitivity"], "stockout_cost_share_of_price", 0.33)


def test_scenario_options_match_the_saved_assumptions():
    f5 = dash.load_results(ROOT)["f5"]
    opts = dash.scenario_options(f5)
    assert opts["lead_time_days"] == f5["assumptions"]["lead_time_days"]["scenarios"]
    assert set(opts) == set(f5["assumptions"])


# --- portfolio view --------------------------------------------------------------------------------

def test_portfolio_shares_sum_to_one():
    f5 = dash.load_results(ROOT)["f5"]
    for level in ("by_class", "by_dept"):
        t = dash.portfolio_shares(f5, "lgbm", level)
        for col in ("demand_share", "lost_cost_share", "inventory_share"):
            assert t[col].sum() == pytest.approx(1.0)
    t = dash.portfolio_shares(f5, "lgbm", "by_class")
    assert t["group"].tolist() == ["smooth", "erratic", "intermittent", "lumpy"]


# --- live single-item simulation ---------------------------------------------------------------------

def test_item_simulation_uses_the_tested_f5_functions():
    H = 28
    fc = np.full(H, 2.0)
    actual = np.tile([0.0, 3.0, 4.0, 1.0], 7)
    rng = np.random.default_rng(0)
    res = pd.DataFrame([{"id": "x", "origin": o, "h": h, "method": "lgbm", "residual": float(rng.normal())}
                        for o in (1801, 1829, 1857, 1885) for h in range(1, H + 1)])
    out = dash.item_simulation(fc, actual, res, "lgbm", "x", price=3.0, lead=2, review=7, service_level=0.95,
                               ss_method="bias_adjusted", spread_origins=[1801, 1829, 1857, 1885],
                               forecast_origin=1913, hold=0.25, short=0.25)
    mean, sd, _ = inv.spread_from_backtest(res, "lgbm", np.array(["x"]), 9, [1801, 1829, 1857, 1885], 1913, H)
    ss = inv.safety_stock(mean, sd, inv.z_value(0.95), "bias_adjusted")
    sim = inv.simulate(fc[None, :], actual[None, :], ss, 2, 7)
    m = inv.series_measures(sim, np.array([3.0]), 0.25, 0.25).iloc[0]
    assert out["safety_stock"] == pytest.approx(float(ss[0]))
    assert out["measures"]["fill_rate"] == pytest.approx(m["served"] / m["demand"])
    assert out["measures"]["total_cost"] == pytest.approx(m["total_cost"])
    assert out["days"]["on_hand"].tolist() == sim.on_hand[0].tolist()
    assert out["days"]["measured"].sum() == 21


def test_item_simulation_refuses_holdout_errors():
    res = pd.DataFrame([{"id": "x", "origin": 1913, "h": 1, "method": "lgbm", "residual": 0.0}])
    with pytest.raises(inv.LeakageError):
        dash.item_simulation(np.ones(28), np.ones(28), res, "lgbm", "x", 1.0, 2, 7, 0.95, "bias_adjusted",
                             [1913], 1913, 0.25, 0.25)


# --- the app ------------------------------------------------------------------------------------

def test_app_without_processed_data_explains_and_still_shows_results(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FORECAST_PROCESSED_DIR", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 3
    text = " ".join(str(x.value) for x in at.info) + " ".join(str(x.value) for x in at.markdown)
    assert "kaggle" in text.lower() and "run_f1" in text
    shown = " ".join(str(m.value) for m in at.metric)
    f5 = dash.load_results(ROOT)["f5"]
    assert f"{f5['holdout']['main']['bias_adjusted']['lgbm']['overall']['fill_rate']:.1%}" in shown


def test_app_inventory_controls_select_saved_scenarios(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FORECAST_PROCESSED_DIR", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.select_slider(key="service_level").set_value(0.9).run()
    at.selectbox(key="assumption").set_value("stockout_cost_share_of_price").run()
    at.select_slider(key="assumption_value").set_value(1.0).run()
    assert not at.exception
    r = dash.load_results(ROOT)
    expected = dash.sweep_rows(r["sweep"], 0.9, "bias_adjusted").set_index("policy").loc["lgbm", "fill_rate"]
    assert f"{expected:.1%}" in " ".join(str(m.value) for m in at.metric)
