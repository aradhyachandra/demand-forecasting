"""F6 checks on the real data: the dashboard's single-item replay equals the F5 simulation item by item,
the app renders every view on the real data, and the committed report and README are exactly what the
renderer makes from results/. Slow; run with: pytest -m realdata"""
import numpy as np
import pytest

from forecast import backtest as bt
from forecast import config, run_f5
from forecast import dashboard as dash

pytestmark = pytest.mark.realdata
ROOT = config.ROOT
PROCESSED = ROOT / "data" / "processed"
APP = ROOT / "app" / "streamlit_app.py"


def test_single_item_replay_matches_the_f5_run_item_by_item():
    f5 = dash.load_results(ROOT)["f5"]
    s, setup = f5["central"], f5["holdout"]["setup"]
    cfg, bt_cfg, data_cfg = config.load("inventory"), config.load("backtest"), config.load("data")
    H, pols = bt_cfg["horizon_days"], cfg["policies"]
    panel = bt.load_panel(PROCESSED / data_cfg["slice_parquet"], config.load("holdout")["holdout_days"])
    w = run_f5.load_window(PROCESSED / "f4_holdout_forecasts.parquet", pols, setup["forecast_origin"], H)
    run_f5.add_prices(w, panel, bt_cfg["weight_days"])
    errors = run_f5.load_backtest_errors(PROCESSED / "f4_backtest_forecasts.parquet", pols, setup["spread_origins"])
    r = run_f5.Runner(w, errors, setup["spread_origins"], H, pols, cfg["top_revenue_share"])
    assert r.restrict_to_series_with_spread(s["lead"] + s["review"]) == setup["n_excluded_no_backtest"]
    ids = w.ids[r.keep]
    pick = np.random.default_rng(6).choice(len(ids), 25, replace=False)
    for pol in ("lgbm", "seasonal_naive"):
        full = r.measures(pol, run_f5.central(cfg))
        assert full["total_cost"].sum() == pytest.approx(f5["holdout"]["main"]["bias_adjusted"][pol]["overall"]["total_cost"])
        for i in pick:
            sid = ids[i]
            fc = dash.item_forecasts(PROCESSED / "f4_holdout_forecasts.parquet", sid, [pol])
            err = dash.item_backtest_errors(PROCESSED / "f4_backtest_forecasts.parquet", sid, pol, setup["spread_origins"])
            price = dash.last_known_price(PROCESSED / "m5_slice.parquet", sid, setup["forecast_origin"])
            assert price == pytest.approx(w.price[r.keep][i])
            out = dash.item_simulation(fc[pol].to_numpy(), fc["actual"].to_numpy(), err, pol, sid, price, s["lead"],
                                       s["review"], s["sl"], s["ss_method"], setup["spread_origins"],
                                       setup["forecast_origin"], s["hold"], s["short"], s["start_factor"], s["unmet"])
            assert out["measures"]["total_cost"] == pytest.approx(full["total_cost"].iloc[i], rel=1e-5), sid
            assert out["measures"]["avg_inv_units"] == pytest.approx(full["avg_inv_units"].iloc[i], rel=1e-5), sid


def test_app_renders_every_view_on_the_real_data():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception
    assert not at.info                                  # no "data missing" message
    at.selectbox(key="store").set_value("CA_3").run()
    at.selectbox(key="dept").set_value("FOODS_3").run()
    at.radio(key="band_method").set_value("TSB").run()
    at.select_slider(key="item_lead").set_value(4).run()
    at.select_slider(key="item_review").set_value(3).run()
    at.selectbox(key="item_policy").set_value("seasonal_naive").run()
    at.radio(key="portfolio_policy").set_value("tsb").run()
    assert not at.exception
    labels = " ".join(m.label for m in at.metric)
    assert "Safety stock (units)" in labels and "Lost-sales cost (simulated)" in labels


def test_report_and_readme_are_rendered_from_results():
    from forecast import run_f6

    for name, text in run_f6.render().items():
        assert (ROOT / name).read_text(encoding="utf-8") == text, f"{name} is not the rendered version"
