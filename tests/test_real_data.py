"""Checks on the real files and the prepared slice. Slow; run with: pytest -m realdata
(needs data/raw/ from Kaggle and a finished `python -m forecast.run_f1`)."""
import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

from forecast import config, raw_files

pytestmark = pytest.mark.realdata
ROOT = config.ROOT


def test_raw_files_match_recorded_fingerprints():
    raw_files.verify(config.raw_paths(config.load("data")), config.load("raw_files"))


def test_recorded_row_counts():
    fp = config.load("raw_files")
    assert fp["sales"]["data_rows"] == 30490
    assert fp["calendar"]["data_rows"] == 1969


def test_slice_parquet_has_every_series_day_once():
    data = config.load("data")
    path = ROOT / data["processed_dir"] / data["slice_parquet"]
    report = json.loads((ROOT / "results" / "f1_data_report.json").read_text())
    c = report["slice_checks"]
    assert pq.ParquetFile(path).metadata.num_rows == c["n_rows"] == c["n_series"] * c["n_days"]
    assert c["duplicate_series_days"] == 0
    assert c["sales_while_unlisted"] == 0


def test_slice_matches_configured_scope():
    data = config.load("data")
    scope = config.load("scope")
    path = ROOT / data["processed_dir"] / data["slice_parquet"]
    t = pq.read_table(path, columns=["state_id", "cat_id"])
    assert set(t.column("state_id").unique().to_pylist()) == {scope["state"]}
    assert set(t.column("cat_id").unique().to_pylist()) == {scope["category"]}


def test_f2_reads_only_training_days():
    from forecast import demand_patterns as dp
    data = config.load("data")
    path = ROOT / data["processed_dir"] / data["slice_parquet"]
    con, last_train = dp.open_training_days(path, config.load("holdout")["holdout_days"], "1GB", 2)
    assert last_train == 1913  # holdout d_1914..d_1941 is never read
    assert con.execute("SELECT MAX(d_num), BOOL_AND(sell_price IS NOT NULL) FROM train_days").fetchone() == (1913, True)
    con.close()


def test_f2_results_are_consistent():
    r = json.loads((ROOT / "results" / "f2_demand_patterns.json").read_text())
    f1 = json.loads((ROOT / "results" / "f1_data_report.json").read_text())
    assert r["training"]["last_d"] == f1["holdout_reserved"]["first_d"] - 1
    assert sum(c["n_series"] for c in r["classes"]) == r["training"]["n_series"] == f1["slice_checks"]["n_series"]
    assert sum(c["units"] for c in r["classes"]) == r["training"]["units"]
    occ = pd.read_csv(ROOT / "results" / "f2_event_occurrences.csv")
    assert occ["date"].max() < f1["holdout_reserved"]["first_date"]


def test_f3_panel_stops_before_the_holdout():
    from forecast import backtest as bt
    data = config.load("data")
    p = bt.load_panel(ROOT / data["processed_dir"] / data["slice_parquet"],
                      config.load("holdout")["holdout_days"], "1GB", 2)
    assert p.last_train == 1913 and p.units.shape == (5748, 1913)
    assert p.wday[0] == 1 and set(p.wday.tolist()) == set(range(1, 8))  # d_1 is a Saturday


def test_f3_results_are_consistent():
    r = json.loads((ROOT / "results" / "f3_backtest.json").read_text())
    f1 = json.loads((ROOT / "results" / "f1_data_report.json").read_text())
    assert r["setup"]["origins"] == [1801, 1829, 1857, 1885]
    assert max(c["last_scored_d"] for c in r["coverage"]) < f1["holdout_reserved"]["first_d"]
    for c in r["coverage"]:
        assert c["n_scored"] + c["n_short_history"] + c["n_no_window_days"] == c["n_series"]
        assert sum(c["classes"].values()) == c["n_scored"]
    assert r["setup"]["n_series_origin_pairs"] == sum(c["n_scored"] for c in r["coverage"])
    data = config.load("data")
    ps = pd.read_parquet(ROOT / data["processed_dir"] / "f3_series_metrics.parquet",
                         columns=["origin", "baseline", "n_days"])
    assert len(ps) == r["setup"]["n_series_origin_pairs"] * len(r["setup"]["baselines"])
    assert (ps["n_days"] >= 1).all() and (ps["n_days"] <= r["setup"]["horizon_days"]).all()


def _f4_data():
    from forecast import model as m
    data = config.load("data")
    return m.load_data(ROOT / data["processed_dir"] / data["slice_parquet"], config.load("holdout")["holdout_days"],
                       config.load("model"), calendar_csv=config.raw_paths(data)["calendar"],
                       state=config.load("scope")["state"], memory_limit="1GB", threads=2)


def test_f4_guard_and_final_recipe_on_the_real_slice():
    import numpy as np
    from forecast import model as m
    d = _f4_data()
    cfg, bt_cfg = config.load("model"), config.load("backtest")
    assert d.first_holdout == 1914 and d.panel.last_train == 1913
    with pytest.raises(m.HoldoutError):
        m.rows_at(d, 1913, 28, cfg, bt_cfg, with_target=False)       # forecasting the holdout needs the flag
    X, y, meta = m.training_table(d, 1913, cfg, bt_cfg)                # the final training rows do not
    assert meta["d_num"].max() == 1913 and np.isfinite(y).all()
    assert (X.nbytes + y.nbytes) / 1e9 <= m.estimate_table_gb(5748, 28, cfg["training_origins"]["n_origins"],
                                                              X.shape[1])
    assert d.cal.loc[1914:1941].shape[0] == 28                          # calendar facts of the holdout days exist
    assert d.cal.index[d.cal["closed"]].tolist() == [331, 697, 1062, 1427, 1792]   # the 5 Christmas days


def test_f4_features_ignore_everything_after_the_origin_on_real_data():
    import numpy as np
    from forecast import model as m
    d = _f4_data()
    cfg, bt_cfg = config.load("model"), config.load("backtest")
    X0, _, _ = m.rows_at(d, 1801, 28, cfg, bt_cfg, with_target=False)
    c = d.panel.col(1802)
    d.panel.units[:, c:] = 50.0
    d.panel.price[:, c:] = 0.5
    d.panel.units[::3, c:c + 10] = np.nan
    X1, _, _ = m.rows_at(d, 1801, 28, cfg, bt_cfg, with_target=False)
    np.testing.assert_array_equal(X0, X1)


def test_f4_results_are_consistent():
    import hashlib
    r = json.loads((ROOT / "results" / "f4_backtest.json").read_text())
    f3 = json.loads((ROOT / "results" / "f3_backtest.json").read_text())
    cfg, final = config.load("model"), config.load("model_final")
    s = r["setup"]
    assert s["origins"] == f3["setup"]["origins"] and s["first_holdout_d"] == 1914
    assert s["f3_baselines_match_f3_results"]
    assert s["n_series_origin_pairs"] == f3["setup"]["n_series_origin_pairs"]
    for b in s["baselines"]:
        assert r["overall"][b]["wrmsse"] == pytest.approx(f3["overall"][b]["wrmsse"], abs=1e-12)
    assert len({m["n"] for m in r["overall"].values()}) == 1          # every method on the same pairs
    assert len(r["grid"]) == len(cfg["grid"]["settings"]) * len(cfg["grid"]["rounds"])
    assert r["chosen"]["wrmsse"] == min(g["wrmsse"] for g in r["grid"])
    assert r["overall"]["lgbm"]["wrmsse"] == pytest.approx(r["chosen"]["wrmsse"], abs=1e-12)
    for t in r["training_tables"]:
        assert t["last_target_d"] <= t["origin"] < 1914
    assert final["chosen_setting"] == r["chosen"]["setting"] and final["rounds"] == r["chosen"]["rounds"]
    assert final["model_yaml_sha256"] == hashlib.sha256((config.CONFIG_DIR / "model.yaml").read_bytes()).hexdigest()
    assert final["final_run"]["forecast_origin"] == 1913 and final["final_run"]["forecast_days"] == [1914, 1941]
    data = config.load("data")
    fc = pd.read_parquet(ROOT / data["processed_dir"] / "f4_backtest_forecasts.parquet",
                         columns=["origin", "method", "d_num", "residual", "forecast", "actual"])
    assert len(fc) == r["f5_forecasts"]["rows"] == s["n_series_origin_pairs"] * 28 * len(s["methods"])
    assert fc["d_num"].max() <= 1913
    ok = fc["actual"].notna()
    assert (fc.loc[ok, "residual"] - (fc.loc[ok, "actual"] - fc.loc[ok, "forecast"])).abs().max() < 1e-5


def test_f4_holdout_result_is_consistent():
    r = json.loads((ROOT / "results" / "f4_holdout.json").read_text())
    final = config.load("model_final")
    assert r["commit"].startswith("1a31ed9")                          # the frozen part-A commit
    assert r["model_final"] == final                                  # scored with the frozen settings
    assert r["setup"]["holdout_days"] == [1914, 1941] and r["setup"]["forecast_origin"] == 1913
    assert r["setup"]["best_baseline_from_backtest"] == json.loads(
        (ROOT / "results" / "f4_backtest.json").read_text())["best_baseline"]["name"]
    assert len({m["n"] for m in r["overall"].values()}) == 1          # every method on the same series
    assert r["coverage"][0]["n_scored"] == r["overall"]["lgbm"]["n"]


def test_f5_spreads_come_from_backtest_origins_only():
    r = json.loads((ROOT / "results" / "f5_inventory.json").read_text())
    h, rb = r["holdout"]["setup"], r["robustness"]["setup"]
    assert h["forecast_origin"] == 1913 and h["spread_origins"] == [1801, 1829, 1857, 1885]
    assert rb["forecast_origin"] == 1885 and rb["spread_origins"] == [1801, 1829, 1857]
    for s in (h, rb):
        assert max(s["spread_origins"]) + 28 <= s["forecast_origin"]      # error days end by the forecast origin
        assert s["n_simulated"] + s["n_excluded_no_backtest"] == s["n_series"]
        assert s["measured_d"][0] > s["forecast_origin"] and s["measured_d"][1] <= s["forecast_origin"] + 28


def test_f5_holdout_errors_cannot_change_the_safety_stock_on_real_data():
    import numpy as np
    from forecast import inventory as inv, run_f5

    processed = ROOT / "data" / "processed"
    w = run_f5.load_window(processed / "f4_holdout_forecasts.parquet", ["lgbm"], 1913, 28)
    bt_err = run_f5.load_backtest_errors(processed / "f4_backtest_forecasts.parquet", ["lgbm"], [1801, 1829, 1857, 1885])
    hold = pd.read_parquet(processed / "f4_holdout_forecasts.parquet", columns=["id", "origin", "h", "method", "residual"])
    hold = hold[hold["method"] == "lgbm"]
    args = ("lgbm", w.ids, 9, [1801, 1829, 1857, 1885])
    a = inv.spread_from_backtest(bt_err, *args, forecast_origin=1913, horizon=28)
    b = inv.spread_from_backtest(pd.concat([bt_err, hold.assign(residual=hold["residual"] * 100)]), *args,
                                 forecast_origin=1913, horizon=28)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_f5_results_are_consistent():
    r = json.loads((ROOT / "results" / "f5_inventory.json").read_text())
    for w in ("holdout", "robustness"):
        for method, by_pol in r[w]["main"].items():
            for o in (g["overall"] for g in by_pol.values()):
                assert 0 < o["fill_rate"] <= 1 and 0 < o["cycle_service_level"] <= 1
                assert abs(o["total_cost"] - o["holding_cost"] - o["lost_cost"]) < 1e-6
                assert abs(o["served"] + o["lost"] - o["demand"]) < 1e-6
        sw = pd.DataFrame(r[w]["sweep"])
        for _, g in sw.groupby(["policy", "ss_method"]):   # more safety stock never lowers stock or fill rate
            g = g.sort_values("service_level")
            assert g["avg_inv_money"].is_monotonic_increasing and g["fill_rate"].is_monotonic_increasing
    # the doc is exactly what the saved results render to
    from forecast import render_f5
    doc = (ROOT / "docs" / "templates" / "F5_inventory.md").read_text().format(**render_f5.values(r))
    assert doc == (ROOT / "docs" / "F5_inventory.md").read_text()
