"""F5 runner helpers on a tiny hand-made forecast table (no real data)."""
import numpy as np
import pandas as pd
import pytest

from forecast import run_f5


def forecast_table(path, ids=("b", "a"), methods=("tsb", "lgbm"), origin=1913, H=4):
    rows = []
    for m in methods:
        for k, sid in enumerate(ids):
            for h in range(1, H + 1):
                f = 1.0 + k + (0.5 if m == "lgbm" else 0)
                rows.append({"id": sid, "dept_id": "FOODS_1", "demand_class": "smooth", "origin": origin, "h": h,
                             "d_num": origin + h, "method": m, "forecast": f, "actual": float(h), "residual": 999.0})
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_load_window_shapes_sorts_and_never_reads_residuals(tmp_path):
    p = tmp_path / "f.parquet"
    forecast_table(p)
    w = run_f5.load_window(p, ["tsb", "lgbm"], 1913, 4)
    assert w.ids.tolist() == ["a", "b"]
    assert w.forecasts["tsb"].tolist() == [[2.0] * 4, [1.0] * 4]
    assert w.forecasts["lgbm"][0, 0] == 2.5
    assert w.actual.tolist() == [[1, 2, 3, 4]] * 2
    assert not hasattr(w, "residual")


def test_load_window_refuses_a_series_with_missing_days(tmp_path):
    p = tmp_path / "f.parquet"
    forecast_table(p)
    df = pd.read_parquet(p)
    df.iloc[1:].to_parquet(p, index=False)   # drop one day of one series
    with pytest.raises(AssertionError):
        run_f5.load_window(p, ["tsb", "lgbm"], 1913, 4)


def test_equal_fill_reads_each_policy_curve():
    sw = pd.DataFrame([
        {"policy": p, "ss_method": "bias_adjusted", "service_level": sl, "fill_rate": f, "avg_inv_money": m}
        for p, pts in {"seasonal_naive": [(0.8, 0.90, 200), (0.9, 0.96, 400)],
                       "tsb": [(0.8, 0.90, 150), (0.9, 0.96, 300)],
                       "lgbm": [(0.8, 0.90, 120), (0.9, 0.96, 240)]}.items() for sl, f, m in pts])
    out = run_f5.equal_fill(sw, [0.93, 0.99], "bias_adjusted", ["seasonal_naive", "tsb", "lgbm"])
    assert out[0]["inventory_money"] == pytest.approx({"seasonal_naive": 300, "tsb": 225, "lgbm": 180})
    assert out[0]["lgbm_vs_tsb_pct"] == pytest.approx(-20) and out[0]["lgbm_vs_seasonal_naive_pct"] == pytest.approx(-40)
    assert out[1]["inventory_money"]["lgbm"] is None and "lgbm_vs_tsb_pct" not in out[1]


def test_ranking_by_total_cost():
    assert run_f5.ranking({"a": {"total_cost": 3}, "b": {"total_cost": 1}, "c": {"total_cost": 2}}) == ["b", "c", "a"]


def test_runner_top_revenue_flag_and_exclusion():
    ids = np.array(["a", "b", "c"])
    H = 10
    w = run_f5.Window(origin=1913, ids=ids, dept=np.array(["D"] * 3), cls=np.array(["smooth"] * 3),
                      forecasts={"tsb": np.ones((3, H))}, actual=np.ones((3, H)),
                      price=np.array([1.0, 2.0, 3.0]), weight=np.array([5.0, 50.0, 1.0]))
    err = pd.DataFrame([{"id": s, "origin": 1801, "h": h, "method": "tsb", "residual": 0.0}
                        for s in ("a", "b") for h in range(1, 29)])   # "c" has no backtest rows
    r = run_f5.Runner(w, err, [1801], 28, ["tsb"], top_share=0.5)
    s = {"lead": 2, "review": 3, "start_factor": 1.0, "unmet": "lost_sales", "hold": 0.25, "short": 0.25,
         "sl": 0.95, "ss_method": "bias_adjusted"}
    assert r.restrict_to_series_with_spread(5) == 1
    m = r.measures("tsb", s)
    assert len(m) == 2 and m["top_revenue"].tolist() == [False, True]   # of a and b, b has the larger weight
    assert m["lost"].sum() == 0   # forecast = demand, no error: nothing is lost
