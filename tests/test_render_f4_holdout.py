"""The F4 holdout write-up only formats saved results: it reads the two JSON reports, never trains,
loads data or writes the holdout result. Made-up reports, so any number in the doc must come from them."""
import json

import pytest

from forecast import render_f4_holdout as rh

METHODS = ["naive", "seasonal_naive", "moving_average", "ses", "croston", "sba", "tsb", "lgbm"]
CLASSES = ["smooth", "intermittent"]


def metrics(w, n=10):
    return {"n": n, "n_scaled": n, "wrmsse": w, "mean_rmsse": w - 0.1, "median_rmsse": w - 0.2,
            "mean_mae": 1.0 + w, "mean_rmse": 2.0, "mean_bias": -0.01 * w, "weekly_wape": w / 3}


def wins(share):
    return {"pairs": {"n": 10, "wins": int(10 * share), "ties": 0, "losses": 10 - int(10 * share),
                      "share_wins": share},
            "series": {"n": 10, "wins": int(10 * share), "ties": 0, "losses": 10 - int(10 * share),
                       "share_wins": share}}


def holdout_report(w):
    return {
        "commit": "abcdef1234567890abcdef1234567890abcdef12", "started_utc": "2026-10-15T10:00:00+00:00",
        "run_minutes": 1.5,
        "model_final": {"chosen_setting": "s9_test", "params": {"num_leaves": 7, "learning_rate": 0.2,
                                                               "min_data_in_leaf": 3}, "rounds": 77},
        "setup": {"forecast_origin": 1913, "holdout_days": [1914, 1941], "training_rows": 123,
                  "methods": METHODS, "best_baseline_from_backtest": "tsb"},
        "coverage": [{"origin": 1913, "n_series": 12, "n_scored": 10, "n_short_history": 2,
                      "n_no_window_days": 0, "scored_days": 280, "classes": {"smooth": 4, "intermittent": 6}}],
        "overall": {b: metrics(v) for b, v in w.items()},
        "by_class": {c: {b: metrics(v + 0.01 * i) for b, v in w.items()} for i, c in enumerate(CLASSES)},
        "by_dept": {"FOODS_1": {b: metrics(v) for b, v in w.items()}},
        "by_origin": {"1913": {b: metrics(v) for b, v in w.items()}},
        "wins_vs_reference": {"reference": "seasonal_naive",
                              "overall": {b: wins(0.7) for b in METHODS if b != "seasonal_naive"},
                              "by_class": {c: {b: wins(0.6) for b in METHODS if b != "seasonal_naive"}
                                           for c in CLASSES}},
        "wins_vs_best_baseline": {"reference": "tsb", "overall": wins(0.6)},
        "weekly_totals": [{"origin": 1913, "week": k, "baseline": b, "forecast": 100.0 + k, "actual": 110.0}
                          for k in (1, 2, 3, 4) for b in METHODS],
        "f5_forecasts": {"file": "data/processed/x.parquet", "rows": 2240},
    }


def backtest_report(w):
    return {"overall": {b: metrics(v) for b, v in w.items()},
            "by_class": {c: {b: metrics(v) for b, v in w.items()} for c in CLASSES},
            "best_baseline": {"name": "tsb", "wrmsse": w["tsb"]}}


HOLD = {"naive": 1.5, "seasonal_naive": 1.2, "moving_average": 0.95, "ses": 0.94, "croston": 0.93,
        "sba": 0.901, "tsb": 0.912, "lgbm": 0.789}
BACK = {"naive": 1.4, "seasonal_naive": 1.1, "moving_average": 0.9, "ses": 0.9, "croston": 0.9,
        "sba": 0.9, "tsb": 0.876, "lgbm": 0.811}
F1 = {"holdout_reserved": {"first_d": 1914, "last_d": 1941, "first_date": "2016-04-25", "last_date": "2016-05-22"}}


@pytest.fixture
def paths(tmp_path, monkeypatch):
    res = tmp_path / "results"
    res.mkdir()
    (res / "f4_holdout.json").write_text(json.dumps(holdout_report(HOLD)))
    (res / "f4_backtest.json").write_text(json.dumps(backtest_report(BACK)))
    (res / "f1_data_report.json").write_text(json.dumps(F1))
    monkeypatch.setattr(rh, "RESULTS", res)
    monkeypatch.setattr(rh, "DOC", tmp_path / "F4_holdout.md")
    monkeypatch.setattr(rh, "FIGURE", tmp_path / "fig.png")
    return tmp_path


def test_doc_numbers_come_from_the_saved_reports(paths):
    rh.main()
    doc = (paths / "F4_holdout.md").read_text()
    assert "0.789" in doc and "0.912" in doc and "0.811" in doc     # holdout model, holdout TSB, backtest model
    assert "abcdef1" in doc and "s9_test" in doc and "77 trees" in doc
    assert "2016-04-25" in doc
    assert "{" not in doc.replace("{}", "")                          # every placeholder filled
    assert (paths / "fig.png").exists()


def test_doc_names_a_different_best_baseline_on_the_holdout(paths):
    rh.main()
    doc = (paths / "F4_holdout.md").read_text()
    assert "SBA" in doc and "fixed before the run" in doc              # SBA 0.901 < TSB 0.912 here


def test_renderer_never_trains_loads_data_or_writes_the_result(paths, monkeypatch):
    from forecast import model

    def boom(*a, **k):
        raise AssertionError("the write-up must not touch data or models")

    monkeypatch.setattr(model, "load_data", boom)
    monkeypatch.setattr(model, "fit", boom)
    before = (paths / "results" / "f4_holdout.json").read_bytes()
    rh.main()
    assert (paths / "results" / "f4_holdout.json").read_bytes() == before
