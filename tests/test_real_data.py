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
