"""Checks on the real files and the prepared slice. Slow; run with: pytest -m realdata
(needs data/raw/ from Kaggle and a finished `python -m forecast.run_f1`)."""
import json

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
