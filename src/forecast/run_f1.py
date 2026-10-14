"""F1: fingerprint the raw files, load them into DuckDB, build the configured slice in SQL, check
it, save it as parquet and write results/f1_data_report.json + docs/F1_data_and_sql.md.

    python -m forecast.run_f1            # verify fingerprints, build the slice
    python -m forecast.run_f1 --record   # first run only: record the raw-file fingerprints
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from forecast import config, prep, raw_files

ROOT = config.ROOT
FINGERPRINTS = config.CONFIG_DIR / "raw_files.yaml"
REPORT = ROOT / "results" / "f1_data_report.json"
TEMPLATE = ROOT / "docs" / "templates" / "F1_data_and_sql.md"
DOC = ROOT / "docs" / "F1_data_and_sql.md"
HOLDOUT_DAYS = 28  # the M5 horizon; the plan's final holdout is the last 28 days (PROJECT_PLAN 4)


def estimate_pandas_bytes_per_row(parquet: Path, sample_rows: int = 200_000) -> float:
    """Memory per row in pandas with the id columns as categories (how later steps load it)."""
    pf = pq.ParquetFile(parquet)
    batch = next(pf.iter_batches(batch_size=sample_rows))
    df = batch.to_pandas()
    for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "weekday",
                "event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        df[col] = df[col].astype("category")
    return float(df.memory_usage(deep=True).sum() / len(df))


def main(record: bool = False) -> dict:
    data_cfg = config.load("data")
    scope = config.load("scope")
    paths = config.raw_paths(data_cfg)

    if record:
        raw_files.write_fingerprints(raw_files.fingerprint(paths), FINGERPRINTS)
    expected = config.load("raw_files")
    raw_files.verify(paths, expected)

    # FORECAST_WORK_DB can move the scratch database elsewhere (used on the remote test machine,
    # which may not delete files inside the project folder). Results do not depend on its location.
    work_db = Path(os.environ.get("FORECAST_WORK_DB")
                   or ROOT / data_cfg["processed_dir"] / data_cfg["duckdb"]["work_db"])
    work_db.parent.mkdir(parents=True, exist_ok=True)
    for f in (work_db, work_db.with_name(work_db.name + ".wal")):
        f.unlink(missing_ok=True)  # rebuilt from the raw files on every run
    con = prep.connect(data_cfg["duckdb"]["memory_limit"], data_cfg["duckdb"]["threads"], work_db)
    prep.register_raw(con, paths)

    options = prep.scope_options(con)
    total_days = prep.n_days(con)
    calendar_days = int(con.execute("SELECT COUNT(*) FROM raw_calendar").fetchone()[0])

    prep.build_slice(con, scope["state"], scope["category"])
    checks = prep.quality_checks(con)
    checks["price_key_duplicates"] = prep.price_key_duplicates(con)
    checks.update(prep.listing_gaps(con))

    holdout = con.execute(
        "SELECT MIN(d_num), MAX(d_num), MIN(date), MAX(date) FROM slice WHERE d_num > ?",
        [total_days - HOLDOUT_DAYS],
    ).fetchone()
    per_store = con.execute(
        "SELECT store_id, COUNT(DISTINCT id) AS n_series, SUM(units) AS units "
        "FROM slice GROUP BY store_id ORDER BY store_id"
    ).fetchall()
    per_dept = con.execute(
        "SELECT dept_id, COUNT(DISTINCT id) AS n_series, SUM(units) AS units "
        "FROM slice GROUP BY dept_id ORDER BY dept_id"
    ).fetchall()

    out = ROOT / data_cfg["processed_dir"] / data_cfg["slice_parquet"]
    prep.write_parquet(con, out)
    con.close()

    bytes_per_row = estimate_pandas_bytes_per_row(out)
    n_rows = checks["n_rows"]
    report = {
        "raw_files": expected,
        "raw_days": total_days,
        "calendar_days": calendar_days,
        "all_series": sum(o["n_series"] for o in options),
        "scope_options": options,
        "scope": {"state": scope["state"], "category": scope["category"]},
        "slice_checks": checks,
        "slice_per_store": [{"store_id": s, "n_series": int(n), "units": int(u)} for s, n, u in per_store],
        "slice_per_dept": [{"dept_id": s, "n_series": int(n), "units": int(u)} for s, n, u in per_dept],
        "holdout_reserved": {
            "days": HOLDOUT_DAYS,
            "first_d": int(holdout[0]), "last_d": int(holdout[1]),
            "first_date": holdout[2].isoformat(), "last_date": holdout[3].isoformat(),
        },
        "parquet": {
            "path": str(out.relative_to(ROOT)),
            "bytes_on_disk": out.stat().st_size,
            "rows": pq.ParquetFile(out).metadata.num_rows,
        },
        "memory_estimate": {
            "pandas_bytes_per_row_with_categories": round(bytes_per_row, 1),
            "pandas_total_gb": round(bytes_per_row * n_rows / 1e9, 2),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    render_doc(report)
    return report


def _fmt(n) -> str:
    return f"{n:,}"


def render_doc(report: dict) -> None:
    c = report["slice_checks"]
    opts_rows = "\n".join(
        f"| {o['state_id']} | {o['cat_id']} | {_fmt(o['n_series'])} | {o['n_stores']} | "
        f"{_fmt(o['n_items'])} | {_fmt(o['n_series'] * report['raw_days'])} |"
        for o in report["scope_options"]
    )
    store_rows = "\n".join(
        f"| {s['store_id']} | {_fmt(s['n_series'])} | {_fmt(s['units'])} |" for s in report["slice_per_store"]
    )
    dept_rows = "\n".join(
        f"| {s['dept_id']} | {_fmt(s['n_series'])} | {_fmt(s['units'])} |" for s in report["slice_per_dept"]
    )
    raw_rows = "\n".join(
        f"| {v['file']} | {_fmt(v['data_rows'])} | {v['bytes'] / 1e6:.1f} MB | `{v['sha256'][:16]}...` |"
        for v in report["raw_files"].values()
    )
    values = {
        "raw_rows": raw_rows,
        "raw_days": _fmt(report["raw_days"]),
        "calendar_days": _fmt(report["calendar_days"]),
        "all_series": _fmt(report["all_series"]),
        "all_rows": _fmt(report["all_series"] * report["raw_days"]),
        "opts_rows": opts_rows,
        "state": report["scope"]["state"],
        "category": report["scope"]["category"],
        "n_series": _fmt(c["n_series"]),
        "n_rows": _fmt(c["n_rows"]),
        "n_days": _fmt(c["n_days"]),
        "first_date": c["first_date"],
        "last_date": c["last_date"],
        "store_rows": store_rows,
        "dept_rows": dept_rows,
        "zero_pct": f"{100 * c['zero_unit_rows'] / c['n_rows']:.1f}",
        "unlisted_pct": f"{100 * c['unlisted_rows'] / c['n_rows']:.1f}",
        "unlisted_rows": _fmt(c["unlisted_rows"]),
        "listed_zero_pct": f"{100 * c['listed_zero_rows'] / c['listed_rows']:.1f}",
        "gap_rows": _fmt(c["unlisted_after_first_listing"]),
        "gap_series": _fmt(c["series_with_listing_gaps"]),
        "never_listed": _fmt(c["series_never_listed"]),
        "sales_while_unlisted": _fmt(c["sales_while_unlisted"]),
        "duplicates": _fmt(c["duplicate_series_days"]),
        "price_dups": _fmt(c["price_key_duplicates"]),
        "null_units": _fmt(c["null_units"]),
        "negative_units": _fmt(c["negative_units"]),
        "rows_without_date": _fmt(c["rows_without_date"]),
        "total_units": _fmt(c["total_units"]),
        "snap_pct": f"{100 * c['snap_rows'] / c['n_rows']:.1f}",
        "event_pct": f"{100 * c['event_rows'] / c['n_rows']:.1f}",
        "h_first_d": report["holdout_reserved"]["first_d"],
        "h_last_d": report["holdout_reserved"]["last_d"],
        "h_first_date": report["holdout_reserved"]["first_date"],
        "h_last_date": report["holdout_reserved"]["last_date"],
        "parquet_mb": f"{report['parquet']['bytes_on_disk'] / 1e6:.0f}",
        "mem_gb": f"{report['memory_estimate']['pandas_total_gb']:.2f}",
        "bytes_per_row": f"{report['memory_estimate']['pandas_bytes_per_row_with_categories']:.0f}",
    }
    DOC.write_text(TEMPLATE.read_text(encoding="utf-8").format(**values), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="record raw-file fingerprints first")
    args = parser.parse_args()
    r = main(record=args.record)
    print(json.dumps(r["slice_checks"], indent=2))
