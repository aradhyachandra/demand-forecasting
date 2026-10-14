# Progress

## F1: Repo setup, data loading, SQL preparation, scope (done 2026-10-14)

Built:
- Skeleton: `src/forecast/`, `sql/`, `tests/`, `config/`, `docs/`, `results/`; `data/raw/` and
  `data/processed/` gitignored. `pyproject.toml` (src layout, pytest config), `requirements.txt` pinned.
- `src/forecast/raw_files.py`: sha256 and row counts of the three raw files, recorded once in
  `config/raw_files.yaml`; every run stops if a file changed.
- `src/forecast/prep.py` + `sql/`: DuckDB views over the CSVs (explicit column types), then
  `sales_long.sql` (wide to long, slice only), `slice_joined.sql` (calendar date, events, the row's own
  state's SNAP flag, weekly price by store-item-week; `is_listed` = price known), `scope_options.sql`,
  `quality_checks.sql`, `listing_gaps.sql`.
- `config/scope.yaml`: CA / FOODS with the reason; `config/data.yaml`: paths and DuckDB settings.
- `src/forecast/run_f1.py`: writes `results/f1_data_report.json` and renders
  `docs/F1_data_and_sql.md` from `docs/templates/`, so every number in the doc comes from the report.

Outputs: `data/processed/m5_slice.parquet` (gitignored), `results/f1_data_report.json`.

Tests: 22 unit tests on hand-made tables in `tests/fixtures/` (`pytest`), 4 real-data checks
(`pytest -m realdata`). All pass.

How to run:
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python -m forecast.run_f1      # add --record only when recording new raw files
.venv/bin/python -m pytest && .venv/bin/python -m pytest -m realdata
```

Next: F2 (demand exploration: seasonality, intermittency classes, price and event effects).
