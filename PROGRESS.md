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


## F2: Demand exploration (done 2026-10-15)

Built:
- `config/holdout.yaml` (holdout_days 28, reason) and `config/demand_patterns.yaml` (class cut-offs,
  calendar and event comparison rules, price-cut definition, each with a reason).
- `sql/f2_*.sql`: `train_days` view (training days only, listed days only; the holdout is never
  read), per-series stats, daily totals, weekly series, price change vs sales check.
- `src/forecast/demand_patterns.py`: Syntetos-Boylan-Croston classes, weekday / month / SNAP
  profiles, event comparison windows, price-cut and no-change windows, week-by-week cut profile.
- `src/forecast/plots.py` (matplotlib 3.10.9, now pinned) and `src/forecast/run_f2.py`.
- `run_f1` now reads the holdout length from `config/holdout.yaml` (was a constant; output unchanged).

Outputs: `results/f2_demand_patterns.json`, `results/f2_event_effects.csv`,
`results/f2_event_occurrences.csv`, five figures in `docs/figures/`, `docs/F2_demand_patterns.md`.
Per-series and per-cut tables in `data/processed/` (gitignored).

Tests: 22 new unit tests on hand-made day tables written to parquet (holdout cut-off, unlisted days,
class edge cases, weekday/month/SNAP, event windows, price cuts); 2 new real-data checks.
Totals: 44 unit, 6 real-data, all pass. Re-running F2 gives identical outputs.

Main findings are in `docs/F2_demand_patterns.md`. Most series are intermittent or lumpy. Weekends
and SNAP days sell more. Monday federal holidays sell more, while Christmas, Thanksgiving and New Year
sell less. Price changes in M5 show up only in weeks with sales, so simple before/after price effects
are biased (see NOTES).

How to run: `.venv/bin/python -m forecast.run_f2` (after run_f1).

Next: F3 (baselines and rolling-origin backtest harness).
