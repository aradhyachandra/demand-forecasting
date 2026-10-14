Read CLAUDE.md and PROJECT_PLAN.md fully. Do milestone F1 only.

Scope:
- If this folder is not yet a git repository, run `git init -b main` and make a first commit of the
  existing plan files (PROJECT_PLAN.md, CLAUDE.md, FIRST_SESSION_PROMPT.md, .gitignore).
- Repo skeleton (src/forecast/, sql/, tests/, config/, docs/, data/raw gitignored), a Python venv with
  pinned requirements (duckdb, pandas, pyarrow; add others only when needed).
- Load data/raw/sales_train_evaluation.csv, calendar.csv and sell_prices.csv into DuckDB; SQL to
  reshape sales to long format, join calendar and prices; record row counts and sha256 of each raw
  file in config.
- Propose the scope slice (default: California stores, FOODS category) with its number of series and
  rows and estimated memory use; record it in config with a reason.
- Tests first, using small hand-made CSV fixtures for the SQL reshaping and joins.
- Save the prepared slice as parquet under data/processed/ (gitignored).

When done, update PROGRESS.md, write docs/F1_data_and_sql.md in plain language, and list what I
should review. Commit nothing beyond the first plan commit until I say.
