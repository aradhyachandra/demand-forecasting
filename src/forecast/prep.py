"""Data preparation in DuckDB: load the raw CSVs, reshape sales to long format, join calendar and
prices, check the result and save the slice as parquet. The SQL itself lives in sql/."""
from __future__ import annotations

from pathlib import Path

import duckdb

from forecast.config import SQL_DIR

CALENDAR_TYPES = {
    "date": "DATE",
    "wm_yr_wk": "INTEGER",
    "weekday": "VARCHAR",
    "wday": "TINYINT",
    "month": "TINYINT",
    "year": "SMALLINT",
    "d": "VARCHAR",
    "event_name_1": "VARCHAR",
    "event_type_1": "VARCHAR",
    "event_name_2": "VARCHAR",
    "event_type_2": "VARCHAR",
    "snap_CA": "TINYINT",
    "snap_TX": "TINYINT",
    "snap_WI": "TINYINT",
}
PRICE_TYPES = {
    "store_id": "VARCHAR",
    "item_id": "VARCHAR",
    "wm_yr_wk": "INTEGER",
    "sell_price": "DOUBLE",
}
ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def connect(memory_limit: str = "2GB", threads: int = 4,
            database: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    """Open DuckDB. For the real data pass a database file: tables then spill to disk instead of
    having to fit inside memory_limit (in memory, the full slice ran far slower)."""
    con = duckdb.connect(str(database))
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET threads = {int(threads)}")
    con.execute("SET preserve_insertion_order = true")
    con.execute("SET enable_progress_bar = false")
    return con


def _types_sql(types: dict[str, str]) -> str:
    return "{" + ", ".join(f"'{k}': '{v}'" for k, v in types.items()) + "}"


def register_raw(con: duckdb.DuckDBPyConnection, paths: dict[str, Path]) -> None:
    """Create views raw_sales, raw_calendar and raw_prices over the CSV files.

    Sales: the six id columns are text and every d_ column an integer (all day columns are
    detected from the header)."""
    sales = Path(paths["sales"]).as_posix()
    header = con.execute(
        f"SELECT * FROM read_csv('{sales}', header = true, all_varchar = true) LIMIT 0"
    ).description
    day_cols = [c[0] for c in header if c[0].startswith("d_")]
    sales_types = {c: "VARCHAR" for c in ID_COLUMNS}
    sales_types.update({c: "INTEGER" for c in day_cols})
    con.execute(
        f"CREATE OR REPLACE VIEW raw_sales AS SELECT * FROM read_csv('{sales}', header = true, "
        f"columns = {_types_sql({c[0]: sales_types[c[0]] for c in header})})"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW raw_calendar AS SELECT * FROM read_csv("
        f"'{Path(paths['calendar']).as_posix()}', header = true, columns = {_types_sql(CALENDAR_TYPES)})"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW raw_prices AS SELECT * FROM read_csv("
        f"'{Path(paths['prices']).as_posix()}', header = true, columns = {_types_sql(PRICE_TYPES)})"
    )


def run_sql(con: duckdb.DuckDBPyConnection, name: str, params: dict | None = None,
            sql_dir: Path = SQL_DIR):
    sql = (Path(sql_dir) / name).read_text(encoding="utf-8")
    return con.execute(sql, params or {})


def scope_options(con) -> list[dict]:
    rows = run_sql(con, "scope_options.sql").fetchall()
    cols = ["state_id", "cat_id", "n_series", "n_stores", "n_items"]
    return [dict(zip(cols, r)) for r in rows]


def n_days(con) -> int:
    return sum(1 for c in con.execute("SELECT * FROM raw_sales LIMIT 0").description
               if c[0].startswith("d_"))


def build_slice(con, state: str, category: str) -> None:
    """Create tables sales_long and slice for one state x category."""
    run_sql(con, "sales_long.sql", {"state": state, "category": category})
    run_sql(con, "slice_joined.sql")
    con.execute("DROP TABLE sales_long")


def quality_checks(con) -> dict:
    cur = run_sql(con, "quality_checks.sql")
    names = [c[0] for c in cur.description]
    row = cur.fetchone()
    out = {}
    for k, v in zip(names, row):
        out[k] = v.isoformat() if hasattr(v, "isoformat") else (int(v) if v is not None else None)
    return out


def listing_gaps(con) -> dict:
    cur = run_sql(con, "listing_gaps.sql")
    names = [c[0] for c in cur.description]
    return {k: int(v or 0) for k, v in zip(names, cur.fetchone())}


def price_key_duplicates(con) -> int:
    """Rows in sell_prices that repeat a (store, item, week) key; must be 0 for the join."""
    return int(con.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT (store_id, item_id, wm_yr_wk)) FROM raw_prices"
    ).fetchone()[0])


def write_parquet(con, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY slice TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
