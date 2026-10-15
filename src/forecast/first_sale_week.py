"""Does an item's first priced week coincide with its first week with a sale?

    python -m forecast.first_sale_week     (after run_f1)

The M5 guide says a missing weekly price means "the product was not sold during the examined week".
For series whose first priced week is after the first week of the data (so their start is observed),
this counts how many sell in that first priced week. Reads data/processed/m5_slice.parquet (training
and holdout days alike: it describes the price file, not demand) and writes
results/f6_first_sale_week.json (counts only).
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

from forecast import config

ROOT = config.ROOT
OUT = ROOT / "results" / "f6_first_sale_week.json"


def first_priced_week_stats(slice_parquet: Path) -> dict:
    p = Path(slice_parquet).as_posix().replace("'", "''")
    row = duckdb.sql(f"""
        WITH days AS (SELECT * FROM read_parquet('{p}')),
        first_week AS (SELECT id, MIN(wm_yr_wk) AS wk FROM days WHERE is_listed GROUP BY id),
        sold AS (SELECT d.id, SUM(d.units) AS units FROM days d JOIN first_week f
                 ON d.id = f.id AND d.wm_yr_wk = f.wk GROUP BY d.id),
        data_start AS (SELECT MIN(wm_yr_wk) AS wk FROM days)
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE f.wk = s.wk),
               COUNT(*) FILTER (WHERE f.wk > s.wk),
               COUNT(*) FILTER (WHERE f.wk > s.wk AND sold.units > 0),
               COUNT(*) FILTER (WHERE f.wk > s.wk AND sold.units = 0)
        FROM first_week f JOIN sold USING (id) CROSS JOIN data_start s""").fetchone()
    keys = ["series", "series_priced_from_first_week", "series_starting_after_first_week",
            "sell_in_first_priced_week", "no_sale_in_first_priced_week"]
    return dict(zip(keys, map(int, row)))


def main() -> dict:
    data_cfg = config.load("data")
    stats = first_priced_week_stats(ROOT / data_cfg["processed_dir"] / data_cfg["slice_parquet"])
    report = {"definition": "series whose first priced week (wm_yr_wk with a sell_price) is after the first "
                            "week of the data; sold = units > 0 in that week",
              **stats}
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
