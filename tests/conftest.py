from pathlib import Path

import pytest

from forecast import prep

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def small_paths():
    return {
        "sales": FIXTURES / "sales_small.csv",
        "calendar": FIXTURES / "calendar_small.csv",
        "prices": FIXTURES / "prices_small.csv",
    }


@pytest.fixture
def con(small_paths):
    c = prep.connect(memory_limit="512MB", threads=1)
    prep.register_raw(c, small_paths)
    yield c
    c.close()


@pytest.fixture
def ca_foods(con):
    prep.build_slice(con, "CA", "FOODS")
    return con.execute("SELECT * FROM slice").df()


def day_table(units_by_series, prices_by_series=None, events=None, start="2011-01-29"):
    """Hand-made series-day table shaped like data/processed/m5_slice.parquet.

    units_by_series: {id: [units per day]}; None means the item was not listed that day.
    prices_by_series: {id: price or [price per day]} (default 2.0 when listed).
    events: {d_num: event_name}. Day 1 is a Saturday (wday 1), as in M5; SNAP = 1st-10th (CA).
    """
    import numpy as np
    import pandas as pd

    prices_by_series = prices_by_series or {}
    events = events or {}
    rows = []
    for sid, units in units_by_series.items():
        item, store = sid[:11], sid[12:16]
        price = prices_by_series.get(sid, 2.0)
        for i, u in enumerate(units):
            d = i + 1
            date = pd.Timestamp(start) + pd.Timedelta(days=i)
            p = price[i] if isinstance(price, list) else price
            listed = u is not None
            rows.append({
                "id": sid, "item_id": item, "dept_id": item[:7], "cat_id": "FOODS",
                "store_id": store, "state_id": "CA", "d": f"d_{d}", "d_num": d, "date": date,
                "wm_yr_wk": 11101 + i // 7, "weekday": date.day_name(), "wday": i % 7 + 1,
                "month": date.month, "year": date.year,
                "event_name_1": events.get(d), "event_type_1": "Test" if d in events else None,
                "event_name_2": None, "event_type_2": None,
                "snap": int(date.day <= 10),
                "sell_price": float(p) if listed else np.nan, "is_listed": listed,
                "units": int(u) if listed else 0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def write_days(tmp_path):
    """Write a day_table(...) to parquet and return its path (so tests use the real read path)."""
    def _write(*args, **kwargs):
        path = tmp_path / "slice.parquet"
        day_table(*args, **kwargs).to_parquet(path, index=False)
        return path
    return _write
