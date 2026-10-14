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
