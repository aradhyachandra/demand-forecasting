"""SQL reshaping and joins, checked against the small hand-made tables in tests/fixtures/."""
import math

import pandas as pd

from forecast import prep


def units_of(df, series_id):
    return df.loc[df["id"] == series_id].sort_values("d_num")["units"].tolist()


def test_reshape_gives_one_row_per_series_per_day(ca_foods):
    assert len(ca_foods) == 3 * 4  # 3 CA FOODS series x 4 days
    assert ca_foods.groupby("id").size().tolist() == [4, 4, 4]


def test_reshape_keeps_the_daily_values(ca_foods):
    assert units_of(ca_foods, "FOODS_1_001_CA_1_evaluation") == [0, 2, 1, 3]
    assert units_of(ca_foods, "FOODS_1_002_CA_1_evaluation") == [0, 0, 4, 0]
    assert units_of(ca_foods, "FOODS_1_001_CA_2_evaluation") == [1, 1, 0, 2]
    assert int(ca_foods["units"].sum()) == 14


def test_slice_excludes_other_states_and_categories(ca_foods):
    assert set(ca_foods["state_id"]) == {"CA"}
    assert set(ca_foods["cat_id"]) == {"FOODS"}
    assert not ca_foods["id"].str.contains("HOBBIES|TX_").any()


def test_calendar_join_maps_day_codes_to_dates(ca_foods):
    one = ca_foods[ca_foods["id"] == "FOODS_1_001_CA_1_evaluation"].sort_values("d_num")
    assert one["d_num"].tolist() == [1, 2, 3, 4]
    assert [str(x.date()) for x in pd.to_datetime(one["date"])] == [
        "2011-01-29", "2011-01-30", "2011-01-31", "2011-02-01"]
    assert one["wm_yr_wk"].tolist() == [11101, 11101, 11102, 11102]
    assert one["event_name_1"].tolist()[1] == "SuperBowl"
    assert one["event_name_1"].isna().sum() == 3


def test_calendar_days_without_sales_add_no_rows(ca_foods):
    assert "d_5" not in set(ca_foods["d"])


def test_snap_uses_the_rows_own_state(con, ca_foods):
    one = ca_foods[ca_foods["id"] == "FOODS_1_001_CA_1_evaluation"].sort_values("d_num")
    assert one["snap"].tolist() == [0, 1, 1, 0]  # snap_CA column
    prep.build_slice(con, "TX", "FOODS")
    tx = con.execute("SELECT snap FROM slice ORDER BY d_num").df()
    assert tx["snap"].tolist() == [1, 0, 0, 1]  # snap_TX column


def test_price_join_uses_store_item_and_week(ca_foods):
    one = ca_foods[ca_foods["id"] == "FOODS_1_001_CA_1_evaluation"].sort_values("d_num")
    assert one["sell_price"].tolist() == [2.0, 2.0, 2.5, 2.5]
    other_store = ca_foods[ca_foods["id"] == "FOODS_1_001_CA_2_evaluation"]
    assert set(other_store["sell_price"]) == {1.9}


def test_missing_price_means_not_yet_listed(ca_foods):
    new_item = ca_foods[ca_foods["id"] == "FOODS_1_002_CA_1_evaluation"].sort_values("d_num")
    prices = new_item["sell_price"].tolist()
    assert math.isnan(prices[0]) and math.isnan(prices[1])
    assert prices[2:] == [3.0, 3.0]
    assert new_item["is_listed"].tolist() == [False, False, True, True]


def test_rows_are_ordered_by_series_then_day(ca_foods):
    keys = list(zip(ca_foods["id"], ca_foods["d_num"]))
    assert keys == sorted(keys)


def test_quality_checks(con, ca_foods):
    q = prep.quality_checks(con)
    assert q["n_rows"] == 12
    assert q["n_series"] == 3
    assert q["n_days"] == 4
    assert q["first_date"] == "2011-01-29" and q["last_date"] == "2011-02-01"
    assert q["duplicate_series_days"] == 0
    assert q["null_units"] == 0 and q["negative_units"] == 0 and q["rows_without_date"] == 0
    assert q["zero_unit_rows"] == 5  # 1 + 3 + 1
    assert q["unlisted_rows"] == 2
    assert q["sales_while_unlisted"] == 0
    assert q["total_units"] == 14
    assert q["snap_rows"] == 6  # 3 series x 2 SNAP days
    assert q["event_rows"] == 3  # 3 series x 1 event day


def test_scope_options_count_series_per_state_and_category(con):
    opts = {(o["state_id"], o["cat_id"]): o for o in prep.scope_options(con)}
    assert opts[("CA", "FOODS")]["n_series"] == 3
    assert opts[("CA", "FOODS")]["n_stores"] == 2
    assert opts[("CA", "FOODS")]["n_items"] == 2
    assert opts[("CA", "HOBBIES")]["n_series"] == 1
    assert opts[("TX", "FOODS")]["n_series"] == 1


def test_n_days_counts_day_columns(con):
    assert prep.n_days(con) == 4


def test_price_key_duplicates(con, small_paths, tmp_path):
    assert prep.price_key_duplicates(con) == 0
    dup = tmp_path / "prices_dup.csv"
    text = small_paths["prices"].read_text()
    dup.write_text(text + "CA_1,FOODS_1_001,11101,2.0\n")
    prep.register_raw(con, {**small_paths, "prices": dup})
    assert prep.price_key_duplicates(con) == 1


def test_parquet_round_trip(con, ca_foods, tmp_path):
    out = tmp_path / "sub" / "slice.parquet"
    prep.write_parquet(con, out)
    back = pd.read_parquet(out)
    assert len(back) == 12
    assert int(back["units"].sum()) == 14
    assert list(back.columns) == list(ca_foods.columns)


def test_listing_gaps(con, ca_foods):
    g = prep.listing_gaps(con)
    assert g["unlisted_after_first_listing"] == 0  # FOODS_1_002 is only unlisted before launch
    assert g["series_with_listing_gaps"] == 0
    assert g["series_never_listed"] == 0
    assert g["listed_rows"] == 10
    assert g["listed_zero_rows"] == 3  # d_1 of FOODS_1_001 CA_1, d_4 of FOODS_1_002, d_3 of CA_2


def test_listing_gap_after_launch_is_counted(small_paths, tmp_path):
    prices = tmp_path / "prices_gap.csv"
    prices.write_text("store_id,item_id,wm_yr_wk,sell_price\n"
                      "CA_1,FOODS_1_001,11101,2.0\n")  # listed week 1, missing week 2
    c = prep.connect(memory_limit="512MB", threads=1)
    prep.register_raw(c, {**small_paths, "prices": prices})
    prep.build_slice(c, "CA", "FOODS")
    g = prep.listing_gaps(c)
    assert g["unlisted_after_first_listing"] == 2  # d_3, d_4 of FOODS_1_001 CA_1
    assert g["series_with_listing_gaps"] == 1
    assert g["series_never_listed"] == 2
