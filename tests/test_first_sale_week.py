"""First priced week vs first week with a sale, on hand-made day tables (weeks are Saturday-Friday,
day 1 = week 11101, as in conftest.day_table)."""
from forecast.first_sale_week import first_priced_week_stats


def test_counts_series_starting_after_week_1_and_selling_in_their_first_priced_week(write_days):
    w = [None] * 7                      # one unpriced week
    path = write_days({
        "FOODS_1_001_CA_1": [1] * 21,                       # priced from week 1: left out (left-censored)
        "FOODS_1_002_CA_1": w + [0, 0, 3, 0, 0, 0, 0] + [1] * 7,   # starts week 2, sells in it
        "FOODS_1_003_CA_1": w + w + [0] * 7,                # starts week 3, no sale in its first week
        "FOODS_1_004_CA_1": w + [0] * 7 + [2] * 7,          # starts week 2, first sale in week 3
    })
    s = first_priced_week_stats(path)
    assert s["series"] == 4
    assert s["series_priced_from_first_week"] == 1
    assert s["series_starting_after_first_week"] == 3
    assert s["sell_in_first_priced_week"] == 1
    assert s["no_sale_in_first_priced_week"] == 2


def test_every_series_counted_once(write_days):
    path = write_days({"FOODS_1_001_CA_1": [None] * 7 + [1] * 7, "FOODS_1_002_CA_1": [2] * 14})
    s = first_priced_week_stats(path)
    assert s["series_priced_from_first_week"] + s["series_starting_after_first_week"] == s["series"] == 2
