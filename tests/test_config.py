from forecast import config


def test_scope_has_state_category_and_reason():
    scope = config.load("scope")
    assert scope["state"] in {"CA", "TX", "WI"}
    assert scope["category"] in {"FOODS", "HOBBIES", "HOUSEHOLD"}
    assert len(scope["reason"]) > 40


def test_data_config_names_the_three_raw_files():
    data = config.load("data")
    assert set(data["files"]) == {"sales", "calendar", "prices"}
    paths = config.raw_paths(data)
    assert paths["sales"].name == "sales_train_evaluation.csv"
    assert data["duckdb"]["memory_limit"] and data["duckdb"]["threads"] >= 1
