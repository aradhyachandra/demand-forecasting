-- Reshape the wide sales table (one column per day, d_1 .. d_1941) into long format
-- (one row per series per day), keeping only the configured slice.
-- Parameters: $state, $category
CREATE OR REPLACE TABLE sales_long AS
WITH scoped AS (
    SELECT *
    FROM raw_sales
    WHERE state_id = $state AND cat_id = $category
)
SELECT
    id, item_id, dept_id, cat_id, store_id, state_id,
    d,
    CAST(units AS INTEGER) AS units
FROM (
    UNPIVOT scoped
    ON COLUMNS('^d_[0-9]+$')
    INTO NAME d VALUE units
);
