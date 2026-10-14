-- How many series each state x category slice has, to choose the scope with numbers.
SELECT
    state_id,
    cat_id,
    COUNT(*) AS n_series,
    COUNT(DISTINCT store_id) AS n_stores,
    COUNT(DISTINCT item_id) AS n_items
FROM raw_sales
GROUP BY state_id, cat_id
ORDER BY state_id, cat_id;
