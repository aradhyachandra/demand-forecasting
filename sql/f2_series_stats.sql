-- Per series: listed training days, days with sales, total units, and the mean and sample variance
-- of the non-zero daily sales (inputs to ADI and CV²).
SELECT
    id, ANY_VALUE(item_id) AS item_id, ANY_VALUE(dept_id) AS dept_id, ANY_VALUE(store_id) AS store_id,
    COUNT(*) AS listed_days,
    COUNT(*) FILTER (WHERE units > 0) AS nonzero_days,
    SUM(units) AS units,
    AVG(units) FILTER (WHERE units > 0) AS mean_nz,
    VAR_SAMP(units) FILTER (WHERE units > 0) AS var_nz
FROM train_days
GROUP BY id
ORDER BY id;
