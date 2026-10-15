-- One row per series and week: the week's price, units sold and number of listed training days.
-- wk counts 7-day weeks from d_1 (a Saturday, the first day of Walmart's wm_yr_wk weeks), so weeks
-- are consecutive integers; a week cut short by the holdout or a launch has n_days < 7.
SELECT
    id, ANY_VALUE(dept_id) AS dept_id,
    (d_num - 1) // 7 AS wk,
    MAX(sell_price) AS price,
    MIN(sell_price) AS price_min,
    SUM(units) AS units,
    COUNT(*) AS n_days
FROM train_days
GROUP BY id, wk
ORDER BY id, wk;
