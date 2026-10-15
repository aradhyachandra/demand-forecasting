-- How often the weekly price changes, split by whether the week had any sales (full 7-day weeks,
-- each compared with the series' previous week). Used to check whether M5 prices are only observed
-- through sales: if a price never changes in a week without sales, it is carried forward.
WITH weeks AS (
    SELECT id, (d_num - 1) // 7 AS wk, MAX(sell_price) AS price, SUM(units) AS units, COUNT(*) AS n_days
    FROM train_days
    GROUP BY id, wk
),
lagged AS (
    SELECT *, LAG(price) OVER (PARTITION BY id ORDER BY wk) AS prev_price
    FROM weeks
)
SELECT
    units = 0 AS zero_sales_week,
    COUNT(*) AS n_weeks,
    COUNT(*) FILTER (WHERE price <> prev_price) AS n_changes
FROM lagged
WHERE prev_price IS NOT NULL AND n_days = 7
GROUP BY zero_sales_week
ORDER BY zero_sales_week;
