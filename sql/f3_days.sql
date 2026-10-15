-- F3: every training series-day with its matrix row (s), day, units, weekly price and listing flag.
-- Only the columns the backtest needs, and only training days (d_num <= {last_train}): the holdout
-- d_{first_holdout}.. is never read. Unlisted days (no price) become NaN in the matrices.
WITH series AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY id) - 1 AS s
    FROM (SELECT DISTINCT id FROM read_parquet('{parquet}') WHERE d_num <= {last_train})
)
SELECT series.s, p.d_num, p.units, p.sell_price, p.is_listed
FROM read_parquet('{parquet}') AS p
JOIN series USING (id)
WHERE p.d_num <= {last_train};
