-- F3: one row per series in the slice, in a fixed order (row number = matrix row in backtest.Panel).
-- Only training days are read (d_num <= {last_train}); the holdout is never read.
SELECT id, ANY_VALUE(dept_id) AS dept_id
FROM read_parquet('{parquet}')
WHERE d_num <= {last_train}
GROUP BY id
ORDER BY id;
