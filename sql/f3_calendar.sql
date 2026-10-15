-- F3: weekday number per training day (1 = Saturday ... 7 = Friday, M5 weeks run Saturday-Friday).
-- Used to find the last full week before an origin, for the last known price.
SELECT DISTINCT d_num, wday
FROM read_parquet('{parquet}')
WHERE d_num <= {last_train}
ORDER BY d_num;
