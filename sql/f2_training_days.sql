-- F2 reads the slice through this view only: training days (d_num <= last training day, so the
-- holdout is never read) on which the item was listed (unlisted = not on sale, not zero demand).
-- {parquet} and {last_train} are filled in by demand_patterns.open_training_days.
CREATE OR REPLACE VIEW train_days AS
SELECT id, item_id, dept_id, store_id, d_num, date, wday, weekday, month, year,
       event_name_1, event_type_1, event_name_2, event_type_2, snap, sell_price, units
FROM read_parquet('{parquet}')
WHERE d_num <= {last_train}
  AND is_listed;
