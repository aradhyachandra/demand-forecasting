-- Attach the calendar (real dates, week id, events, SNAP days) and the weekly price to every
-- series-day of the slice.
-- * SNAP: the calendar has one flag per state; each row takes its own state's flag.
-- * Price: a LEFT JOIN, because an item has no price in weeks before it was first sold in that
--   store. is_listed = price known for that week.
CREATE OR REPLACE TABLE slice AS
SELECT
    s.id, s.item_id, s.dept_id, s.cat_id, s.store_id, s.state_id,
    s.d,
    CAST(substr(s.d, 3) AS INTEGER) AS d_num,
    c.date,
    c.wm_yr_wk,
    c.weekday,
    c.wday,
    c.month,
    c.year,
    c.event_name_1, c.event_type_1, c.event_name_2, c.event_type_2,
    CAST(CASE s.state_id
        WHEN 'CA' THEN c.snap_CA
        WHEN 'TX' THEN c.snap_TX
        WHEN 'WI' THEN c.snap_WI
    END AS TINYINT) AS snap,
    p.sell_price,
    p.sell_price IS NOT NULL AS is_listed,
    s.units
FROM sales_long AS s
JOIN raw_calendar AS c
    ON c.d = s.d
LEFT JOIN raw_prices AS p
    ON p.store_id = s.store_id
   AND p.item_id = s.item_id
   AND p.wm_yr_wk = c.wm_yr_wk
ORDER BY s.id, d_num;
