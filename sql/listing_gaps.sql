-- Days with no price AFTER an item's first listed day in that store (a gap in the listing,
-- as opposed to "not launched yet"). Also the share of zero-sales days among listed days only.
WITH first_listed AS (
    SELECT id, MIN(d_num) AS first_d
    FROM slice
    WHERE is_listed
    GROUP BY id
)
SELECT
    SUM(CASE WHEN NOT s.is_listed AND s.d_num > f.first_d THEN 1 ELSE 0 END) AS unlisted_after_first_listing,
    COUNT(DISTINCT CASE WHEN NOT s.is_listed AND s.d_num > f.first_d THEN s.id END) AS series_with_listing_gaps,
    (SELECT COUNT(*) FROM (SELECT DISTINCT id FROM slice) AS a) - COUNT(DISTINCT f.id) AS series_never_listed,
    SUM(CASE WHEN s.is_listed THEN 1 ELSE 0 END) AS listed_rows,
    SUM(CASE WHEN s.is_listed AND s.units = 0 THEN 1 ELSE 0 END) AS listed_zero_rows
FROM slice AS s
LEFT JOIN first_listed AS f ON f.id = s.id;
