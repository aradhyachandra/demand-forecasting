-- Basic checks on the joined slice; every value ends up in results/f1_data_report.json.
SELECT
    COUNT(*)                                             AS n_rows,
    COUNT(DISTINCT id)                                   AS n_series,
    COUNT(DISTINCT d)                                    AS n_days,
    MIN(date)                                            AS first_date,
    MAX(date)                                            AS last_date,
    COUNT(*) - COUNT(DISTINCT (id, d))                   AS duplicate_series_days,
    SUM(CASE WHEN units IS NULL THEN 1 ELSE 0 END)       AS null_units,
    SUM(CASE WHEN units < 0 THEN 1 ELSE 0 END)           AS negative_units,
    SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END)        AS rows_without_date,
    SUM(CASE WHEN units = 0 THEN 1 ELSE 0 END)           AS zero_unit_rows,
    SUM(CASE WHEN NOT is_listed THEN 1 ELSE 0 END)       AS unlisted_rows,
    SUM(CASE WHEN NOT is_listed AND units > 0 THEN 1 ELSE 0 END) AS sales_while_unlisted,
    SUM(units)                                           AS total_units,
    SUM(CASE WHEN snap = 1 THEN 1 ELSE 0 END)            AS snap_rows,
    SUM(CASE WHEN event_name_1 IS NOT NULL THEN 1 ELSE 0 END) AS event_rows
FROM slice;
