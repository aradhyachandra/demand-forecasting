-- One row per training day (and department): total units and the number of series on sale, plus
-- the day's calendar facts. The slice is one state, so every row of a day has the same SNAP flag.
SELECT
    d_num, ANY_VALUE(date) AS date, ANY_VALUE(wday) AS wday, ANY_VALUE(weekday) AS weekday,
    ANY_VALUE(month) AS month, ANY_VALUE(year) AS year, MAX(snap) AS snap,
    ANY_VALUE(event_name_1) AS event_name_1, ANY_VALUE(event_type_1) AS event_type_1,
    ANY_VALUE(event_name_2) AS event_name_2, ANY_VALUE(event_type_2) AS event_type_2,
    {dept_col}
    SUM(units) AS units,
    COUNT(*) AS listed_series
FROM train_days
GROUP BY d_num {dept_group}
ORDER BY d_num {dept_group};
