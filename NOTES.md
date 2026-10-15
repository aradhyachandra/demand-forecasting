# Notes and open questions

## Decisions in F1

- **Data terms (checked 2026-10-14):** the Kaggle rules allow non-commercial, academic and educational
  use and forbid redistributing the data. Raw and processed data stay out of git; only code and
  aggregated results are published.
- **Unlisted days:** days with no price for an item-store-week are treated as "not on sale", not as
  zero demand. Evidence from F1: no sales at all on such days, and no item disappears after its first
  listed week (all unlisted days are before launch). F2 onward should drop them from training and scoring.
- **Scratch database:** `run_f1` builds in a DuckDB file (`data/processed/work.duckdb`, gitignored),
  rebuilt every run. On the remote test machine it was moved with `FORECAST_WORK_DB` because that
  machine may not delete files inside the project folder; results do not depend on where it lives.

## Decisions in F2

- **Per-series tables stay private:** ADI/CV²/class per series and the list of price cuts go to
  `data/processed/` (gitignored), because per-item sales figures are close to the data the Kaggle
  terms say not to redistribute. Only aggregates are in `results/`.
- **Month profile uses full calendar years only (2012-2015)**, each month relative to its own year,
  because units per listed series-day fall over time as slower items are listed.
- **Price-cut summary kept as configured** even though the result looked odd (bigger cuts, lower
  sales). It was not re-tuned. The doc adds a week-by-week profile and the price-change-vs-sales check
  to explain it. The config reason "M5 price cuts are often short promotions" turned out wrong for
  this slice: in 92% of cuts the price is still down in week 2.
- **For F4:** use the last price known at the forecast origin, not the future weekly price (a price
  change in a future week reveals that the item sold that week).

## TO VERIFY

- **Holdout dates:** the final holdout is d_1914 to d_1941 (2016-04-25 to 2016-05-22), the last 28 days
  of `sales_train_evaluation.csv`. These were the competition's public-leaderboard labels. Confirm
  nobody tunes on them before the one F4 run.
- **Python on the Mac:** dependencies were tested on Python 3.10 on the remote machine; the pinned
  versions publish wheels for Python 3.13 (your Mac). Confirm `pip install -r requirements.txt` works there.
- **How M5 weekly prices are built:** in full weeks with no sales the price changed in 9 of 221,569
  weeks, against 1.9% of weeks with sales (`results/f2_demand_patterns.json`, price_change_vs_sales).
  That suggests prices are derived from transactions and carried forward. Check the M5 competition
  guide's wording on sell_prices ("average across seven days"?) before relying on it in the report.
- **Christmas closure:** Christmas shows about -100% (almost no units). Assumed to be store closure;
  confirm with the M5 documentation if quoted.
- **SNAP vs early-month effect:** the +10.5% SNAP lift cannot be separated from other early-month
  income (wages, other benefits). Do not call it a SNAP effect in the report without that caveat.
- **Event effects use 5-6 occurrences each:** treat single-event numbers as rough; only the large ones
  (Christmas, Thanksgiving, New Year, Labor Day, Presidents' Day) are consistent across years.
