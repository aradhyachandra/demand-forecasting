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

## Decisions in F3

- **Origin = last day of known history.** Windows are derived from `holdout.yaml` + horizon +
  n_origins, so changing the holdout moves them; a real-data test pins the current values.
- **RMSSE denominator on listed days**, over pairs of consecutive days both listed. M5 instead starts
  each series at its first non-zero sale. With the explicit listing flag, listed days are the cleaner
  start; it does change values slightly versus the M5 definition.
- **WRMSSE here is series-level only** (item-store), averaged over the 4 origins. The M5 WRMSSE also
  averages 11 aggregate levels (store, dept, total ...), so our numbers are not comparable with M5
  leaderboard scores.
- **Last known price = last full M5 week (Saturday-Friday) ended by the origin**, because the
  origin's own week also covers later days and (F2) its price can reveal later sales. Checked: weeks
  in the slice change on Saturday (d_1800 starts a new wm_yr_wk).
- **Zero-scale series** would be left out of RMSSE/WRMSSE but kept in MAE, RMSE and win counts (which
  use RMSE). On the real slice there were none at any origin, because every scored series has at
  least 56 listed days and some sales.
- **Win counts use RMSE** (same order as RMSSE within a series). Ties (within 1e-9) are mostly items
  with no recent sales where both rules forecast (almost) zero.
- **Smoothing constants fixed at 0.1** (SES, Croston/SBA, TSB) and not tuned; F4 may tune on the
  backtest windows only.

## Decisions in F4 (part A)

- **LightGBM on macOS needs OpenMP.** The 4.7.0 wheel failed to load (`Library not loaded:
  @rpath/libomp.dylib`); fixed with `brew install libomp` (23.1.3). Noted in requirements.txt.
- **Calendar from `calendar.csv`, including dates after the last training day.** It has no sales,
  and its events and SNAP days are published in advance. The model needs it for "days to the next
  event" near the end of a window and, in part B, for the holdout days. Reading it is not reading the
  holdout. The guard applies to sales, prices, listing and target days.
- **Origin-week listing treated as known at the origin.** Units and the listed flag of days <= T
  are features. The listed flag comes from the weekly price, which (F2) exists only for weeks with
  sales. So in an item's very first week the flag for days <= T could depend on a sale later in
  that week. This only touches launch weeks (no item is ever de-listed, F1), and a real store knows
  what was on its shelf. Prices of the unfinished week are never used (F3's last_full_week rule).
- **Christmas = closed.** It is dropped from the training targets, forecast as 0, and treated like
  an unlisted day in the history features. F3 eligibility (listed days >= 56) still counts it, so
  the model and the baselines are scored on the same series.
- **Number of trees is scored at checkpoints** (200/400/800) of one deterministic 800-tree run per
  setting and origin (`deterministic: true`, fixed seed). The first n trees are the n-tree model.
- **Only `event_name_1` / `event_type_1` are used.** The second event of a day is dropped. That
  happens on 5 training days (Easter / Orthodox Easter, Cinco de Mayo, Father's day).
- **"Train on everything up to d_1913" in part B** means every day up to d_1913 is available as
  history for the features. The training targets are the same recipe as in the backtest: one year of
  origins every 14 days, the last target day d_1913.
- **No sample weights** (revenue / scale) in the loss for this first model. A possible later
  improvement, to be decided on the backtest only.

- **OS memory numbers were added after the run.** `resources.os_measurement` in
  `results/f4_backtest.json` was parsed from the `/usr/bin/time -l` output of the run and merged
  into the JSON (`python -m forecast.run_f4` alone does not write it). Everything else in the JSON
  is written by the run. Re-training the chosen setting at d_1801 reproduced the saved forecasts
  exactly (max difference 0.0), so the run is deterministic.

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
- **F3 ranking among smoothing rules is close:** TSB, SES, SBA, moving average and Croston lie within
  about 0.01 WRMSSE. Do not call TSB clearly "the best" in the report; the robust result is smoothing
  vs naive rules. The ranking may also depend on the fixed alphas (0.1).
- **First backtest window under-forecasts the slice total** (TSB -8% to -14% per week from origin
  d_1801). Likely cause: the 28 days before 2016-01-02 include Christmas (stores closed) and the
  year-end weeks, which pull the level down. Check before quoting.
- **TSB with alpha_probability 0.1 ends up very close to SES** (nearly identical win counts). Plausible
  for these series but worth a sanity check on a few series plots before the report.
- **Launch-week listing flag (F4):** see the decision above. Check how many training and scored
  series-days fall in an item's first listed week before calling the features strictly leak-free in
  the report. Expected to be tiny: almost every item is listed before the backtest years.
- **Tweedie power 1.1 was not tuned** (fixed to keep the grid small). Poisson or 1.2-1.5 might
  score differently; any change must be decided on the backtest windows, before the holdout.
- **The chosen setting is at the edge of the grid (F4).** s1_small (31 leaves) at 200 trees is the
  smallest model and the fewest trees tried. Every setting got slightly worse with more trees. A
  smaller model (fewer trees or leaves) might do a little better. Extending the grid is allowed
  (backtest only) but would change config/model_final.yaml, so decide before the holdout run. The
  whole grid spans only 0.831-0.843 WRMSSE, all well below TSB (0.873), so the main conclusion does
  not depend on it.
- **Memory on the 8 GB Mac (F4):** maximum resident memory 1.85 GB, but macOS reports a peak memory
  footprint of 5.41 GB for the full 20-fit run (0 swaps). One table + one fit alone measured 1.28 GB
  resident / 2.9 GB footprint. The gap looks like LightGBM's per-thread buffers kept between fits.
  If the Mac swaps on a re-run, lower `num_threads` or fit each origin in a fresh process.
- **The model under-forecasts the slice total a bit more than TSB** (-4.3% vs -2.6% over the
  backtest windows), with a similar weekly error. F5 should size safety stock from the residuals,
  not assume an unbiased forecast.
- **Price features carry little gain (about 0.2% together).** That fits F2 (M5 prices only change
  with sales, and only the last full week's price may be used). Do not write "price does not matter"
  in the report; the model just cannot see future price cuts.
