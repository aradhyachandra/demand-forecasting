# Notes and open questions

## Decisions in F1

- **Data terms (checked 2026-10-14):** the Kaggle rules allow non-commercial, academic and educational
  use and forbid redistributing the data. Raw and processed data stay out of git; only code and
  aggregated results are published.
- **Unlisted days:** days with no price for an item-store-week are treated as "not on sale", not as
  zero demand. Evidence from F1: no sales at all on such days, and no item disappears after its first
  listed week (all unlisted days are before launch). F2 onward should drop them from training and scoring.
  *F6 rewording: "launch" in M5 is in effect an item's first week with a sale. Of the 3,456 series whose
  prices start after the first week of the data, 3,440 sell in their first priced week
  (`results/f6_first_sale_week.json`, `python -m forecast.first_sale_week`), consistent with the M5
  guide's "not sold during the examined week" (see TO VERIFY, weekly prices). So the excluded days are
  "days before an item's first week with a sale"; the data cannot distinguish "not yet on sale" from
  "on the shelf but unsold" on them. Linked: the F4 launch-week listing-flag note below.*
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

## Decisions in F4 (part B)

- **The comparison baseline was fixed before the holdout run** (TSB, the best in the backtest). On
  the holdout SBA scores slightly better than TSB. The write-up reports both, but TSB stays the
  reference, so the baseline is not picked after seeing the holdout.
- **The holdout write-up is rendered from saved results only** (`render_f4_holdout`). It loads no
  data and trains nothing; a test checks that it does not touch `model.load_data` / `model.fit`
  and leaves `results/f4_holdout.json` unchanged.
- **The part A real-data check no longer asserts that the holdout result is missing.** A new check
  ties `results/f4_holdout.json` to commit 1a31ed9 and the frozen `model_final.yaml`.

## Decisions in F5

- **Safety stock from summed-window backtest errors.** Error = actual - forecast summed over every
  window of protection-period length (lead + review days) inside each 28-day backtest window,
  overlapping windows pooled over origins, sd with ddof 1. Holdout: origins d_1801..d_1885; the
  pre-holdout check at d_1885: d_1801..d_1857 only. `spread_from_backtest` refuses any origin whose
  error days reach the forecast days, and the holdout forecasts are loaded without their residuals.
- **Bias handled in the main method** (safety stock = mean + z x sd, as suggested); z x sd only is the
  comparison. Decided before the simulation and kept as the headline even though the comparison did
  better (see TO VERIFY).
- **End of horizon: only reviews whose protection period fits.** One forecast of 28 days, so with lead
  2 and review 7 the reviews are days 1, 8, 15 and the measured days 3-23 (d_1916..d_1936). Days
  before the first possible delivery are a warm-up. Capping the last forecast sum at day 28 was
  rejected: it would need a shorter-window spread and would run stock down artificially at the end.
  The number of reviews and measured days changes with lead time and review period (shown in the
  sensitivity table); review 14 leaves a single review.
- **Forecasts are not updated at later reviews** (one origin per window). Stated as a limitation.
- **Start stock = 1.0 x S of the first review, nothing on order** (0.5 and 1.5 as sensitivity).
- **Orders and start stock rounded up to whole units**; no case packs, shelf capacity or shelf life.
- **Units valued at the last known price at the origin** (F3 rule), so no holdout price is used.
- **Item-stores without any backtest window are left out** (2 on the holdout, 3 at d_1885), counted
  in the results.
- **Comparison fill rates fixed in config before the run** (90/95/98%). 98% is not reached by the
  main-method curves on the holdout within the sweep (up to 99.5% target), so it is reported as n/a.
- **Ranking is reported two ways:** total cost at the same target (changes in 4 of 20 scenarios,
  because a fixed target gives the naive policy, with wider errors, more stock and a higher fill
  rate) and inventory at equal fill rate (model lowest in all 20). The second is the like-for-like one.

## Decisions in F6

- **F6 presents and reproduces; no new modelling.** `docs/REPORT.md` and `README.md` are rendered by
  `run_f6` from `results/*.json` through templates in `docs/templates/`; an unknown placeholder fails.
  Differences between saved numbers (e.g. "5.5% less") are computed in the renderer. Tests check the
  wording rules (no "savings", no "caused") and that the summary has five bullets.
- **One new count, added through code at the author's request:** `results/f6_first_sale_week.json`
  (`python -m forecast.first_sale_week`, tested on hand-made tables): 3,440 of the 3,456 series whose
  prices start after week 1 sell in their first priced week. Used to word the excluded days as "days
  before an item's first week with a sale".
- **M5 guide read for the TO VERIFY items** (downloaded with the author's OK into the session
  scratchpad only, not the repo); see the two price / Christmas items below.
- **Dashboard (`app/streamlit_app.py`, data functions in `src/forecast/dashboard.py`)** reads only
  `results/` and `data/processed/` (path overridable with `FORECAST_PROCESSED_DIR`). Inventory controls
  select pre-computed F5 rows; the single-item replay calls the F5 functions, and a real-data test
  checks it equals the full `run_f5` simulation item by item (25 random items x 2 policies). The
  product view's band is the 10th-90th percentile of the item's daily backtest errors added to the
  forecast: a picture of past error, not a calibrated interval. Without `data/processed/` it shows how
  to download from Kaggle; the inventory and portfolio views still work.
- **streamlit 1.66.0 and altair 6.3.0 pinned**; `pip install --dry-run` showed no change to any
  existing pin (pyarrow stays 25.0.1).
- **Clean-clone check, option (b):** the frozen holdout code was re-run at 1a31ed9 in a separate
  `git clone` inside the temp folder (not a worktree of the real repo); its JSON matched the committed
  one except run-time fields, and its forecasts parquet was byte-identical. Nothing was changed because
  of it. README step 4 documents the same procedure. Details: `docs/F6_reproducibility.md`.
- **Not run in the working copy:** `forecast.run_f4_holdout`.
- **README screenshot (`docs/figures/f6_dashboard_portfolio.png`, 2026-10-16):** the Portfolio view
  only, captured with headless Chrome (Playwright, from a throwaway scratch venv, not a project
  dependency) while the app ran with an empty processed-data folder, so no per-item sales could appear.
  Every number in it comes from `results/` through the app. It is a static image, not regenerated by
  the pipeline; retake it if the F5 results change. The same capture showed the dashboard header
  saying "not observed savings"; changed to "not observed results", and a test now checks the app text
  for the wording rules.

## TO VERIFY

- **Holdout dates:** the final holdout is d_1914 to d_1941 (2016-04-25 to 2016-05-22), the last 28 days
  of `sales_train_evaluation.csv`. These were the competition's public-leaderboard labels. Confirm
  nobody tunes on them before the one F4 run.
  *Done 2026-10-15: scored once by `run_f4_holdout` from commit 1a31ed9 (result committed in c38584c);
  nothing was tuned on it.*
- **Python on the Mac:** dependencies were tested on Python 3.10 on the remote machine; the pinned
  versions publish wheels for Python 3.13 (your Mac). Confirm `pip install -r requirements.txt` works there.
  *F6: resolved. F4-F6 ran on Python 3.13.5 on the Mac, and the clean-clone check installed the pinned
  requirements into a fresh 3.13.5 venv with `pip check` clean (docs/F6_reproducibility.md).*
- **How M5 weekly prices are built:** in full weeks with no sales the price changed in 9 of 221,569
  weeks, against 1.9% of weeks with sales (`results/f2_demand_patterns.json`, price_change_vs_sales).
  That suggests prices are derived from transactions and carried forward. Check the M5 competition
  guide's wording on sell_prices ("average across seven days"?) before relying on it in the report.
  *F6: resolved 2026-10-15. Source: *The M5 Competition: Competitors' Guide* (University of Nicosia /
  MOFC, final version 10 March 2020), File 2 "sell_prices.csv": "The price is provided per week
  (average across seven days). If not available, this means that the product was not sold during the
  examined week." Retrieved from the Internet Archive copy (snapshot 2022-01-06) of the official link
  https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx
  (the live link now redirects to a 404 on unic.ac.cy); file sha256 8db2d62bf7e8bdcf..., kept in the
  session scratchpad only, not in the repo. A weekly average of transaction prices fits the data: a
  new price appears only in weeks with sales. Note the guide's "not sold that week" does not hold
  literally after launch: in this slice weeks without sales still carry a (carried-forward) price, and
  missing prices occur only before an item's first priced week (F1), which in practice is its first
  week with a sale (3,440 of 3,456 later-starting series, `results/f6_first_sale_week.json`). The
  report cites the guide in a footnote and words the exclusion as "days before an item's first week
  with a sale", noting it cannot tell "not yet on sale" from "on the shelf but unsold".*
- **Christmas closure:** Christmas shows about -100% (almost no units). Assumed to be store closure;
  confirm with the M5 documentation if quoted.
  *F6: still open. The Competitors' Guide (above) was checked and does not mention Christmas or store
  closures; its calendar section only lists the event fields. The report says "almost no sales,
  consistent with stores being closed ... the M5 guide does not mention closures, so this is an
  inference from the data".*
- **SNAP vs early-month effect:** the +10.5% SNAP lift cannot be separated from other early-month
  income (wages, other benefits). Do not call it a SNAP effect in the report without that caveat.
  *F6: resolved by wording. The report calls it an early-month association that cannot be separated
  from other income, not a measured SNAP effect.*
- **Event effects use 5-6 occurrences each:** treat single-event numbers as rough; only the large ones
  (Christmas, Thanksgiving, New Year, Labor Day, Presidents' Day) are consistent across years.
  *F6: resolved by wording. The report shows the number of years and the lowest-to-highest year next
  to every event figure and calls single numbers rough.*
- **F3 ranking among smoothing rules is close:** TSB, SES, SBA, moving average and Croston lie within
  about 0.01 WRMSSE. Do not call TSB clearly "the best" in the report; the robust result is smoothing
  vs naive rules. The ranking may also depend on the fixed alphas (0.1).
  *F6: resolved by wording. The report says the robust result is "smoothing beats naive rules" and
  shows SBA beating TSB on the holdout; TSB is only the reference fixed in advance.*
- **First backtest window under-forecasts the slice total** (TSB -8% to -14% per week from origin
  d_1801). Likely cause: the 28 days before 2016-01-02 include Christmas (stores closed) and the
  year-end weeks, which pull the level down. Check before quoting.
  *F6: still open; not quoted in the report.*
- **TSB with alpha_probability 0.1 ends up very close to SES** (nearly identical win counts). Plausible
  for these series but worth a sanity check on a few series plots before the report.
  *F6: still open; not quoted in the report.*
- **Launch-week listing flag (F4):** see the decision above. Check how many training and scored
  series-days fall in an item's first listed week before calling the features strictly leak-free in
  the report. Expected to be tiny: almost every item is listed before the backtest years.
  *F6: partly answered. The first priced week is in effect the first week with a sale (3,440 of the
  3,456 series that start after week 1, `results/f6_first_sale_week.json`), so in an item's first week
  the listed flag for days <= T does carry a hint of a sale later that week. How many training and
  scored series-days fall in such weeks is still not counted. The report says a test checks that no
  feature moves when later values change, mentions this first-week hint with a pointer here, and does
  not use the words "strictly leak-free". Linked: F1 decision on unlisted days above.*
- **Tweedie power 1.1 was not tuned** (fixed to keep the grid small). Poisson or 1.2-1.5 might
  score differently; any change must be decided on the backtest windows, before the holdout.
  *F6: still open; listed as a limitation ("Tweedie power 1.1 not tuned").*
- **The chosen setting is at the edge of the grid (F4).** s1_small (31 leaves) at 200 trees is the
  smallest model and the fewest trees tried. Every setting got slightly worse with more trees. A
  smaller model (fewer trees or leaves) might do a little better. Extending the grid is allowed
  (backtest only) but would change config/model_final.yaml, so decide before the holdout run. The
  whole grid spans only 0.831-0.843 WRMSSE, all well below TSB (0.873), so the main conclusion does
  not depend on it.
  *F6: still open; in the report's limitations with the grid range.*
- **Memory on the 8 GB Mac (F4):** maximum resident memory 1.85 GB, but macOS reports a peak memory
  footprint of 5.41 GB for the full 20-fit run (0 swaps). One table + one fit alone measured 1.28 GB
  resident / 2.9 GB footprint. The gap looks like LightGBM's per-thread buffers kept between fits.
  If the Mac swaps on a re-run, lower `num_threads` or fit each origin in a fresh process.
  *F6: the clean-clone re-run of run_f4 peaked at 1.68 GB resident / 3.72 GB macOS footprint on the
  8 GB Mac and finished (`results/f6_clean_clone_check.json`); no swap problem was seen.*
- **The model under-forecasts the slice total a bit more than TSB** (-4.3% vs -2.6% over the
  backtest windows), with a similar weekly error. F5 should size safety stock from the residuals,
  not assume an unbiased forecast.
  *F6: confirmed on the holdout (-4.9% on the slice total, below the actual in 4 of 4 weeks); carried
  into the report as a finding and a limitation.*
- **Price features carry little gain (about 0.2% together).** That fits F2 (M5 prices only change
  with sales, and only the last full week's price may be used). Do not write "price does not matter"
  in the report; the model just cannot see future price cuts.
  *F6: resolved by wording ("says the model cannot see future price changes, not that price does not
  matter").*
- **Cost and lead-time assumptions (F5) are judgement calls:** 25% of price per year holding cost,
  lost unit = 25% of price (grocery gross-margin proxy), lead time 2 days, weekly review. None are in
  the data; check against published grocery margins / supply-chain sources before quoting money
  figures in the report, and always quote them with the sensitivity range.
  *F6: still open (no outside source checked). The report calls every money figure a simulated
  outcome under stated assumptions, lists the reasons and ranges in an appendix, and leads with the
  equal-fill-rate comparison, which does not depend on the cost assumptions.*
- **Achieved cycle service level is about 3 points below target (F5)** for every policy (e.g. 92% at
  a 95% target). Likely causes: skewed (non-normal) summed errors of intermittent items and spreads
  estimated from only 3-4 backtest windows. An empirical quantile of the summed errors instead of the
  normal z is a possible improvement, to be decided on the backtest windows only.
  *F6: still open; in the report as a limitation and recommendation 3.*
- **Bias-adjusted safety stock did worse than z x sd only (F5)**, on the holdout and on the
  pre-holdout check: it needs more stock for the same fill rate. Probably the per-series mean error
  from a few windows is noise. A pooled (class- or slice-level) or shrunk bias term could be tested,
  deciding on the d_1885 check (pre-holdout) only, not on the holdout.
  *F6: still open; in the report as recommendation 2 (pooled bias correction, chosen on the backtest).*
- **Lost-sales concentration (F5):** the top 10% of item-stores carry over 90% of lost-sales cost,
  partly by construction (fewer than 20% of item-stores lose any sale in 21 days). Check that this is
  not driven by a few items with single large-order days before calling it a finding.
  *F6: still open. The report says the concentration is partly built in and that the single
  large-order-day question was not checked.*
- **Sales are not demand (F5):** M5 records sales; on days the real shelf was empty, demand was
  higher. The simulation treats sales as demand.
  *F6: in the report as a limitation ("Sales are not demand").*
