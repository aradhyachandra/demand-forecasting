# Demand Forecasting & Inventory Planning: Project Plan

Author: Aradhya Chandra. Status: plan only (no code yet). Target: about 1 week of build sessions.

## 0. One-paragraph summary

Using Walmart's daily unit sales from the M5 forecasting dataset (US stores, 2011-2016, with prices
and calendar events), this project (1) prepares the data with SQL, (2) forecasts daily demand for a
manageable slice of products 28 days ahead, (3) compares simple baselines with a machine-learning
model in an honest rolling backtest, and (4) turns the forecasts into an inventory decision: how much
stock to hold for each product so shelves rarely run empty without tying up too much money. The
output is a consulting-style findings report, a small dashboard and reproducible code.

**Principle (same as my other projects):** every number in a document comes from saved results;
every business assumption sits in config with a reason; the final test period is touched once.

## 1. The business questions

1. How predictable is demand for these products, and which ones are hard (intermittent, promotional)?
2. How much better than simple rules ("same as last week", moving average) can a model forecast?
3. Do prices and events (holidays, SNAP benefit days) move demand, and by how much?
4. Given the forecasts and their errors, how much safety stock does each product need for a target
   service level, and what does that cost compared with today's naive approach?
5. Where is the money: which products drive most stockout risk or excess inventory?

## 2. Data

- Source: M5 Forecasting competition data (Walmart; University of Nicosia / MOFC), hosted on Kaggle:
  https://www.kaggle.com/c/m5-forecasting-accuracy . Files: sales_train_evaluation.csv (daily unit
  sales of ~30,490 item-store series, 1,941 days), calendar.csv (dates, events, SNAP days),
  sell_prices.csv (weekly prices per item-store).
- Download needs a Kaggle account and accepting the competition rules. TO VERIFY: the data's terms of
  use for a public portfolio repo. The raw files are never committed (data/raw/ gitignored); only code,
  small aggregated results and figures are.
- Scope to fit an 8 GB laptop: one state (California, 4 stores) and the FOODS category, or a similar
  slice decided in F1 (target: a few thousand series). Record the slice in config.

## 3. SQL layer (deliberate: shows SQL on real data)

- Use DuckDB (embedded, no server) to load the CSVs and do the data prep in SQL: reshape the wide
  sales table to long (item, store, date, units), join calendar and prices, build weekly aggregates,
  flag zero-sales runs and price changes. SQL files live in `sql/` and are run by Python.
- Tests compare SQL outputs with small hand-made tables.

## 4. Forecasting design

- Horizon: 28 days (the M5 horizon), daily, item-store level; also report weekly totals.
- **Rolling-origin backtest:** several forecast origins in the training years (for example the last 4
  x 28-day windows before the holdout) for model choice and tuning. **Final holdout:** the last 28 days
  of data, touched once after everything is frozen and committed.
- Baselines: seasonal naive (same weekday last week), 28-day moving average, simple exponential
  smoothing or Croston-type method for intermittent series.
- Main model: one global gradient-boosting model (LightGBM, pinned) across all series with lag,
  rolling-mean, calendar, event, SNAP and price features. No feature may use data after the forecast
  origin (tested, as in the churn project).
- Metrics: MAE and RMSE per series scaled by the naive error (RMSSE-style), bias, and a
  revenue-weighted error so high-value items count more. Report counts of series where the model beats
  each baseline, not only averages.
- Segment results: fast movers vs intermittent items; with vs without promotions/events.

## 5. Inventory layer (assumptions in config, never hidden)

- Policy: periodic review, order-up-to level = forecast demand over lead time + review period + safety
  stock; safety stock = z x forecast-error spread, with z from a target service level (for example 95%).
- Assumptions (config, each with a reason and a sensitivity range): lead time (days), review period,
  holding cost per unit-day (share of price), stockout cost per lost unit (share of price or margin).
- **Simulation on the holdout:** replay actual demand day by day under (a) the naive-forecast policy
  and (b) the model-forecast policy; measure fill rate, stockout days, average inventory and total
  cost. Report the trade-off curve (inventory vs service level) and how results change with the
  assumptions.
- Wording rule: simulated outcomes under stated assumptions, not observed savings.

## 6. Deliverables

1. Findings report (consulting style, ~8-10 pages): data and scope, demand patterns, forecast accuracy
   vs baselines (holdout), drivers (price/events), inventory policy results and trade-offs,
   limitations. All numbers generated from saved results.
2. Streamlit dashboard: pick a product, see history, forecast and interval; inventory simulation with
   service-level and cost sliders; portfolio view of where stockout risk and excess stock sit.
3. Repo: README with results table, how to reproduce (including the Kaggle download), licence notes.

## 7. Milestones (about one Claude Code session each; tests first)

| ID | What | Done when |
|---|---|---|
| F1 | Repo setup, Kaggle files loaded into DuckDB, SQL reshaping and joins, scope chosen | SQL outputs tested against hand-made tables; row counts and slice recorded; checksums of raw files stored |
| F2 | Demand exploration: seasonality, intermittency, price and event effects | Saved summary tables and figures; series classified (smooth / intermittent / lumpy) |
| F3 | Baselines + backtest harness | Rolling-origin backtest runs for all baselines; metrics saved per series and overall |
| F4 | LightGBM global model, feature tests, tuning on backtest only, then one holdout run | Leakage test passes; holdout run once from a committed, frozen setup; comparison table vs baselines |
| F5 | Inventory policy + simulation | Fill rate, stockouts, inventory and cost per policy and per assumption scenario |
| F6 | Report, dashboard, README, clean-clone check | Report numbers generated from results; dashboard runs; reproducibility check passes |

## 8. Working rules for any coding tool

- Read this plan and PROGRESS.md first; one milestone per session; tests first.
- No leakage: features for an origin use only data up to that origin. Never tune on the holdout.
- All assumptions and scope choices live in `config/` with a one-line reason.
- No invented numbers: report figures come from saved results.
- Keep the 8 GB RAM limit in mind: work on the chosen slice, prefer DuckDB/parquet, avoid loading
  everything into pandas at once.
- Mark uncertain things TO VERIFY in NOTES.md; end each session updating PROGRESS.md and a short
  plain-language note in docs/.

## 9. Risks

- Intermittent demand (many zero days) makes averages misleading: segment results and use scaled
  metrics.
- Memory: the full dataset is large; the slice and DuckDB keep it manageable.
- Cost assumptions dominate the money answer: show scenarios, not single numbers.
- Data terms: confirm before publishing anything beyond aggregated results.

## 10. README outline

What and why · Results (holdout) · Key findings · How it works (diagram) · Reproduce (Kaggle download)
· Data and terms · Assumptions · Limitations · Author.
