"""Demand forecasting and inventory dashboard. Run with:  streamlit run app/streamlit_app.py

Reads results/ (committed aggregates) and data/processed/ (built locally from your own Kaggle download;
the M5 data may not be redistributed). Nothing is refitted. The inventory views select pre-computed F5
scenarios; the single-item replay calls the tested F5 simulation functions. Every inventory and money
figure is a SIMULATED OUTCOME UNDER STATED ASSUMPTIONS, not an observed saving.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))      # works without `pip install -e .`

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from forecast import dashboard as dash  # noqa: E402

PROCESSED = Path(os.environ.get("FORECAST_PROCESSED_DIR", ROOT / "data" / "processed"))
LABEL = {"seasonal_naive": "Seasonal naive", "tsb": "TSB", "lgbm": "LightGBM"}
COLOR = {"Seasonal naive": "#a3a29c", "TSB": "#eb6834", "LightGBM": "#2a78d6", "Actual (holdout)": "#e34948",
         "History (training)": "#6b6b66"}
ASSUMPTION_LABEL = {
    "lead_time_days": "Lead time (days)", "review_period_days": "Review period (days)",
    "starting_inventory_factor": "Starting stock (x order-up-to level)", "unmet_demand": "Unmet demand",
    "holding_cost_rate_per_year": "Holding cost (share of price per year)",
    "stockout_cost_share_of_price": "Lost-sale cost (share of price)"}
SS_LABEL = {"bias_adjusted": "mean + z x sd (main)", "z_sd_only": "z x sd only (comparison)"}

st.set_page_config(page_title="Demand forecasting and inventory", layout="wide")


@st.cache_data
def results():
    return dash.load_results(ROOT)


@st.cache_data
def index():
    return dash.series_index(PROCESSED / "m5_slice.parquet", PROCESSED / "f4_holdout_forecasts.parquet")


def money(x):
    return f"${x:,.0f}"


def scale(labels):
    return alt.Scale(domain=labels, range=[COLOR[x] for x in labels])


r = results()
f5, hold = r["f5"], r["f4_holdout"]
setup = f5["holdout"]["setup"]
origin, policies, central = setup["forecast_origin"], f5["policies"], f5["central"]
missing = dash.missing_processed(PROCESSED)

st.title("Demand forecasting and inventory")
st.caption(f"Walmart M5 data, California FOODS ({setup['n_series']:,} item-stores). One 28-day holdout "
           f"(d_{setup['forecast_days'][0]}..d_{setup['forecast_days'][1]}) forecast once from d_{origin} by the "
           "frozen model. Inventory figures are simulated outcomes under stated assumptions "
           "(config/inventory.yaml), not observed savings. Full write-up: docs/REPORT.md.")

product_tab, inventory_tab, portfolio_tab = st.tabs(["Product", "Inventory scenarios", "Portfolio"])


def data_message():
    st.info("This view needs the processed M5 data in data/processed/, which is not in the repository "
            "(the Kaggle terms forbid redistribution). To build it: download the M5 data from "
            "https://www.kaggle.com/c/m5-forecasting-accuracy/data (Kaggle account, accept the competition "
            "rules), put sales_train_evaluation.csv, calendar.csv and sell_prices.csv in data/raw/, then run "
            "python -m forecast.run_f1, run_f3, run_f4 and run_f4_holdout as described in README.md. "
            f"Missing: {', '.join(missing)}.")


# --- Product --------------------------------------------------------------------------------------
with product_tab:
    st.subheader("One item: history, holdout and forecasts")
    if missing:
        data_message()
    else:
        idx = index()
        c1, c2, c3, c4 = st.columns(4)
        store = c1.selectbox("Store", sorted(idx["store_id"].unique()), key="store")
        dept = c2.selectbox("Department", sorted(idx["dept_id"].unique()), key="dept")
        items = idx[(idx["store_id"] == store) & (idx["dept_id"] == dept)]
        item = c3.selectbox("Item", items["item_id"].tolist(), key="item")
        days = c4.select_slider("History shown (days)", [56, 112, 224, 364], value=112, key="history_days")
        row = items[items["item_id"] == item].iloc[0]
        sid = row["id"]
        st.session_state["sid"] = sid

        spread_origins = setup["spread_origins"]
        band_method = st.radio("Error band around", [LABEL[p] for p in policies], index=policies.index("lgbm"),
                               horizontal=True, key="band_method")
        band_key = next(p for p in policies if LABEL[p] == band_method)
        hist = dash.item_history(PROCESSED / "m5_slice.parquet", sid, origin, days)
        fc = dash.item_forecasts(PROCESSED / "f4_holdout_forecasts.parquet", sid, policies)
        lo, hi = dash.band_from_residuals(
            fc[band_key].to_numpy(), dash.backtest_residuals(PROCESSED / "f4_backtest_forecasts.parquet", sid,
                                                             band_key, spread_origins), 0.1, 0.9)
        fc["lo"], fc["hi"] = lo, hi

        lines = pd.concat([
            hist.assign(series="History (training)", value=hist["units"])[["d_num", "series", "value"]],
            fc.assign(series="Actual (holdout)", value=fc["actual"])[["d_num", "series", "value"]],
            *[fc.assign(series=LABEL[p], value=fc[p])[["d_num", "series", "value"]] for p in policies]])
        order = ["History (training)", "Actual (holdout)", *[LABEL[p] for p in policies]]
        band = alt.Chart(fc).mark_area(opacity=0.18, color=COLOR[band_method]).encode(
            x="d_num:Q", y=alt.Y("lo:Q", title="Units per day"), y2="hi:Q")
        chart = alt.Chart(lines).mark_line(strokeWidth=1.6).encode(
            x=alt.X("d_num:Q", title="Day (d_num)", scale=alt.Scale(zero=False), axis=alt.Axis(format="d")),
            y=alt.Y("value:Q", title="Units per day"),
            color=alt.Color("series:N", scale=scale(order), sort=order, title=None),
            tooltip=[alt.Tooltip("series:N"), alt.Tooltip("d_num:Q", title="Day"),
                     alt.Tooltip("value:Q", title="Units", format=".2f")])
        origin_rule = alt.Chart(pd.DataFrame({"x": [origin + 0.5]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart(band + chart + origin_rule, width="stretch")
        st.caption(f"{sid} ({row['demand_class']} at the origin). Gaps in the history = days before the item's first week with a sale. The shaded "
                   f"band is the {band_method} forecast plus the 10th to 90th percentile of this item's daily "
                   f"backtest errors (origins {', '.join(f'd_{o}' for o in spread_origins)}): a rough picture "
                   "of past error, not a calibrated interval. Holdout actuals are shown after the fact.")
        err = pd.DataFrame([{"Method": LABEL[p], "Forecast total (28 days)": fc[p].sum(),
                             "Actual total": fc["actual"].sum(),
                             "Mean absolute error per day": (fc[p] - fc["actual"]).abs().mean()} for p in policies])
        st.dataframe(err.style.format({"Forecast total (28 days)": "{:,.1f}", "Actual total": "{:,.0f}",
                                       "Mean absolute error per day": "{:.2f}"}), hide_index=True)

# --- Inventory --------------------------------------------------------------------------------------
with inventory_tab:
    st.subheader("Pre-computed F5 scenarios (whole slice, holdout)")
    st.write(f"Each item-store orders once every {central['review']} days up to: forecast demand until the next "
             f"delivery + safety stock (from backtest errors). Delivery after {central['lead']} days; unmet "
             "demand is lost. The controls pick among scenarios saved by `run_f5`; nothing is recomputed. "
             "A fixed target does not give every policy the same fill rate, so compare them at equal fill rate "
             "(chart).")
    a, b = st.columns(2)
    sl = a.select_slider("Target cycle service level", f5["service_level_sweep"], value=central["sl"],
                         format_func=lambda x: f"{x:.1%}", key="service_level")
    ss_method = b.radio("Safety stock", list(SS_LABEL), format_func=SS_LABEL.get, horizontal=True, key="ss_method")
    rows = dash.sweep_rows(r["sweep"], sl, ss_method, policies=policies)
    for col, (_, x) in zip(st.columns(len(policies)), rows.iterrows()):
        col.metric(f"{LABEL[x['policy']]}: fill rate", f"{x['fill_rate']:.1%}")
        col.metric("Average inventory (simulated)", money(x["avg_inv_money"]))
        col.metric(f"Total cost, {setup['measured_days']} days (simulated)", money(x["total_cost"]))
    curve = r["sweep"][(r["sweep"]["window"] == "holdout") & (r["sweep"]["ss_method"] == ss_method)].assign(
        Policy=lambda d: d["policy"].map(LABEL))
    labels = [LABEL[p] for p in policies]
    lines = alt.Chart(curve).mark_line(point=True).encode(
        x=alt.X("fill_rate:Q", title="Fill rate", axis=alt.Axis(format=".0%"), scale=alt.Scale(zero=False)),
        y=alt.Y("avg_inv_money:Q", title="Average inventory ($, simulated)"),
        color=alt.Color("Policy:N", scale=scale(labels), sort=labels),
        tooltip=["Policy:N", alt.Tooltip("service_level:Q", title="Target", format=".1%"),
                 alt.Tooltip("fill_rate:Q", title="Fill rate", format=".1%"),
                 alt.Tooltip("avg_inv_money:Q", title="Inventory", format="$,.0f")])
    chosen = alt.Chart(rows.assign(Policy=rows["policy"].map(LABEL))).mark_point(size=160, filled=False, strokeWidth=2,
                                                                                 color="#e34948").encode(
        x="fill_rate:Q", y="avg_inv_money:Q")
    st.altair_chart(lines + chosen, width="stretch")

    st.markdown("**Change one assumption** (the others stay central; 95% target, main safety stock)")
    opts = dash.scenario_options(f5)
    c1, c2 = st.columns(2)
    assumption = c1.selectbox("Assumption", list(opts), format_func=ASSUMPTION_LABEL.get, key="assumption",
                              on_change=lambda: st.session_state.pop("assumption_value", None))
    value = c2.select_slider("Value", opts[assumption], value=f5["assumptions"][assumption]["central"],
                             key="assumption_value")
    sc = dash.scenario_rows(r["sensitivity"], assumption, value, policies)
    check = dash.scenario_check(f5, assumption, value)
    st.dataframe(pd.DataFrame({
        "Policy": sc["policy"].map(LABEL), "Fill rate": sc["fill_rate"], "Cycle service level": sc["cycle_service_level"],
        "Avg inventory": sc["avg_inv_money"], "Holding cost": sc["holding_cost"], "Lost-sales cost": sc["lost_cost"],
        "Total cost": sc["total_cost"]}).style.format({
            "Fill rate": "{:.1%}", "Cycle service level": "{:.1%}", "Avg inventory": "${:,.0f}",
            "Holding cost": "${:,.0f}", "Lost-sales cost": "${:,.0f}", "Total cost": "${:,.0f}"}), hide_index=True)
    ef = next(e for e in check["equal_fill"] if abs(e["fill_rate"] - central["sl"]) < 1e-9)
    st.caption(f"{check['reviews']} reviews, {check['measured_days']} measured days. Cheapest by total cost: "
               f"{' > '.join(LABEL[p] for p in check['ranking_by_total_cost'])}. At an equal "
               f"{central['sl']:.0%} fill rate the model-based policy needs "
               f"{ef['lgbm_vs_seasonal_naive_pct']:+.1f}% inventory vs seasonal naive and "
               f"{ef['lgbm_vs_tsb_pct']:+.1f}% vs TSB.")

    st.markdown("**One item, replayed live** (the item chosen on the Product tab)")
    if missing:
        data_message()
    elif "sid" not in st.session_state:
        st.write("Open the Product tab and choose an item first.")
    else:
        sid = st.session_state["sid"]
        c1, c2, c3, c4 = st.columns(4)
        pol = c1.selectbox("Policy", policies, index=policies.index("lgbm"), format_func=LABEL.get, key="item_policy")
        isl = c2.select_slider("Target", f5["service_level_sweep"], value=central["sl"],
                               format_func=lambda x: f"{x:.1%}", key="item_sl")
        lead = c3.select_slider("Lead time (days)", opts["lead_time_days"], value=central["lead"], key="item_lead")
        review = c4.select_slider("Review period (days)", opts["review_period_days"], value=central["review"],
                                  key="item_review")
        fc = dash.item_forecasts(PROCESSED / "f4_holdout_forecasts.parquet", sid, [pol])
        errors = dash.item_backtest_errors(PROCESSED / "f4_backtest_forecasts.parquet", sid, pol, setup["spread_origins"])
        price = dash.last_known_price(PROCESSED / "m5_slice.parquet", sid, origin)
        if errors["residual"].count() < 2:
            st.write("This item has no backtest window to size a safety stock (left out in F5).")
        else:
            out = dash.item_simulation(fc[pol].to_numpy(), fc["actual"].to_numpy(), errors, pol, sid, price, lead,
                                       review, isl, central["ss_method"], setup["spread_origins"], origin,
                                       central["hold"], central["short"], central["start_factor"], central["unmet"])
            m = out["measures"]
            k = st.columns(4)
            k[0].metric("Fill rate (measured days)", "n/a" if m["fill_rate"] is None else f"{m['fill_rate']:.1%}")
            k[1].metric("Safety stock (units)", f"{out['safety_stock']:.1f}")
            k[2].metric("Average stock (units)", f"{m['avg_inv_units']:.1f}")
            k[3].metric("Total cost (simulated)", f"${m['total_cost']:,.2f}")
            d = out["days"].assign(day_d=lambda x: x["day"] + origin)
            stock = alt.Chart(d).mark_area(opacity=0.35, color=COLOR["LightGBM"], interpolate="step-after").encode(
                x=alt.X("day_d:Q", title="Day (d_num)", scale=alt.Scale(zero=False), axis=alt.Axis(format="d")),
                y=alt.Y("on_hand:Q", title="Units"),
                tooltip=["day_d:Q", "demand:Q", "served:Q", "lost:Q", "arrived:Q", "on_hand:Q"])
            dem = alt.Chart(d).mark_bar(color="#a3a29c", opacity=0.8, size=4).encode(x="day_d:Q", y="demand:Q")
            lost = alt.Chart(d[d["lost"] > 0]).mark_point(color="#eb6834", shape="triangle-down", size=70,
                                                          filled=True).encode(x="day_d:Q", y="lost:Q")
            st.altair_chart(stock + dem + lost, width="stretch")
            st.caption(f"Shaded: stock at the end of each day; bars: actual demand (holdout); orange: lost units. "
                       f"Reviews on days {', '.join(f'd_{origin + x}' for x in out['reviews'])}; price "
                       f"${price:.2f} (last known at d_{origin}); safety stock from {out['windows']} backtest "
                       "windows. Simulated, after the fact.")

# --- Portfolio ----------------------------------------------------------------------------------------
with portfolio_tab:
    st.subheader("Where lost sales and inventory sit (holdout, central scenario)")
    pol = st.radio("Policy", policies, index=policies.index("lgbm"), format_func=LABEL.get, horizontal=True,
                   key="portfolio_policy")
    mon = f5["holdout"]["money"][pol]
    k = st.columns(4)
    k[0].metric("Lost-sales cost (simulated)", money(mon["lost_cost"]))
    k[1].metric("Item-stores with any lost sale", f"{mon['series_with_any_lost_sale']:,} of {setup['n_simulated']:,}")
    k[2].metric(f"Top {f5['concentration_share']:.0%}: share of lost-sales cost", f"{mon['top_share_of_lost_cost']:.0%}")
    k[3].metric(f"Top {f5['concentration_share']:.0%}: share of inventory money",
                f"{mon['top_share_of_inventory_money']:.0%}")
    names = {"demand_share": "Demand units", "lost_cost_share": "Lost-sales cost", "inventory_share": "Inventory money"}
    for level, title in (("by_class", "By demand class"), ("by_dept", "By department")):
        t = dash.portfolio_shares(f5, pol, level).melt(id_vars="group", value_vars=list(names), var_name="measure",
                                                       value_name="share")
        t["measure"] = t["measure"].map(names)
        bars = alt.Chart(t).mark_bar().encode(
            y=alt.Y("group:N", title=None, sort=t["group"].unique().tolist()),
            yOffset=alt.YOffset("measure:N", sort=list(names.values())),
            x=alt.X("share:Q", title="Share of the slice total", axis=alt.Axis(format=".0%")),
            color=alt.Color("measure:N", sort=list(names.values()),
                            scale=alt.Scale(range=["#a3a29c", "#e34948", "#2a78d6"]), title=None),
            tooltip=["group:N", "measure:N", alt.Tooltip("share:Q", format=".1%")])
        st.markdown(f"**{title}**")
        st.altair_chart(bars, width="stretch")
    g = r["by_group"][(r["by_group"]["ss_method"] == central["ss_method"]) & (r["by_group"]["policy"] == pol)]
    st.dataframe(pd.DataFrame({
        "Group type": g["group_type"], "Group": g["group"], "Item-stores": g["n_series"], "Fill rate": g["fill_rate"],
        "Avg inventory": g["avg_inv_money"], "Lost-sales cost": g["lost_cost"], "Total cost": g["total_cost"]})
        .style.format({"Fill rate": "{:.1%}", "Avg inventory": "${:,.0f}", "Lost-sales cost": "${:,.0f}",
                       "Total cost": "${:,.0f}", "Item-stores": "{:,}"}), hide_index=True, width="stretch")
    st.caption("Aggregates from results/f5_inventory.json and results/f5_main_by_group.csv. Sales are not demand: "
               "on days the real shelf was empty, demand was higher than recorded.")
