import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO
from datetime import datetime

# PDF export (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

st.set_page_config(page_title="Retail Sales Dashboard | Powell Ndlovu", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.small-note {opacity: 0.8; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("merged_train_store_open_days.csv", low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # numeric safety
    for c in ["Sales","Customers","Promo","DayOfWeek","SchoolHoliday","CompetitionDistance","Store"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Create Year/Month if missing
    if "Year" not in df.columns or df["Year"].isna().all():
        df["Year"] = df["Date"].dt.year
    if "Month" not in df.columns or df["Month"].isna().all():
        df["Month"] = df["Date"].dt.month

    # Ensure StoreType exists
    if "StoreType" not in df.columns:
        df["StoreType"] = "Unknown"

    return df.dropna(subset=["Date"])

df = load_data()

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.title("Filters")

min_date = df["Date"].min()
max_date = df["Date"].max()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)

store_types = sorted([x for x in df["StoreType"].dropna().unique()])
sel_store_types = st.sidebar.multiselect("StoreType", options=store_types, default=store_types)

stores = sorted(df["Store"].dropna().unique())
sel_store = st.sidebar.selectbox("Store (for store-level view)", options=[None] + stores)

promo_only = st.sidebar.checkbox("Promo days only", value=False)
school_holiday_only = st.sidebar.checkbox("School holiday days only", value=False)

granularity = st.sidebar.radio("Time granularity", options=["Daily", "Weekly", "Monthly"], index=0)
roll_days = st.sidebar.slider("Rolling window (days)", min_value=7, max_value=90, value=28, step=7)

top_n = st.sidebar.slider("Top N stores", min_value=5, max_value=50, value=20, step=5)

st.sidebar.markdown("---")
st.sidebar.caption("Portfolio by **Powell Andile Ndlovu**")

# ----------------------------
# Filter data
# ----------------------------
start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask = (df["Date"] >= start) & (df["Date"] <= end)
mask &= df["StoreType"].isin(sel_store_types)

if promo_only:
    mask &= (df["Promo"] == 1)
if school_holiday_only:
    mask &= (df["SchoolHoliday"] == 1)

dff = df.loc[mask].copy()

# ----------------------------
# Helpers
# ----------------------------
def to_period_index(d: pd.Series, g: str) -> pd.Series:
    if g == "Daily":
        return d.dt.floor("D")
    if g == "Weekly":
        return d.dt.to_period("W").dt.start_time
    return d.dt.to_period("M").dt.start_time

def safe_promo_uplift(frame: pd.DataFrame) -> float:
    m0 = frame[frame["Promo"] == 0]["Sales"].mean()
    m1 = frame[frame["Promo"] == 1]["Sales"].mean()
    if pd.isna(m0) or m0 <= 0 or pd.isna(m1):
        return np.nan
    return (m1 / m0 - 1) * 100

def compute_series(frame: pd.DataFrame, g: str) -> pd.DataFrame:
    tmp = frame.copy()
    tmp["Period"] = to_period_index(tmp["Date"], g)
    out = tmp.groupby("Period", as_index=False)[["Sales","Customers"]].sum().sort_values("Period")
    return out

def robust_promo_uplift_by_type(frame: pd.DataFrame) -> pd.DataFrame:
    p = frame.groupby(["StoreType","Promo"])["Sales"].mean().unstack()
    if 0 not in p.columns:
        p[0] = np.nan
    if 1 not in p.columns:
        p[1] = np.nan
    p = p.rename(columns={0:"NoPromo", 1:"Promo"})
    p["PromoUplift%"] = np.where(
        p["NoPromo"] > 0,
        (p["Promo"] / p["NoPromo"] - 1) * 100,
        np.nan
    )
    return p.reset_index().sort_values("PromoUplift%", ascending=False)

def executive_ai_insights(frame: pd.DataFrame, series: pd.DataFrame) -> dict:
    """
    'AI insights' here are automated, explainable insights (no external LLM).
    Returns dict with headline + bullets + risks + actions.
    """
    insights = {"headline":"", "bullets":[], "risks":[], "actions":[]}

    if len(frame) == 0:
        insights["headline"] = "No data under current filters."
        insights["bullets"].append("Widen the date range or relax filters to generate insights.")
        return insights

    total_sales = frame["Sales"].sum()
    avg_sales = series["Sales"].mean() if len(series) else np.nan
    promo_uplift = safe_promo_uplift(frame)

    # Trend (last vs previous)
    if len(series) >= 2:
        last = series["Sales"].iloc[-1]
        prev = series["Sales"].iloc[-2]
        pct = (last/prev - 1)*100 if prev else np.nan
    else:
        last, prev, pct = np.nan, np.nan, np.nan

    # Concentration (top stores share)
    by_store = frame.groupby("Store", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    top_share = (by_store["Sales"].head(10).sum()/total_sales*100) if total_sales else np.nan

    # Seasonality peak month
    peak_month = None
    if "Month" in frame.columns and frame["Month"].notna().any():
        m = frame.groupby("Month")["Sales"].mean()
        if len(m):
            peak_month = int(m.idxmax())

    insights["headline"] = "Executive AI Insights (automated)"

    insights["bullets"].append(f"Total sales in selection: {total_sales:,.0f}. Average per {granularity.lower()} period: {avg_sales:,.0f}." if not np.isnan(avg_sales) else f"Total sales in selection: {total_sales:,.0f}.")
    if not np.isnan(pct):
        direction = "up" if pct > 0 else "down"
        insights["bullets"].append(f"Most recent period is {direction} {abs(pct):.1f}% vs the previous period (signal of momentum).")

    if not np.isnan(promo_uplift):
        insights["bullets"].append(f"Promotion uplift (mean): {promo_uplift:.1f}%. Promotions appear beneficial under current filters.")
        if promo_uplift < 5:
            insights["risks"].append("Promo uplift is weak (<5%): promotions may be eroding margin without meaningful sales lift.")
            insights["actions"].append("Target promos to the store types/months where uplift is strongest; test smaller discounts or different promo mechanics.")
    else:
        insights["risks"].append("Promo uplift cannot be estimated (missing Promo or NoPromo days in filtered data).")
        insights["actions"].append("Relax Promo-only filter or widen the date range to include both promo and non-promo days.")

    if not np.isnan(top_share):
        insights["bullets"].append(f"Sales concentration: top 10 stores contribute ~{top_share:.1f}% of total sales.")
        if top_share > 40:
            insights["risks"].append("High concentration: performance depends heavily on a small number of stores.")
            insights["actions"].append("Create a watchlist for top stores, monitor dips early, and replicate what works to mid-tier stores.")

    if peak_month is not None:
        insights["bullets"].append(f"Peak month (avg sales): Month {peak_month} under current filters.")
        insights["actions"].append("Prepare stock/staffing ahead of peak periods; align promos with seasonal demand spikes.")

    # StoreType promo uplift leader
    try:
        pbt = robust_promo_uplift_by_type(frame)
        leader = pbt.dropna(subset=["PromoUplift%"]).head(1)
        if len(leader):
            stype = leader["StoreType"].iloc[0]
            uplift = leader["PromoUplift%"].iloc[0]
            insights["bullets"].append(f"Best promo response by StoreType: {stype} ({uplift:.1f}% uplift).")
            insights["actions"].append(f"Prioritize promo budget for StoreType '{stype}' where uplift is highest.")
    except Exception:
        pass

    return insights

def build_pdf_report(frame: pd.DataFrame, series: pd.DataFrame, insights: dict) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=1.6*cm, bottomMargin=1.6*cm
    )
    styles = getSampleStyleSheet()
    title = styles["Title"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    body.spaceAfter = 6

    small = ParagraphStyle("small", parent=body, fontSize=9, leading=11, textColor=colors.HexColor("#444444"))

    story = []
    story.append(Paragraph("Retail Sales Executive Summary", title))
    story.append(Paragraph("Portfolio report by <b>Powell Andile Ndlovu</b>", small))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", small))
    story.append(Spacer(1, 10))

    # KPIs
    total_sales = float(frame["Sales"].sum()) if len(frame) else 0.0
    stores_n = int(frame["Store"].nunique()) if len(frame) else 0
    records_n = int(len(frame))
    promo_uplift = safe_promo_uplift(frame)
    avg_period_sales = float(series["Sales"].mean()) if len(series) else np.nan

    story.append(Paragraph("Headline KPIs", h2))
    kpi_data = [
        ["Metric", "Value"],
        ["Records", f"{records_n:,}"],
        ["Stores", f"{stores_n:,}"],
        ["Total Sales", f"{total_sales:,.0f}"],
        [f"Avg Sales per {granularity} period", "-" if np.isnan(avg_period_sales) else f"{avg_period_sales:,.0f}"],
        ["Promo uplift (mean)", "-" if np.isnan(promo_uplift) else f"{promo_uplift:.1f}%"],
    ]
    table = Table(kpi_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Insights
    story.append(Paragraph("Executive AI Insights (automated)", h2))
    if insights.get("bullets"):
        bullets = "".join([f"<li>{b}</li>" for b in insights["bullets"]])
        story.append(Paragraph(f"<ul>{bullets}</ul>", body))
    if insights.get("risks"):
        story.append(Paragraph("Risks / Watch-outs", styles["Heading3"]))
        risks = "".join([f"<li>{r}</li>" for r in insights["risks"]])
        story.append(Paragraph(f"<ul>{risks}</ul>", body))
    if insights.get("actions"):
        story.append(Paragraph("Recommended actions", styles["Heading3"]))
        actions = "".join([f"<li>{a}</li>" for a in insights["actions"]])
        story.append(Paragraph(f"<ul>{actions}</ul>", body))

    story.append(Spacer(1, 10))

    # Top stores + top store types
    if len(frame):
        story.append(Paragraph("Top Stores (by total sales)", h2))
        top_stores = frame.groupby("Store", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(10)
        top_data = [["Store", "Sales"]] + [[int(r["Store"]), f'{r["Sales"]:,.0f}'] for _, r in top_stores.iterrows()]
        t2 = Table(top_data, hAlign="LEFT")
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(t2)
        story.append(Spacer(1, 10))

        story.append(Paragraph("Promo uplift by StoreType", h2))
        pbt = robust_promo_uplift_by_type(frame).head(8)
        pbt_data = [["StoreType", "NoPromo", "Promo", "Uplift%"]]
        for _, r in pbt.iterrows():
            pbt_data.append([str(r["StoreType"]), "-" if pd.isna(r["NoPromo"]) else f'{r["NoPromo"]:,.0f}',
                             "-" if pd.isna(r["Promo"]) else f'{r["Promo"]:,.0f}',
                             "-" if pd.isna(r["PromoUplift%"]) else f'{r["PromoUplift%"]:.1f}%'])
        t3 = Table(pbt_data, hAlign="LEFT")
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(t3)

    story.append(Spacer(1, 8))
    story.append(Paragraph("Note: This report is descriptive and reflects your selected filters.", small))

    doc.build(story)
    return buf.getvalue()

# ----------------------------
# Page header
# ----------------------------
st.title("Retail Sales & Store Performance Dashboard")
st.caption("Elite portfolio dashboard by Powell Andile Ndlovu | Interactive analytics + forecasting + executive reporting")

# Build aggregated series
series = compute_series(dff, granularity)
promo_uplift = safe_promo_uplift(dff)

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Records", f"{len(dff):,}")
k2.metric("Stores", f"{dff['Store'].nunique():,}")
k3.metric("Total Sales", f"{dff['Sales'].sum():,.0f}")
k4.metric(f"Avg Sales per {granularity} period", "-" if len(series)==0 else f"{series['Sales'].mean():,.0f}")
k5.metric("Promo uplift (mean)", "-" if np.isnan(promo_uplift) else f"{promo_uplift:.1f}%")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🧠 Executive AI Insights",
    "🏷️ Promo & Segments",
    "🏬 Store Explorer + Forecast",
    "🧾 Executive PDF Report"
])

# ============================
# TAB 1: Overview
# ============================
with tab1:
    st.subheader("Sales over time")

    if len(series) == 0:
        st.info("No data to plot under current filters.")
    else:
        # rolling in *rows* (granularity-aware approximation)
        denom = 1 if granularity == "Daily" else 7 if granularity == "Weekly" else 30
        roll_n = max(2, int(roll_days / denom))
        series["Rolling"] = series["Sales"].rolling(roll_n, min_periods=2).mean()

        fig = px.line(series, x="Period", y="Sales", title=f"Total Sales ({granularity})")
        fig.add_trace(go.Scatter(x=series["Period"], y=series["Rolling"], mode="lines", name=f"Rolling (~{roll_days}d)"))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"Top stores (by total sales) — Top {top_n}")
        top = dff.groupby("Store", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(top_n)
        fig2 = px.bar(top, x="Store", y="Sales", title=f"Top {top_n} Stores by Total Sales")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("Customers vs Sales (relationship)")
        tmp = dff.dropna(subset=["Customers","Sales"]).copy()
        if len(tmp) > 0:
            fig_sc = px.scatter(
                tmp.sample(min(len(tmp), 5000), random_state=7),
                x="Customers", y="Sales", color="Promo",
                title="Customers vs Sales (colored by Promo)",
                trendline="ols"
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Not enough data to plot Customers vs Sales after filtering.")

# ============================
# TAB 2: Executive AI Insights
# ============================
with tab2:
    st.subheader("Executive AI insights panel (automated + explainable)")
    st.caption("These insights are generated from your filtered data using transparent rules (no external AI calls).")

    insights = executive_ai_insights(dff, series)

    st.markdown("### Headline")
    st.success(insights.get("headline", "Executive AI Insights"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Key insights")
        if insights.get("bullets"):
            for b in insights["bullets"]:
                st.write("•", b)
        else:
            st.info("No insights available.")

    with c2:
        st.markdown("### Risks / watch-outs")
        if insights.get("risks"):
            for r in insights["risks"]:
                st.warning(r)
        else:
            st.write("• None detected under current filters.")

    st.markdown("### Recommended actions")
    if insights.get("actions"):
        for a in insights["actions"]:
            st.write("✅", a)
    else:
        st.write("✅ No actions suggested.")

    with st.expander("Show supporting tables"):
        st.write("Promo uplift by StoreType")
        st.dataframe(robust_promo_uplift_by_type(dff).head(25), use_container_width=True)

# ============================
# TAB 3: Promo & Segments
# ============================
with tab3:
    st.subheader("Promo effectiveness (distribution)")
    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(px.box(dff, x="Promo", y="Sales", points="outliers",
                               title="Sales distribution: Promo vs No Promo"), use_container_width=True)
    with c2:
        st.plotly_chart(px.box(dff, x="Promo", y="Customers", points="outliers",
                               title="Customers distribution: Promo vs No Promo"), use_container_width=True)

    st.subheader("Promo uplift by StoreType (robust)")
    promo_by_type = robust_promo_uplift_by_type(dff)
    st.plotly_chart(px.bar(promo_by_type, x="StoreType", y="PromoUplift%",
                           title="Promo Uplift (%) by StoreType",
                           hover_data=["NoPromo","Promo"]), use_container_width=True)
    st.dataframe(promo_by_type, use_container_width=True)

    st.subheader("Promo uplift heatmap: StoreType × Month")
    tmp = dff.copy()
    g = tmp.groupby(["StoreType","Month","Promo"])["Sales"].mean().unstack()
    if 0 not in g.columns:
        g[0] = np.nan
    if 1 not in g.columns:
        g[1] = np.nan
    g = g.rename(columns={0:"NoPromo", 1:"Promo"})
    g["Uplift%"] = np.where(g["NoPromo"] > 0, (g["Promo"]/g["NoPromo"] - 1) * 100, np.nan)
    g = g.reset_index()
    piv = g.pivot_table(index="StoreType", columns="Month", values="Uplift%", aggfunc="mean")
    hm = go.Figure(data=go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorbar=dict(title="Uplift%")))
    hm.update_layout(xaxis_title="Month", yaxis_title="StoreType")
    st.plotly_chart(hm, use_container_width=True)

# ============================
# TAB 4: Store Explorer + Forecast (Prophet)
# ============================
with tab4:
    st.subheader("Store explorer + Prophet forecasting")

    if sel_store is None:
        st.info("Pick a store in the sidebar to explore store-level trends and forecast.")
    else:
        sdf = dff[dff["Store"] == sel_store].copy()
        if len(sdf) == 0:
            st.warning("No rows match the filters for this store.")
        else:
            daily_s = sdf.groupby("Date", as_index=False)[["Sales","Customers"]].sum().sort_values("Date")
            daily_s["Rolling"] = daily_s["Sales"].rolling(28, min_periods=7).mean()

            fig = px.line(daily_s, x="Date", y="Sales", title=f"Store {sel_store}: Daily Sales")
            fig.add_trace(go.Scatter(x=daily_s["Date"], y=daily_s["Rolling"], mode="lines", name="Rolling(28d)"))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Forecast (Prophet)")
            st.caption("If Prophet isn't installed, you’ll see an install hint. Works great on Streamlit Cloud too.")

            horizon = st.slider("Forecast horizon (days)", min_value=14, max_value=180, value=60, step=7)
            seasonality_mode = st.selectbox("Seasonality mode", options=["additive", "multiplicative"], index=0)
            weekly = st.checkbox("Weekly seasonality", value=True)
            yearly = st.checkbox("Yearly seasonality", value=True)

            try:
                from prophet import Prophet  # type: ignore

                ds = daily_s[["Date","Sales"]].rename(columns={"Date":"ds","Sales":"y"})
                m = Prophet(
                    seasonality_mode=seasonality_mode,
                    weekly_seasonality=weekly,
                    yearly_seasonality=yearly,
                    daily_seasonality=False
                )
                m.fit(ds)
                future = m.make_future_dataframe(periods=horizon)
                fc = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]]

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=ds["ds"], y=ds["y"], mode="lines", name="Actual"))
                fig_fc.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", name="Forecast"))
                fig_fc.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"], mode="lines",
                                            name="Upper", line=dict(width=0), showlegend=False))
                fig_fc.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"], mode="lines",
                                            name="Lower", line=dict(width=0), fill="tonexty", showlegend=False))
                fig_fc.update_layout(title=f"Store {sel_store}: Prophet Forecast", xaxis_title="Date", yaxis_title="Sales")
                st.plotly_chart(fig_fc, use_container_width=True)

                st.dataframe(fc.tail(20), use_container_width=True)

            except Exception as e:
                st.error("Prophet is not available in this environment.")
                st.code(str(e))
                st.info("Fix: install prophet, e.g.  pip install prophet  (or add prophet to requirements.txt for Streamlit Cloud).")

# ============================
# TAB 5: Executive PDF Report Export
# ============================
with tab5:
    st.subheader("Interactive executive summary PDF export")
    st.caption("Generates a PDF report based on your current filters, including KPI table + AI insights + top stores + promo uplift table.")

    insights = executive_ai_insights(dff, series)

    c1, c2 = st.columns([1,2])
    with c1:
        if st.button("Generate PDF"):
            st.session_state["pdf_bytes"] = build_pdf_report(dff, series, insights)

    with c2:
        st.markdown("### What will be included")
        st.write("• Headline KPIs (records, stores, total sales, average sales, promo uplift)")
        st.write("• Executive AI insights (key insights, risks, recommended actions)")
        st.write("• Top stores table")
        st.write("• Promo uplift by StoreType table")

    if "pdf_bytes" in st.session_state:
        st.download_button(
            "⬇️ Download Executive Report (PDF)",
            data=st.session_state["pdf_bytes"],
            file_name="Executive_Report_Powell_Ndlovu.pdf",
            mime="application/pdf"
        )
        st.success("PDF generated — download it above.")

# ----------------------------
# Convenience download: filtered data
# ----------------------------
st.markdown("---")
st.download_button(
    "⬇️ Download filtered data (CSV)",
    data=dff.to_csv(index=False).encode("utf-8"),
    file_name="filtered_sales_data.csv",
    mime="text/csv"
)
