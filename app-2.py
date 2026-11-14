import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Retail Sales Analyzer", page_icon="📈", layout="wide")

MODEL_OPTIONS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]
JUDGE_DEFAULT = "llama-3.3-70b-versatile"

ANALYST_PROMPT = """You are a senior retail business analyst advising a store owner who has
no analytics background. You are given pre-computed statistics from their monthly sales data.
Use ONLY these numbers; never invent figures. If something can't be concluded, say so.

Write in plain, simple language using Markdown with exactly these sections:
## Dataset Summary
## Sales Trends
## Key Insights
## Risks & Weak Spots
## Recommended Actions
(For Recommended Actions give 5 specific, prioritized actions, each with expected impact
and how to measure it.)"""

JUDGE_PROMPT = """You are an impartial senior analyst. You receive the source statistics and
several AI-generated analyses of the same retail sales data. Do the following in Markdown:

## Comparison Table
A table comparing the analyses on: accuracy vs. the statistics, depth of insight,
actionability, clarity. Rate each 1-5.
## Agreements
Points all analyses agree on.
## Differences & Errors
Where they differ, and any claims NOT supported by the statistics.
## Final Recommended Action Plan
A consolidated, prioritized plan (top 5-7 actions) to improve sales, with a suggested
timeline (30 / 60 / 90 days) and KPIs to track. Use only numbers present in the statistics."""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_api_key() -> str:
    typed = st.session_state.get("api_key_input", "").strip()
    if typed:
        return typed
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
    categories = {"Electronics": 220, "Clothing": 160, "Groceries": 300, "Home & Kitchen": 140, "Toys": 90}
    regions = ["North", "South", "East", "West"]
    seasonal = {1: 0.85, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.0, 6: 0.95,
                7: 0.9, 8: 0.95, 9: 1.0, 10: 1.1, 11: 1.35, 12: 1.5}
    rows = []
    for i, m in enumerate(months):
        growth = 1 + 0.012 * i
        for cat, base in categories.items():
            for reg in regions:
                units = max(1, int(rng.normal(base / 4, base / 20) * seasonal[m.month] * growth))
                if cat == "Toys" and m.year == 2025:
                    units = int(units * 0.7)  # planted decline for demo
                price = {"Electronics": 180, "Clothing": 45, "Groceries": 12,
                         "Home & Kitchen": 60, "Toys": 25}[cat]
                rows.append([m, cat, reg, units, round(units * price * rng.uniform(0.95, 1.05), 2)])
    return pd.DataFrame(rows, columns=["Date", "Category", "Region", "Units", "Revenue"])


def load_file(upload) -> pd.DataFrame:
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload)
    return pd.read_excel(upload)


def pct(a, b):
    return None if b in (0, None) or pd.isna(b) else round((a - b) / b * 100, 1)


def compute_stats(df, date_col, rev_col, qty_col, cat_col, reg_col, prod_col):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[rev_col] = pd.to_numeric(d[rev_col], errors="coerce")
    d = d.dropna(subset=[date_col, rev_col])
    d["Month"] = d[date_col].dt.to_period("M").dt.to_timestamp()

    monthly = d.groupby("Month")[rev_col].sum().sort_index()
    stats = {"monthly": monthly, "clean_df": d}
    lines = []

    lines.append(f"Rows analysed: {len(d)} | Period: {monthly.index.min():%b %Y} to {monthly.index.max():%b %Y} "
                 f"({len(monthly)} months)")
    lines.append(f"Total revenue: {monthly.sum():,.2f}")
    lines.append(f"Average monthly revenue: {monthly.mean():,.2f}")
    lines.append(f"Best month: {monthly.idxmax():%b %Y} ({monthly.max():,.2f})")
    lines.append(f"Worst month: {monthly.idxmin():%b %Y} ({monthly.min():,.2f})")

    if len(monthly) >= 2:
        mom = monthly.pct_change().dropna() * 100
        lines.append(f"Latest month vs previous month: {mom.iloc[-1]:+.1f}%")
        lines.append(f"Average month-over-month change: {mom.mean():+.1f}%")
    if len(monthly) >= 6:
        last3, prev3 = monthly.iloc[-3:].sum(), monthly.iloc[-6:-3].sum()
        lines.append(f"Last 3 months vs previous 3 months: {pct(last3, prev3):+.1f}%")
    if len(monthly) >= 13:
        yoy = pct(monthly.iloc[-1], monthly.iloc[-13])
        lines.append(f"Latest month vs same month last year: {yoy:+.1f}%")

    # Seasonality
    by_moy = d.groupby(d[date_col].dt.month)[rev_col].mean()
    if len(by_moy) >= 6:
        lines.append(f"Strongest calendar month (avg): {int(by_moy.idxmax())} | weakest: {int(by_moy.idxmin())}")

    # Yearly totals
    yearly = d.groupby(d[date_col].dt.year)[rev_col].sum()
    if len(yearly) > 1:
        lines.append("Revenue by year: " + ", ".join(f"{int(y)}: {v:,.0f}" for y, v in yearly.items()))

    def breakdown(col, label, top=6):
        g = d.groupby(col)[rev_col].sum().sort_values(ascending=False)
        share = (g / g.sum() * 100).round(1)
        lines.append(f"\nRevenue by {label} (top {min(top, len(g))}): " +
                     "; ".join(f"{k}: {v:,.0f} ({share[k]}%)" for k, v in g.head(top).items()))
        if len(g) > top:
            lines.append(f"Bottom {label}s: " + "; ".join(f"{k}: {v:,.0f}" for k, v in g.tail(3).items()))
        # growth of each segment: last 3 months vs previous 3
        months = sorted(d["Month"].unique())
        if len(months) >= 6:
            l3, p3 = months[-3:], months[-6:-3]
            a = d[d["Month"].isin(l3)].groupby(col)[rev_col].sum()
            b = d[d["Month"].isin(p3)].groupby(col)[rev_col].sum()
            ch = ((a - b) / b * 100).replace([np.inf, -np.inf], np.nan).dropna().sort_values()
            if len(ch):
                lines.append(f"{label} momentum (last 3 vs prior 3 months): " +
                             "; ".join(f"{k}: {v:+.1f}%" for k, v in ch.items()))
        return g

    if cat_col and cat_col != "(none)":
        stats["by_cat"] = breakdown(cat_col, cat_col)
    if reg_col and reg_col != "(none)":
        stats["by_reg"] = breakdown(reg_col, reg_col)
    if prod_col and prod_col != "(none)":
        stats["by_prod"] = breakdown(prod_col, prod_col, top=8)
    if qty_col and qty_col != "(none)":
        q = pd.to_numeric(d[qty_col], errors="coerce").sum()
        lines.append(f"\nTotal units sold: {q:,.0f} | Average revenue per unit: {monthly.sum() / q:,.2f}" if q else "")

    stats["text"] = "\n".join(l for l in lines if l)
    return stats


def call_llm(client, model, system, user, temperature=0.3, max_tokens=1800):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.text_input("Groq API key", type="password", key="api_key_input",
                  help="Or set GROQ_API_KEY as env variable / in .streamlit/secrets.toml")
    models = st.multiselect("Models to compare (2-3)", MODEL_OPTIONS, default=MODEL_OPTIONS[:2])
    judge_model = st.selectbox("Model for comparison & final plan", MODEL_OPTIONS,
                               index=MODEL_OPTIONS.index(JUDGE_DEFAULT))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    st.caption("Each selected model analyses the same statistics independently; "
               "a judge model then compares them and builds the final action plan.")

st.title("📈 AI Retail Sales Analyzer")
st.caption("Upload monthly sales data → get AI summaries, trends, insights, a comparison of AI analyses, and an action plan.")

# ----------------------------------------------------------------------------
# Step 1: data
# ----------------------------------------------------------------------------
st.header("1️⃣ Load data")
src = st.radio("Data source", ["Upload CSV / Excel", "Use sample data"], horizontal=True)
df = None
if src == "Upload CSV / Excel":
    up = st.file_uploader("Upload sales file", type=["csv", "xlsx", "xls"])
    if up:
        try:
            df = load_file(up)
        except Exception as e:
            st.error(f"Could not read file: {e}")
else:
    df = sample_data()
    st.info("Using generated demo data (2024–2025, 5 categories, 4 regions).")

if df is None:
    st.stop()

st.dataframe(df.head(20), use_container_width=True)
st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")

# ----------------------------------------------------------------------------
# Step 2: column mapping
# ----------------------------------------------------------------------------
st.header("2️⃣ Map your columns")
cols = list(df.columns)
opt = ["(none)"] + cols


def guess(keys, options, default=0):
    for i, c in enumerate(options):
        if any(k in str(c).lower() for k in keys):
            return i
    return default


c1, c2, c3 = st.columns(3)
date_col = c1.selectbox("Date / month column *", cols, index=guess(["date", "month", "period"], cols))
rev_col = c2.selectbox("Revenue / sales column *", cols, index=guess(["revenue", "sales", "amount", "total"], cols))
qty_col = c3.selectbox("Quantity column", opt, index=guess(["unit", "qty", "quantity"], opt, 0))
c4, c5, c6 = st.columns(3)
cat_col = c4.selectbox("Category column", opt, index=guess(["categor"], opt, 0))
reg_col = c5.selectbox("Region / store column", opt, index=guess(["region", "store", "branch", "city"], opt, 0))
prod_col = c6.selectbox("Product column", opt, index=guess(["product", "item", "sku"], opt, 0))

# ----------------------------------------------------------------------------
# Step 3: local stats + charts
# ----------------------------------------------------------------------------
st.header("3️⃣ Data overview")
try:
    stats = compute_stats(df, date_col, rev_col, qty_col, cat_col, reg_col, prod_col)
except Exception as e:
    st.error(f"Could not compute statistics — check your column mapping. ({e})")
    st.stop()

if len(stats["monthly"]) == 0:
    st.error("No valid rows after parsing dates and revenue. Check the column mapping.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
monthly = stats["monthly"]
m1.metric("Total revenue", f"{monthly.sum():,.0f}")
m2.metric("Avg monthly", f"{monthly.mean():,.0f}")
m3.metric("Best month", f"{monthly.idxmax():%b %Y}")
m4.metric("Worst month", f"{monthly.idxmin():%b %Y}")

st.subheader("Monthly revenue trend")
st.line_chart(monthly)
cc1, cc2 = st.columns(2)
if "by_cat" in stats:
    cc1.subheader("Revenue by category")
    cc1.bar_chart(stats["by_cat"])
if "by_reg" in stats:
    cc2.subheader("Revenue by region")
    cc2.bar_chart(stats["by_reg"])
with st.expander("Statistics sent to the AI"):
    st.text(stats["text"])

# ----------------------------------------------------------------------------
# Step 4: AI analyses
# ----------------------------------------------------------------------------
st.header("4️⃣ AI analysis & comparison")
business_ctx = st.text_input("Optional business context",
                             placeholder="e.g. small supermarket in Islamabad, planning a Ramadan promotion")

if st.button("🤖 Run AI analysis", type="primary", use_container_width=True):
    api_key = get_api_key()
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif len(models) < 1:
        st.warning("Select at least one model.")
    else:
        client = Groq(api_key=api_key)
        user_msg = f"SALES STATISTICS\n{stats['text']}"
        if business_ctx:
            user_msg += f"\n\nBUSINESS CONTEXT: {business_ctx}"

        analyses = {}
        progress = st.progress(0.0, text="Starting...")
        for i, mdl in enumerate(models):
            progress.progress(i / (len(models) + 1), text=f"Analysing with {mdl}...")
            try:
                analyses[mdl] = call_llm(client, mdl, ANALYST_PROMPT, user_msg, temperature)
            except Exception as e:
                analyses[mdl] = f"⚠️ Error from {mdl}: {e}"

        verdict = None
        good = {k: v for k, v in analyses.items() if not v.startswith("⚠️")}
        if good:
            progress.progress(len(models) / (len(models) + 1), text="Comparing analyses...")
            joined = "\n\n".join(f"=== ANALYSIS FROM {k} ===\n{v}" for k, v in good.items())
            try:
                verdict = call_llm(client, judge_model, JUDGE_PROMPT,
                                   f"SOURCE STATISTICS\n{stats['text']}\n\n{joined}",
                                   temperature=0.2, max_tokens=2200)
            except Exception as e:
                verdict = f"⚠️ Comparison failed: {e}"
        progress.progress(1.0, text="Done")
        st.session_state["analyses"] = analyses
        st.session_state["verdict"] = verdict

if "analyses" in st.session_state:
    analyses = st.session_state["analyses"]
    tabs = st.tabs([f"🧠 {m}" for m in analyses] + ["⚖️ Comparison & Action Plan"])
    for tab, (mdl, text) in zip(tabs[:-1], analyses.items()):
        with tab:
            st.markdown(text)
    with tabs[-1]:
        st.markdown(st.session_state.get("verdict") or "No comparison available.")

    report = [f"# Retail Sales AI Report\n_Generated {datetime.now():%Y-%m-%d %H:%M}_\n",
              "## Statistics\n```\n" + stats["text"] + "\n```\n"]
    for mdl, text in analyses.items():
        report.append(f"# Analysis by {mdl}\n{text}\n")
    report.append("# Comparison & Final Action Plan\n" + (st.session_state.get("verdict") or ""))
    st.download_button("⬇️ Download full report (Markdown)", "\n".join(report),
                       file_name="retail_sales_ai_report.md", mime="text/markdown")
    st.caption("AI output can contain errors. Verify key numbers against your own records before acting.")
