"""
UBPIS — Universal Business & Predictive Insights System
Author: Kona Omeshwar Reddy
Preferred Name: Omeshwar Reddy Kona
Version: 1.1
License: MIT

Notes:
- Upload-first app. No demo or sample datasets.
- Works with arbitrary uploaded datasets — no required column names.
- Large datasets handled safely with preview limits.
"""

# -----------------------------
# Imports
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import io, warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import RandomizedSearchCV

# Optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="UBPIS Pro", layout="wide")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Custom CSS (theme)
# -----------------------------
st.markdown("""
<style>
body { background-color:#ffffff; color:#333333; font-family:'Arial', sans-serif; }
[data-testid="stSidebar"] { background-color:#222222 !important; color:#ffd700 !important; }
[data-testid="stSidebar"] * { color:#ffd700 !important; opacity:1 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color:#ffd700 !important; font-weight:bold !important; }
div[role="radiogroup"] label p, div[role="checkbox"] label p { color:#ffd700 !important; opacity:1 !important; }
[data-testid="stSidebar"] *:hover { color:#ffffff !important; }
.css-12w0qpk { background-color:#f9f9f9 !important; border-radius:10px; padding:18px; box-shadow:0px 4px 10px rgba(0,0,0,0.05); }
.stButton button { background-color:#ffd700 !important; color:#222222; font-weight:bold; }
.stButton button:hover { background-color:#e6c200 !important; }
.block-container { padding:2rem 2.5rem !important; }
.css-1lcbmhc .st-c8 { background-color: #1e1e1e !important; border-radius: 8px; padding: 0.2rem; }
.css-1lcbmhc button[role="tab"] { background-color: #333333 !important; color: #ffffff !important; font-weight: bold; border-radius: 6px; margin: 0.1rem; padding: 0.5rem 1rem; transition: all 0.3s ease; }
.css-1lcbmhc button[role="tab"]:hover { background-color: #444444 !important; transform: scale(1.05); }
.css-1lcbmhc button[aria-selected="true"] { background-color: #ffd700 !important; color: #000000 !important; transform: scale(1.08); box-shadow: 0px 4px 12px rgba(0,0,0,0.25); }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Utils
# -----------------------------
def safe_to_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.Series(pd.NaT, index=series.index)

def summarize_df(df):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing": int(df.isnull().sum().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
    }

def detect_numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def detect_currency_like(col):
    if not isinstance(col, str):
        return False
    cn = col.lower()
    tokens = ["revenue","sales","amount","price","profit","cost","income","total"]
    return any(t in cn for t in tokens)

def format_currency(x, symbol="₹"):
    try:
        return f"{symbol}{float(x):,.2f}"
    except Exception:
        try:
            return f"{float(x):,.2f}"
        except Exception:
            return str(x)

def create_lag_features_series(series, nlags=3):
    s = pd.Series(series).reset_index(drop=True).astype(float)
    df = pd.DataFrame({"y": s})
    for lag in range(1, nlags+1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["roll_mean_3"] = df["y"].rolling(3).mean()
    df["roll_std_3"] = df["y"].rolling(3).std()
    df = df.dropna().reset_index(drop=True)
    if df.shape[0] == 0:
        return np.empty((0, nlags+2)), np.empty((0,)), []
    cols = [f"lag_{i}" for i in range(1, nlags+1)] + ["roll_mean_3", "roll_std_3"]
    X = df[cols].values
    y = df["y"].values
    return X, y, cols

def recursive_forecast_from_lags(initial_lags, model, steps, nlags=3):
    preds = []
    lags = [float(x) for x in initial_lags]
    for _ in range(int(steps)):
        roll_mean = np.mean(lags[:3]) if len(lags)>=3 else np.mean(lags)
        roll_std = np.std(lags[:3]) if len(lags)>=3 else np.std(lags)
        X = np.array(lags[:nlags] + [roll_mean, roll_std]).reshape(1,-1)
        pred = model.predict(X)[0]
        preds.append(float(pred))
        lags = [pred] + lags
    return preds

# -----------------------------
# Session state
# -----------------------------
if "df" not in st.session_state: st.session_state.df = None
if "openai_key" not in st.session_state: st.session_state.openai_key = None
if "models" not in st.session_state: st.session_state.models = {}

def clear_dataset():
    st.session_state.df = None
    st.session_state.models = {}

# -----------------------------
# Sidebar (controls)
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ("Home","Data Explorer","Dashboard","ML Predictions","Anomalies","AI Assistant","Settings"))
st.sidebar.markdown("---")
st.sidebar.title("Data & Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV / XLSX", type=["csv","xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.sidebar.success(f"Loaded {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")

openai_key_input = st.sidebar.text_input("OpenAI API key (optional)", type="password")
if openai_key_input:
    st.session_state.openai_key = openai_key_input

if st.sidebar.button("Clear dataset & cache"):
    clear_dataset()
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.sidebar.success("Dataset and cache cleared")

# -----------------------------
# Require dataset decorator
# -----------------------------
def require_data():
    if st.session_state.df is None:
        st.info("Please upload a CSV or Excel file to proceed.")
        st.stop()
    return True

# -----------------------------
# Pages (Home, Explorer, Dashboard, ML, Anomalies, AI, Settings)
# -----------------------------

# --- HOME ---
if page=="Home":
    st.header("Universal Business & Prediction Insight System — Home")
    require_data()
    df = st.session_state.df
    info = summarize_df(df)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows",info["rows"])
    c2.metric("Columns",info["columns"])
    c3.metric("Missing",info["missing"])
    c4.metric("Memory (MB)",info["memory_mb"])
    st.markdown("---")
    st.subheader("Preview sample rows")
    st.dataframe(df.head(50), use_container_width=True)

# --- DATA EXPLORER ---
elif page=="Data Explorer":
    st.header("Data Explorer")
    require_data()
    df = st.session_state.df
    st.subheader("Preview & Schema")
    st.dataframe(df.head(200), use_container_width=True)
    with st.expander("Column types & null counts"):
        st.write(pd.DataFrame({"dtype":df.dtypes.astype(str),"nulls":df.isnull().sum()}))
    with st.expander("Summary statistics"):
        st.write(df.describe(include="all").T)
    st.subheader("Download")
    buf_csv = io.StringIO()
    df.to_csv(buf_csv,index=False)
    st.download_button("Download CSV",buf_csv.getvalue(),file_name="dataset.csv")
    buf_xlsx = io.BytesIO()
    try:
        df.to_excel(buf_xlsx,index=False,engine="openpyxl")
        st.download_button("Download Excel",buf_xlsx.getvalue(),file_name="dataset.xlsx")
    except Exception:
        pass
# --- DASHBOARD (FLEXIBLE) ---
elif page=="Dashboard":
    st.header("Dashboard — Flexible Visual Builder")
    if not require_data(): st.stop()
    df = st.session_state.df.copy()

    st.subheader("Visualization Controls")
    with st.expander("Dataset & quick options", expanded=True):
        st.write("Preview (first 5 rows):")
        st.dataframe(df.head(5))
        max_rows = st.number_input("Limit rows to process (0 = all)", min_value=0, value=0, step=100)
        if max_rows and max_rows > 0:
            df = df.head(max_rows)

    # ---------------------------
    # UNIVERSAL COLUMN DETECTION
    # ---------------------------
    # We do robust detection so visuals work for any schema.
    cols = df.columns.tolist()
    num_cols = detect_numeric_cols(df)
    text_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    date_cols = [c for c in cols if "date" in c.lower() or "time" in c.lower()]
    revenue_candidates = [c for c in num_cols if any(tok in c.lower() for tok in ["rev","sale","amount","total","income","price"])]
    # fallback revenue: pick first numeric if nothing obvious
    revenue_col = (revenue_candidates[0] if revenue_candidates else (num_cols[0] if num_cols else None))
    # fallback category: pick first text column
    category_col = text_cols[0] if text_cols else (cols[0] if cols else None)
    category_col_2 = text_cols[1] if len(text_cols) > 1 else None

    # Provide the selection UI but prefill with safe defaults
    valid_cols = cols.copy()
    if not valid_cols:
        st.warning("Dataset is empty — no columns available.")
        st.stop()

    # Safe session-state defaults
    if 'x_col' not in st.session_state or st.session_state.get('x_col') not in valid_cols:
        st.session_state['x_col'] = category_col if category_col else valid_cols[0]
    if 'y_cols' not in st.session_state or not set(st.session_state.get('y_cols', [])).issubset(valid_cols):
        st.session_state['y_cols'] = [revenue_col] if revenue_col else [valid_cols[0]]
    if 'color_col' not in st.session_state or (st.session_state.get('color_col') != "None" and st.session_state.get('color_col') not in valid_cols):
        st.session_state['color_col'] = "None"

    # Chart builder controls (keeps your original UI choices)
    chart_mode = st.selectbox("Choose chart mode", ["Quick Presets", "Custom Builder"])
    if chart_mode == "Quick Presets":
        preset = st.selectbox("Preset Visual", ["Auto Overview", "Top Items", "Distribution", "Heatmap"])
    else:
        preset = None

    chart_type = st.selectbox("Chart Type", ["Auto","Bar","Line","Area","Scatter","Pie","Treemap","Heatmap","Box","Scatter Matrix","Stacked Area"])
    x_col = st.selectbox("X-axis / Category", valid_cols, index=valid_cols.index(st.session_state['x_col']) if st.session_state['x_col'] in valid_cols else 0)
    st.session_state['x_col'] = x_col
    y_cols = st.multiselect("Y-axis (one or more)", valid_cols, default=st.session_state['y_cols'])
    st.session_state['y_cols'] = y_cols
    color_col = st.selectbox("Color / Legend Column (optional)", ["None"] + valid_cols, index=(["None"] + valid_cols).index(st.session_state['color_col']) if st.session_state['color_col'] in (["None"]+valid_cols) else 0)
    st.session_state['color_col'] = color_col

    agg_func = st.selectbox("Aggregation function (for grouped charts)", ["sum","mean","median","count","max","min"])
    groupby_cols = st.multiselect("Group by columns (optional)", valid_cols, default=[])

    st.markdown("---")
    st.subheader("Advanced Visual Options")
    show_annotations = st.checkbox("Show data labels / annotations", value=False)
    normalize = st.checkbox("Normalize numeric columns before plotting (z-score)", value=False)
    rolling_window = st.number_input("Rolling window for smoothing (0 = off)", min_value=0, max_value=60, value=0)

    # Prepare dataframe for plotting (safe conversions)
    plot_df = df.copy()
    # attempt best-effort conversions for numeric/date where appropriate
    for c in plot_df.columns:
        # try numeric conversion for mixed-type numeric columns
        if plot_df[c].dtype == object:
            try:
                plot_df[c] = pd.to_numeric(plot_df[c], errors='ignore')
            except Exception:
                pass
    # apply normalization if asked
    if normalize and num_cols:
        try:
            plot_df[num_cols] = (plot_df[num_cols] - plot_df[num_cols].mean()) / (plot_df[num_cols].std(ddof=0) + 1e-9)
        except Exception:
            pass

    # Apply groupby aggregation if requested (and sanitize groupby columns)
    safe_groupby = [c for c in groupby_cols if c in plot_df.columns]
    try:
        if safe_groupby and y_cols:
            safe_y = [c for c in y_cols if c in plot_df.columns]
            if safe_y:
                gb = plot_df.groupby(safe_groupby)[safe_y].agg(agg_func).reset_index()
                plot_df_plot = gb
            else:
                plot_df_plot = plot_df
        else:
            plot_df_plot = plot_df
    except Exception:
        plot_df_plot = plot_df

    # Auto-detect chart_type if Auto selected
    if chart_type == "Auto":
        if len(y_cols) >= 2:
            chart_type = "Scatter Matrix"
        elif len(y_cols) == 1 and x_col:
            chart_type = "Bar"
        elif len(y_cols) == 1:
            chart_type = "Pie"
        else:
            chart_type = "Bar"

    # Visualization output (robust, with fallbacks)
    st.markdown("### Visualization Output")
    try:
        # Ensure requested columns exist, otherwise warn and fallback
        selected_x = x_col if x_col in plot_df_plot.columns else None
        selected_y = [c for c in y_cols if c in plot_df_plot.columns]
        selected_color = color_col if (color_col != "None" and color_col in plot_df_plot.columns) else None

        if chart_type == "Bar":
            if not selected_x or not selected_y:
                st.warning("Select X-axis and at least one Y-axis for Bar chart (or pick different columns).")
            else:
                fig = px.bar(plot_df_plot, x=selected_x, y=selected_y, color=selected_color, title="Bar Chart")
                if rolling_window and len(selected_y) == 1 and selected_x in plot_df_plot.columns:
                    try:
                        plot_df_plot['_ma'] = plot_df_plot[selected_y[0]].rolling(window=rolling_window, min_periods=1).mean()
                        fig.add_trace(go.Scatter(x=plot_df_plot[selected_x], y=plot_df_plot['_ma'], name=f"{rolling_window}-period MA"))
                    except Exception:
                        pass
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line":
            if not selected_x or not selected_y:
                st.warning("Select X-axis and Y-axis for Line chart.")
            else:
                fig = px.line(plot_df_plot, x=selected_x, y=selected_y, color=selected_color, title="Line Chart", markers=True)
                if rolling_window:
                    for y in selected_y:
                        if y in plot_df_plot.columns:
                            try:
                                plot_df_plot[f"{y}_ma"] = plot_df_plot[y].rolling(window=rolling_window, min_periods=1).mean()
                                fig.add_trace(go.Scatter(x=plot_df_plot[selected_x], y=plot_df_plot[f"{y}_ma"], name=f"{y} MA"))
                            except Exception:
                                pass
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type in ["Area", "Stacked Area"]:
            if not selected_x or not selected_y:
                st.warning("Select X-axis and Y-axis for Area chart.")
            else:
                fig = px.area(plot_df_plot, x=selected_x, y=selected_y, color=selected_color, title="Area Chart")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter":
            if len(selected_y) != 1 or not selected_x:
                st.warning("Select X-axis and exactly one Y-axis for Scatter.")
            else:
                fig = px.scatter(plot_df_plot, x=selected_x, y=selected_y[0], color=selected_color, title="Scatter Plot", hover_data=plot_df_plot.columns.tolist())
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pie":
            # For pie, allow either category column as names and a numeric as values, or aggregate automatically
            if selected_x and selected_y and len(selected_y) == 1:
                # If selected_x is category and selected_y numeric, use that
                if plot_df_plot[selected_x].nunique() > 1 and np.issubdtype(plot_df_plot[selected_y[0]].dtype, np.number):
                    fig = px.pie(plot_df_plot, names=selected_x, values=selected_y[0], title="Pie Chart")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # fallback: group by x and sum numeric columns
                    numeric_for_pie = [c for c in plot_df_plot.columns if np.issubdtype(plot_df_plot[c].dtype, np.number)]
                    if numeric_for_pie:
                        agg = plot_df_plot.groupby(selected_x)[numeric_for_pie[0]].sum().reset_index()
                        fig = px.pie(agg, names=selected_x, values=numeric_for_pie[0], title="Pie Chart")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No suitable numeric column found for Pie chart.")
            else:
                # Try to auto-build a pie using detected revenue_col and category_col
                if revenue_col and category_col and revenue_col in plot_df_plot.columns and category_col in plot_df_plot.columns:
                    agg = plot_df_plot.groupby(category_col)[revenue_col].sum().reset_index()
                    fig = px.pie(agg, names=category_col, values=revenue_col, title=f"{revenue_col} share by {category_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Pie chart needs one category and one numeric column.")

        elif chart_type == "Treemap":
            # Build treemap using x as path and first numeric y
            numeric_for_treemap = [c for c in plot_df_plot.columns if np.issubdtype(plot_df_plot[c].dtype, np.number)]
            if selected_x and numeric_for_treemap:
                path = [selected_x] + safe_groupby if safe_groupby else [selected_x]
                fig = px.treemap(plot_df_plot, path=path, values=numeric_for_treemap[0], title="Treemap")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Treemap needs a category column and at least one numeric column.")

        elif chart_type in ["Heatmap", "Advanced Heatmap"]:
            numeric_df = plot_df_plot.select_dtypes(include=[np.number])
            if numeric_df.shape[1] >= 2:
                corr = numeric_df.corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # try pivot heatmap
                if selected_x and selected_y:
                    try:
                        pivot = pd.pivot_table(plot_df_plot, index=selected_x, values=selected_y, aggfunc=agg_func)
                        fig = px.imshow(pivot.fillna(0).values, x=pivot.columns, y=pivot.index, title="Pivot Heatmap")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Heatmap build failed: {e}")
                else:
                    st.warning("Not enough numeric columns for a heatmap.")

        elif chart_type == "Box":
            if not selected_y:
                st.warning("Select at least one numeric column for Boxplot.")
            else:
                for y in selected_y:
                    if y in plot_df_plot.columns:
                        fig = px.box(plot_df_plot, y=y, points="all", title=f"Boxplot: {y}")
                        st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Matrix":
            dims = [c for c in selected_y if c in plot_df_plot.columns and np.issubdtype(plot_df_plot[c].dtype, np.number)]
            if len(dims) < 2:
                st.warning("Select 2 or more numeric columns for Scatter Matrix.")
            else:
                fig = px.scatter_matrix(plot_df_plot, dimensions=dims, color=selected_color, title="Scatter Matrix")
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Select a chart type to render.")
    except Exception as e:
        st.error(f"Visualization error: {e}")

    # Quick preset shortcuts (kept for backward compatibility)
    if chart_mode == "Quick Presets" and preset:
        if preset == "Auto Overview":
            rev_cols = [c for c in df.columns if detect_currency_like(c)]
            date_cols_local = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            if rev_cols and date_cols_local:
                temp = df.copy()
                temp[date_cols_local[0]] = safe_to_datetime(temp[date_cols_local[0]])
                daily = temp.groupby(temp[date_cols_local[0]].dt.date)[rev_cols[0]].sum().reset_index()
                fig = px.line(daily, x=daily.columns[0], y=rev_cols[0], title="Revenue Overview")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Auto Overview requires a date and revenue-like numeric column.")
        elif preset == "Top Items":
            prod_cols = [c for c in df.columns if any(tok in c.lower() for tok in ["product","item","sku","name"])]
            rev_cols = [c for c in df.columns if detect_currency_like(c)]
            if prod_cols and rev_cols:
                top10 = df.groupby(prod_cols[0])[rev_cols[0]].sum().nlargest(10).reset_index()
                fig = px.bar(top10, x=prod_cols[0], y=rev_cols[0], title="Top 10 Items")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Top Items preset needs a product-like text column and a revenue-like numeric column.")
        elif preset == "Distribution":
            if num_cols:
                for n in num_cols[:3]:
                    fig = px.histogram(df, x=n, nbins=30, title=f"Distribution: {n}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns available for distribution preset.")
        elif preset == "Heatmap":
            numeric = df.select_dtypes(include=[np.number])
            if not numeric.empty:
                corr = numeric.corr()
                st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation Heatmap"), use_container_width=True)
            else:
                st.info("No numeric columns for heatmap preset.")

    # Export current plot data
    st.markdown("---")
    st.subheader("Export / Save")
    if st.button("Download current filtered dataset as CSV"):
        buf = io.StringIO()
        try:
            plot_df_plot.to_csv(buf, index=False)
            st.download_button("Click to download", buf.getvalue(), file_name="ubpis_filtered_data.csv")
        except Exception as e:
            st.error(f"Failed to prepare download: {e}")
    if st.button("Save filtered dataset to outputs folder"):
        try:
            out_path = OUTPUTS_DIR / f"ubpis_filtered_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
            plot_df_plot.to_csv(out_path, index=False)
            st.success(f"Saved to {out_path}")
        except Exception as e:
            st.error(f"Save failed: {e}")

    # Correlation & Boxplots
    st.markdown("---")
    st.subheader("Correlation & Boxplots")
    numeric_cols_local = detect_numeric_cols(df)
    if len(numeric_cols_local) >= 2:
        corr = df[numeric_cols_local].corr()
        try:
            st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation matrix"), use_container_width=True)
        except Exception:
            st.write(corr)
    sel = st.multiselect("Pick numeric columns for boxplots", numeric_cols_local, default=numeric_cols_local[:3])
    for c in sel:
        if c in df.columns:
            st.plotly_chart(px.box(df, y=c, points="all", title=f"Boxplot: {c}"), use_container_width=True)

# --- ML PREDICTIONS ---
elif page=="ML Predictions":
    st.header("ML Predictions — Profit/Loss Forecasting")
    if not require_data(): st.stop()
    df = st.session_state.df.copy()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = detect_numeric_cols(df)
    
    product_col = st.selectbox("Product column",[None]+cat_cols)
    target_col = st.selectbox("Target numeric column (e.g., profit)",[None]+num_cols)
    nlags = st.number_input("Lag features",1,12,3)
    steps = st.number_input("Days ahead to predict",1,365,30)
    test_fraction = st.slider("Test fraction",0.05,0.5,0.2)
    tune = st.checkbox("Enable randomized hyperparameter tuning",value=True)
    
    if st.button("Run ML predictions"):
        if product_col is None or target_col is None:
            st.error("Select product column and numeric target")
        else:
            products = df[product_col].dropna().unique()
            results = []
            errors = []
            st.session_state.models = {}
            prog = st.progress(0)
            total = len(products) if len(products)>0 else 1
            for i,p in enumerate(products):
                try:
                    sub = df[df[product_col]==p].dropna(subset=[target_col]).copy()
                    dcols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
                    if dcols:
                        sub[dcols[0]] = safe_to_datetime(sub[dcols[0]])
                        sub = sub.sort_values(by=dcols[0])
                    X,y,feat_cols = create_lag_features_series(sub[target_col],nlags)
                    if len(X)==0: continue
                    model = RandomForestRegressor(n_estimators=100,random_state=42)
                    if tune:
                        params = {
                            "n_estimators":[50,100,200],
                            "max_depth":[3,5,10,None],
                            "min_samples_split":[2,5,10]
                        }
                        model = RandomizedSearchCV(model,params,n_iter=5,cv=2)
                    model.fit(X,y)
                    st.session_state.models[p] = model
                    forecast = recursive_forecast_from_lags(X[-1],model,int(steps),nlags)
                    results.append(pd.DataFrame({"product":[p]*len(forecast),"day_ahead":range(1,len(forecast)+1),target_col:forecast}))
                except Exception as e:
                    errors.append(f"{p}: {e}")
                prog.progress((i+1)/total)
            if results:
                df_forecast = pd.concat(results,ignore_index=True)
                st.success("Forecast completed")
                st.dataframe(df_forecast.head(50),use_container_width=True)
                buf_csv = io.StringIO()
                df_forecast.to_csv(buf_csv,index=False)
                st.download_button("Download Forecast CSV",buf_csv.getvalue(),file_name="forecast.csv")
            if errors:
                st.warning("Some errors occurred:\n"+"\n".join(errors))

# --- ANOMALIES ---
elif page=="Anomalies":
    st.header("Anomaly Detection")
    if not require_data(): st.stop()
    df = st.session_state.df.copy()
    num_cols = detect_numeric_cols(df)
    if not num_cols:
        st.warning("No numeric columns available for anomaly detection.")
    else:
        selected_col = st.selectbox("Select numeric column for anomaly detection",num_cols)
        method = st.selectbox("Detection method",["Z-score","IsolationForest"])
        threshold = st.slider("Z-score threshold",2.0,5.0,3.0) if method=="Z-score" else None
        if st.button("Detect anomalies"):
            try:
                if method=="Z-score":
                    mean,std = df[selected_col].mean(), df[selected_col].std()
                    df["anomaly"] = (np.abs(df[selected_col]-mean)/std>threshold).astype(int)
                else:
                    iso = IsolationForest(contamination=0.05,random_state=42)
                    df["anomaly"] = iso.fit_predict(df[[selected_col]])
                    df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x==-1 else 0)
                st.plotly_chart(px.scatter(df, x=df.index, y=selected_col, color="anomaly", title=f"Anomaly detection: {selected_col}"), use_container_width=True)
            except Exception as e:
                st.error(f"Failed: {e}")

# --- AI ASSISTANT ---
elif page=="AI Assistant":
    st.header("AI Assistant — Ask about your data")
    if not require_data(): st.stop()
    df = st.session_state.df.copy()
    query = st.text_area("Enter your question")
    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a question")
        else:
            # Offline simple answers
            if "profit" in query.lower():
                ans = f"Total profit: {df['profit'].sum()}" if "profit" in df.columns else "No profit column found"
                st.info(ans)
            elif OPENAI_AVAILABLE and st.session_state.openai_key:
                try:
                    openai.api_key = st.session_state.openai_key
                    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                        messages=[{"role":"user","content":query}])
                    st.info(resp["choices"][0]["message"]["content"])
                except Exception as e:
                    st.error(f"OpenAI API failed: {e}")
            else:
                st.info("No OpenAI key provided. Answering offline only.")

# --- SETTINGS ---
elif page=="Settings":
    st.header("Settings & App Info")
    st.markdown("""
    **Universal Business & Prediction Insight System**
    - Author: Kona Omeshwar Reddy 
    - Version: 12.0 — Flexible Final Edition  
    - Features: Multi-chart, ML Forecast, Anomaly Detection, AI Assistant, Data Explorer, CSV/Excel export
    """)
    st.markdown("---")
    st.subheader("Theme Customization")
    st.info("Theme is controlled via Streamlit markdown CSS. Adjust colors, fonts, hover effects in the code section.")
    st.subheader("App Settings")
    st.checkbox("Enable advanced forecasting (ARIMA/Prophet) [placeholder]", value=False)
    st.checkbox("Show moving average overlays by default", value=False)
    st.text_input("Default export filename prefix", value="ubpis_export")
    st.markdown("**Danger Zone**")
    if st.button("Clear all session data"):
        clear_dataset()
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.success("All session data cleared.")

# End of file
