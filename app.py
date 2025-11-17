import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import time
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from forecasting_utils import recursive_forecast
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib
import pickle

USER_DB_FILE = "users.pkl"

def load_user_db():
    if os.path.exists(USER_DB_FILE):
        return pickle.load(open(USER_DB_FILE, "rb"))
    return {"admin": hashlib.sha256("password123".encode()).hexdigest()}   # default admin

def save_user_db(db):
    pickle.dump(db, open(USER_DB_FILE, "wb"))

# Load user DB into session
if "user_db" not in st.session_state:
    st.session_state["user_db"] = load_user_db()

USER_CREDENTIALS = st.session_state["user_db"]


# --------------------------------------------------------------
# LOAD RESOURCES
# --------------------------------------------------------------
@st.cache_resource
def load_all_data():
    """Load inventory dataset, forecasting dataset, and all models."""

    # ---- File paths ----
    inv_file = r"C:\Users\Anushka\Documents\AIML Final Practical Hackathon\dataset_with_stock_and_forecast_simulated_new.csv"
    stock_model_file = "models/xgb_stock_risk_model.pkl"  # <-- 3-class model
    forecast_file = r"C:\Users\Anushka\Documents\AIML Final Practical Hackathon\prepared_dataset_one_store.csv"
    rf_file = "models/rf_model.pkl"
    xgb_file = "models/xgb_model.pkl"

    required_files = [
        inv_file, stock_model_file,
        forecast_file, rf_file, xgb_file
    ]

    for f in required_files:
        if not os.path.exists(f):
            st.error(f" Missing required file: **{f}**")
            raise FileNotFoundError(f)

    # ---- Load datasets ----
    inv_df = pd.read_csv(inv_file)
    forecast_df = pd.read_csv(forecast_file)
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    # ---- Load stockout classifier ----
    raw_stock_model = joblib.load(stock_model_file)
    stock_model = raw_stock_model[0] if isinstance(raw_stock_model, tuple) else raw_stock_model

    # ---- Load forecasting models ----
    def load_model(path):
        m = pickle.load(open(path, "rb"))
        return m[0] if isinstance(m, tuple) else m

    models = {
        "Random Forest": load_model(rf_file),
        "XGBoost": load_model(xgb_file),
    }

    st.success("All datasets and models loaded successfully.")
    return inv_df, forecast_df, stock_model, models


try:
    inv_df, forecast_df, stock_model, models = load_all_data()
except:
    st.stop()

risk_map = {0: "LOW RISK", 1: "MEDIUM RISK", 2: "HIGH RISK"}

# --------------------------------------------------------------
# PAGE 1 — FORECASTING DASHBOARD
# --------------------------------------------------------------
def forecasting_dashboard():
    st.title(" Product-Level Demand Forecasting")

    # Sidebar Filters
    st.sidebar.header("Filters")
    item = st.sidebar.selectbox("Select Item", sorted(forecast_df["item_id"].unique()))
    store = st.sidebar.selectbox("Store", sorted(forecast_df["store_id"].unique()))
    model_choice = st.sidebar.selectbox("Model", list(models.keys()))
    horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)

    model = models[model_choice]

    # Filter last 60 days history for selected item-store
    history = (
        forecast_df[(forecast_df["item_id"] == item) & (forecast_df["store_id"] == store)]
        .sort_values("date")
        .tail(60)
        .copy()
    )

    if history.empty:
        st.error("No historical data available for this item/store!")
        return

    if "sales" not in history.columns:
        if "units_sold" in history.columns:
            history["sales"] = history["units_sold"]
        else:
            history["sales"] = 0

    keep_cols = [
        "item_id", "dept_id", "cat_id", "store_id", "date",
        "sales", "sell_price", "snap_CA", "snap_TX", "snap_WI", "snap", "is_event",
        "wday", "month", "year", "wm_yr_wk"
    ]
    history = history[[c for c in keep_cols if c in history.columns]].copy()

    for lag in [1, 7, 14, 28]:
        history[f"lag_{lag}"] = history["sales"].shift(lag).fillna(0)
    for win in [7, 30, 60]:
        history[f"rolling_mean_{win}"] = history["sales"].rolling(win).mean().fillna(0)
        history[f"rolling_std_{win}"] = history["sales"].rolling(win).std().fillna(0)

    if 'sell_price' in history.columns:
        history['rolling_price_mean_30'] = history['sell_price'].rolling(30).mean().fillna(history['sell_price'].iloc[-1])
        history['price_ratio'] = history['sell_price'] / history['rolling_price_mean_30'].replace(0,1)
    else:
        history['rolling_price_mean_30'] = 0
        history['price_ratio'] = 1

    for f in model.feature_names_in_:
        if f not in history.columns:
            history[f] = 0
    features = list(model.feature_names_in_)

    st.subheader(f"Forecasting for: **{item} | {store}**")

    if st.button("Generate Forecast"):
        forecast = recursive_forecast(model, features, history, horizon)

        # Store 30-day total forecast for Inventory Optimization
        st.session_state["future_demand_30_demo"] = forecast["forecast"].sum()

        fig = px.line()
        fig.add_scatter(x=history["date"], y=history["sales"], mode="lines", name="History")
        fig.add_scatter(x=forecast["date"], y=forecast["forecast"], mode="lines", name="Forecast")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(forecast)
        csv = forecast.to_csv(index=False).encode()
        st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")

# --------------------------------------------------------------
# PAGE 2 — INVENTORY OPTIMIZATION
# --------------------------------------------------------------
def inventory_optimization():
    st.title("🛒 Inventory Optimization & Stock Risk Prediction")
    st.write(f"Welcome, **{st.session_state['username']}**")

    # Sidebar
    with st.sidebar:
        st.info(f"Logged in as: {st.session_state['username']}")
        if st.button("Logout", key="logout_inventory"):
            st.session_state["authenticated"] = False
            st.session_state["page"] = "login"
            st.rerun()

        

        st.header("Product Selection")
        item_id = st.selectbox("Item ID", sorted(inv_df["item_id"].unique()))

        sku = inv_df[inv_df["item_id"] == item_id].tail(1)
        if sku.empty:
            st.error("No data available for selected item!")
            return

        # Use session forecast if available
        if "future_demand_30_demo" in st.session_state:
            future_demand_30 = st.session_state["future_demand_30_demo"]
        else:
            future_demand_30 = sku["forecast_30"].values[0] if "forecast_30" in sku.columns else 100

        st.markdown("**Demo Current Stock for Risk Testing:**")
        st.markdown(f"- Low Risk: ≥ {int(1.5*future_demand_30)} units")
        st.markdown(f"- Medium Risk: {int(0.5*future_demand_30)} – {int(1.5*future_demand_30)} units")
        st.markdown(f"- High Risk: < {int(0.5*future_demand_30)} units")

        current_stock = st.number_input("Current Stock", 0, value=int(future_demand_30), step=1)
        lead_time = st.number_input("Supplier Lead Time (days)", 1, value=7)

    if st.button("Run Optimization"):
        stockout_features = [
            "lag_1", "lag_7", "lag_14", "lag_28",
            "rolling_mean_7", "rolling_std_7",
            "rolling_mean_30", "rolling_std_30",
            "rolling_mean_60", "rolling_std_60",
            "sell_price", "price_ratio",
            "wday", "month", "year",
            "snap", "is_event",
            "current_stock",
            "future_demand_30"
        ]

        X = sku.copy()
        X["current_stock"] = current_stock
        X["future_demand_30"] = future_demand_30

        for f in stockout_features:
            if f not in X.columns:
                X[f] = 0
        X = X[stockout_features]

        pred = int(stock_model.predict(X)[0])
        risk = risk_map[pred]

        # Safe risk probabilities
        # if hasattr(stock_model, "predict_proba"):
        #     probs = stock_model.predict_proba(X)[0]
        #     prob_dict = {}
        #     for i, p in enumerate(probs):
        #         prob_dict[risk_map[i]] = f"{p*100:.1f}%"
        #     st.subheader("📊 Risk Probabilities")
        #     st.write(prob_dict)

        rolling_std_30 = sku["rolling_std_30"].values[0] if "rolling_std_30" in sku.columns else 0
        daily = future_demand_30 / 30
        safety = rolling_std_30 * 1.65
        reorder_point = daily * lead_time + safety
        reorder_qty = max(0, reorder_point - current_stock)

        st.subheader(" Stock Risk")
        if pred == 2:
            st.error(f"HIGH RISK — {risk}")
        elif pred == 1:
            st.warning(f"MEDIUM RISK — {risk}")
        else:
            st.success(f"LOW RISK — {risk}")

        st.markdown("---")
        st.subheader(" Inventory Recommendation")
        #st.write(f"Daily Demand: **{daily:.2f} units**")
        st.write(f"Safety Stock: **{safety:.0f} units**")
        st.write(f"Reorder Point: **{reorder_point:.0f} units**")

        if reorder_qty > 0:
            st.error(f"REORDER REQUIRED → **Order {reorder_qty:.0f} units**")
        else:
            st.success("Stock sufficient. No reorder required.")

        chart = pd.DataFrame({
            "Units": [current_stock, reorder_point, future_demand_30]
        }, index=["Current Stock", "Reorder Point", "30-day Forecast"])
        st.bar_chart(chart)

        

#Page 3
def analytics_page():
    st.title(" Sales Analysis Dashboard")

    # Ensure date is datetime
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])

    # Sidebar filters
    st.sidebar.header("Filters")
    item = st.sidebar.selectbox("Select Item", sorted(forecast_df["item_id"].unique()))
    store = st.sidebar.selectbox("Select Store", sorted(forecast_df["store_id"].unique()))

    # Filter data
    filtered_df = forecast_df[
        (forecast_df["item_id"] == item) & 
        (forecast_df["store_id"] == store)
    ].copy()

    if filtered_df.empty:
        st.warning("No data available for this item/store!")
        return

    # --- DAILY SALES ---
    daily_sales = filtered_df.groupby('date')['sales'].sum().reset_index()

    # --- WEEKLY SALES ---
    filtered_df["week"] = filtered_df["date"].dt.isocalendar().week
    filtered_df["year"] = filtered_df["date"].dt.year

    weekly_sales = (
        filtered_df.groupby(["year", "week"])["sales"]
        .sum()
        .reset_index()
    )

    weekly_sales["week_start"] = pd.to_datetime(
        weekly_sales['year'].astype(str) + '-W' +
        weekly_sales['week'].astype(str) + '-1',
        format='%G-W%V-%u'
    )

    # --- MONTHLY SALES ---
    filtered_df["month"] = filtered_df["date"].dt.month

    monthly_sales = (
        filtered_df.groupby(["year", "month"])["sales"]
        .sum()
        .reset_index()
    )

    monthly_sales["month_year"] = pd.to_datetime(
        monthly_sales["year"].astype(str) + "-" +
        monthly_sales["month"].astype(str) + "-01"
    )

    # BEST / WORST MONTH
    best_idx = monthly_sales['sales'].idxmax()
    worst_idx = monthly_sales['sales'].idxmin()

    best_month = monthly_sales.loc[best_idx]
    worst_month = monthly_sales.loc[worst_idx]

    st.subheader(
        f" Best Month: **{best_month['month_year'].strftime('%B %Y')}** "
        f"({best_month['sales']} units)"
    )
    st.subheader(
        f" Worst Month: **{worst_month['month_year'].strftime('%B %Y')}** "
        f"({worst_month['sales']} units)"
    )

    # ------------------------------- PLOTS -------------------------------

    # DAILY
    st.subheader(" Daily Sales Trend")
    fig_daily = px.line(
        daily_sales, x="date", y="sales",
        title=f"Daily Sales for {item} | {store}"
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    # WEEKLY
    st.subheader(" Weekly Sales Trend")
    fig_weekly = px.line(
        weekly_sales, x="week_start", y="sales",
        title="Weekly Sales"
    )
    st.plotly_chart(fig_weekly, use_container_width=True)

    # MONTHLY
    st.subheader(" Monthly Sales Summary")
    fig_monthly = px.bar(
        monthly_sales, x="month_year", y="sales",
        title="Monthly Sales"
    )

    # Highlight Best Month
    fig_monthly.add_scatter(
        x=[best_month["month_year"]],
        y=[best_month["sales"]],
        mode="markers+text",
        text=["Best"],
        marker=dict(color="green", size=12)
    )

    # Highlight Worst Month
    fig_monthly.add_scatter(
        x=[worst_month["month_year"]],
        y=[worst_month["sales"]],
        mode="markers+text",
        text=["Worst"],
        marker=dict(color="red", size=12)
    )

    st.plotly_chart(fig_monthly, use_container_width=True)

    # RAW DATA
    st.subheader(" Raw Filtered Data")
    st.dataframe(filtered_df)

# --------------------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------------------
def login_page():
    st.title(" Login")

    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    if st.button("Login"):
        hashed_pw = hashlib.sha256(pw.encode()).hexdigest()

        if user in USER_CREDENTIALS and USER_CREDENTIALS[user] == hashed_pw:
            st.session_state["authenticated"] = True
            st.session_state["username"] = user
            st.session_state["page"] = "forecasting"
            st.rerun()
        else:
            st.error(" Invalid username or password")

    st.write("Don't have an account?")
    if st.button("Register Here"):
        st.session_state["page"] = "register"
        st.rerun()

def register_page():
    st.title(" Register")

    user = st.text_input("Choose Username")
    pw = st.text_input("Choose Password", type="password")
    pw2 = st.text_input("Confirm Password", type="password")

    if st.button("Create Account"):
        if not user or not pw:
            st.error(" Username and password required")
            return

        if pw != pw2:
            st.error(" Passwords do not match")
            return

        if user in USER_CREDENTIALS:
            st.error(" Username already exists")
            return

        hashed_pw = hashlib.sha256(pw.encode()).hexdigest()

        USER_CREDENTIALS[user] = hashed_pw
        save_user_db(USER_CREDENTIALS)

        st.success(" Account created! Please log in.")
        st.session_state["page"] = "login"
        st.rerun()

    if st.button("Back to Login"):
        st.session_state["page"] = "login"
        st.rerun()

# --------------------------------------------------------------
# PAGE 4 — CLUSTERING & HEATMAP ANALYSIS
# --------------------------------------------------------------
def clustering_page():
    st.title(" Product Clustering & Heatmap Analysis")

    st.write("""
    This module applies **KMeans clustering** to group similar products 
    and generates a **high-contrast heatmap** of cluster centers.
    """)

    # --------------------------
    # Dataset
    # --------------------------
    df = inv_df.copy()  # use your inventory dataset

    features = [
        "sales", 
        "sell_price",
        "lag_7",
        "rolling_mean_30",
        "rolling_std_30",
        "price_ratio",
        "is_event",
        "snap",
        "current_stock"
    ]

    # If any feature missing → fill with 0
    for f in features:
        if f not in df.columns:
            df[f] = 0

    X = df[features].fillna(0)

    # --------------------------
    # Sidebar: Choose K
    # --------------------------
    st.sidebar.subheader("Clustering Settings")
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4)

    # --------------------------
    # Scale
    # --------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------
    # KMeans
    # --------------------------
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    st.success("✔ Clustering Completed")

    # --------------------------
    # Optional: Business Labels
    # --------------------------
    label_map = {
        0: "High-Demand Fast Movers",
        1: "Slow Movers",
        2: "Event-Driven Products",
        3: "Price-Sensitive Products"
    }

    df["cluster_label"] = df["cluster"].map(
        lambda x: label_map.get(x, f"Cluster {x}")
    )

    # --------------------------
    # Heatmap of Cluster Centers
    # --------------------------
    st.subheader(" High-Contrast Cluster Centers Heatmap")

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        centers,
        cmap="Spectral",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="black",
        ax=ax
    )
    st.pyplot(fig)

    # --------------------------
    # Clustered Scatter Plot
    # --------------------------
    st.subheader(" Cluster Visualization (Sales vs Price)")

    if "sales" in df.columns and "sell_price" in df.columns:
        fig2 = px.scatter(
            df,
            x="sales",
            y="sell_price",
            color="cluster_label",
            hover_data=["item_id"],
            title="Cluster Scatter Plot",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --------------------------
    # Cluster Distribution
    # --------------------------
    st.subheader(" Cluster Distribution")
    st.write(df["cluster_label"].value_counts())

    # --------------------------
    # Sample Items from Each Cluster
    # --------------------------
    st.subheader(" Sample Items from Each Cluster")

    for c in sorted(df["cluster"].unique()):
        st.write(f"### Cluster {c} — {df[df['cluster']==c]['cluster_label'].iloc[0]}")
        st.dataframe(
            df[df["cluster"] == c][["item_id", "sales", "sell_price"]].head(5)
        )

    # --------------------------
    # Show full clustered dataset
    # --------------------------
    st.subheader(" Full Clustered Dataset")
    st.dataframe(df)


# --------------------------------------------------------------
# PAGE 5 — STORE WIDE ANALYSIS
# --------------------------------------------------------------
def store_analytics_page():
    # Ensure 'date' column is datetime
    # NOTE: This assumes forecast_df is the full, unfiltered dataframe.
    if 'date' in forecast_df.columns:
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])

    # --- DASHBOARD TITLE ---
    st.title("Store-Wide Sales Analysis Dashboard")
    st.write("Analyzes the combined sales performance of all items within a selected store.")

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Store Filter")
    # Only filter by store, no item filter here
    store = st.sidebar.selectbox("Select Store for Analysis", sorted(forecast_df["store_id"].unique()))

    # Filter dataframe only by store
    # Use .copy() to avoid SettingWithCopyWarning
    filtered_df = forecast_df[forecast_df["store_id"] == store].copy()

    if filtered_df.empty:
        st.warning(f"No data available for Store {store}!")
    else:
        st.header(f"Total Sales Performance for Store: *{store}*")

        # --- AGGREGATIONS ---

        # Daily Sales (Summed across all items)
        daily_sales = filtered_df.groupby('date')['sales'].sum().reset_index()

        # Monthly Sales (Summed across all items)
        filtered_df['month'] = filtered_df['date'].dt.month
        filtered_df['year'] = filtered_df['date'].dt.year

        monthly_sales = filtered_df.groupby(['year','month'])['sales'].sum().reset_index()
        monthly_sales['month_year'] = pd.to_datetime(monthly_sales['year'].astype(str) + '-' + monthly_sales['month'].astype(str) + '-01')

        # --- BEST/WORST MONTH (Store-wide) ---
        best_idx = monthly_sales['sales'].idxmax()
        worst_idx = monthly_sales['sales'].idxmin()
        best_month = monthly_sales.loc[best_idx]
        worst_month = monthly_sales.loc[worst_idx]

        col_best, col_worst = st.columns(2)
        with col_best:
            st.metric(" Best Selling Month", f"{best_month['month_year'].strftime('%B %Y')}", f"{best_month['sales']} units")
        with col_worst:
            st.metric(" Worst Selling Month", f"{worst_month['month_year'].strftime('%B %Y')}", f"{worst_month['sales']} units")

        st.markdown("---")

        # --- PLOTS (for total store sales) ---
        st.subheader("Daily Sales Trend")
        fig_daily = px.line(daily_sales, x='date', y='sales', title='Daily Total Sales Trend')
        st.plotly_chart(fig_daily, use_container_width=True)

        st.subheader("Monthly Sales Breakdown")
        fig_monthly = px.bar(monthly_sales, x='month_year', y='sales', title='Monthly Total Sales Volume')

        # Highlight best/worst months
        fig_monthly.add_scatter(
            x=[best_month['month_year'], worst_month['month_year']], 
            y=[best_month['sales'], worst_month['sales']],
            mode='markers+text', 
            text=["Best Month", "Worst Month"],
            textposition="top center",
            marker=dict(color=['green', 'red'], size=12)
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        st.markdown("---")

        ## 📊 Item-Level Contribution Analysis
        st.subheader("Top/Bottom 10 Items")
        item_sales_summary = filtered_df.groupby('item_id')['sales'].sum().reset_index().sort_values(by='sales', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top 10 Selling Items")
            top_10_items = item_sales_summary.head(10)
            fig_top = px.bar(top_10_items, x='item_id', y='sales', title=f'Top 10 Items in Store {store}')
            st.plotly_chart(fig_top, use_container_width=True)

        with col2:
            st.markdown("#### Bottom 10 Selling Items")
            bottom_10_items = item_sales_summary.tail(10).sort_values(by='sales', ascending=True)
            fig_bottom = px.bar(bottom_10_items, x='item_id', y='sales', title=f'Bottom 10 Items in Store {store}')
            st.plotly_chart(fig_bottom, use_container_width=True)    


# --------------------------------------------------------------
# MAIN NAVIGATION
# --------------------------------------------------------------

# MAIN NAVIGATION
if not st.session_state.get("authenticated", False):
    if st.session_state.get("page") == "register":
        register_page()
    else:
        login_page()
    st.stop()

# If logged in, show sidebar options
choice = st.sidebar.radio(
    "Choose Page",
    ["Forecasting", "Inventory Optimization", "Sales Analytics", "Clustering","Store Analytics"]
)

# st.sidebar.write("---")
# if st.sidebar.button("Logout"):
#     st.session_state["authenticated"] = False
#     st.session_state["page"] = "login"
#     st.rerun()

if st.sidebar.button("Logout", key="logout_main"):
    st.session_state["authenticated"] = False
    st.session_state["page"] = "login"
    st.rerun()


if choice == "Forecasting":
    forecasting_dashboard()
elif choice == "Inventory Optimization":
    inventory_optimization()
elif choice == "Sales Analytics":
    analytics_page()
elif choice == "Clustering":
    clustering_page()
elif choice == "Store Analytics":
    store_analytics_page()


