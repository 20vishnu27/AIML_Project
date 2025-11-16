import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import time
import plotly.express as px

from forecasting_utils import recursive_forecast

# --------------------------------------------------------------
# GLOBAL CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="Retail Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state init
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "page" not in st.session_state:
    st.session_state["page"] = "login"
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "user_db" not in st.session_state:
    st.session_state["user_db"] = {
        "admin": "password123",
        "shruti": "projectai",
    }

USER_CREDENTIALS = st.session_state["user_db"]

# --------------------------------------------------------------
# LOAD RESOURCES
# --------------------------------------------------------------
@st.cache_resource
def load_all_data():
    """Load inventory dataset, forecasting dataset, and all models (with tuple fix)."""

    # ---- File paths ----
    inv_file = "data/data_stock.csv"
    stock_model_file = "models/xgb_stockout_model.pkl"

    forecast_file = "data/sales_data.csv"
    rf_file = "models/rf_model.pkl"
    xgb_file = "models/xgb_model.pkl"

    required_files = [
        inv_file, stock_model_file,
        forecast_file, rf_file, xgb_file
    ]

    # ---- Validate paths ----
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"❌ Missing required file: **{f}**")
            raise FileNotFoundError(f)

    # ---- Load datasets ----
    inv_df = pd.read_csv(inv_file)
    forecast_df = pd.read_csv(forecast_file)
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    # ---- Load stockout classifier ----
    raw_stock_model = joblib.load(stock_model_file)

    # FIX: extract model if saved as tuple
    stock_model = (
        raw_stock_model[0] if isinstance(raw_stock_model, tuple)
        else raw_stock_model
    )

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
    st.title("📦 Product-Level Demand Forecasting")

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

    # Ensure a 'sales' column exists
    if "sales" not in history.columns:
        if "units_sold" in history.columns:
            history["sales"] = history["units_sold"]
        else:
            history["sales"] = 0

    # Keep only necessary columns
    keep_cols = [
        "item_id", "dept_id", "cat_id", "store_id", "date",
        "sales", "sell_price", "snap_CA", "snap_TX", "snap_WI", "snap", "is_event",
        "wday", "month", "year", "wm_yr_wk"
    ]
    history = history[[c for c in keep_cols if c in history.columns]].copy()

    # ---- PRECOMPUTE LAGS AND ROLLING FEATURES ----
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

    # Ensure all model features exist
    for f in model.feature_names_in_:
        if f not in history.columns:
            history[f] = 0

    # Features used for prediction
    features = list(model.feature_names_in_)

    st.subheader(f"Forecasting for: **{item} | {store}**")

    if st.button("Generate Forecast"):
        # Call recursive forecast
        forecast = recursive_forecast(model, features, history, horizon)

        # Plot history + forecast
        fig = px.line()
        fig.add_scatter(x=history["date"], y=history["sales"], mode="lines", name="History")
        fig.add_scatter(x=forecast["date"], y=forecast["forecast"], mode="lines", name="Forecast")
        st.plotly_chart(fig, use_container_width=True)

        # Show forecast dataframe
        st.dataframe(forecast)

        # Download CSV
        csv = forecast.to_csv(index=False).encode()
        st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")


# --------------------------------------------------------------
# PAGE 2 — INVENTORY OPTIMIZATION
# --------------------------------------------------------------
def inventory_optimization():
    st.title("🛒 Inventory Optimization & Stockout Prediction")
    st.write(f"Welcome, **{st.session_state['username']}**")

    # Sidebar
    with st.sidebar:
        st.info(f"Logged in as: {st.session_state['username']}")
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.session_state["page"] = "login"
            st.rerun()

        st.header("Product Selection")

        item_id = st.selectbox("Item ID", sorted(inv_df["item_id"].unique()))
        current_stock = st.number_input("Current Stock", 0, value=100)
        lead_time = st.number_input("Supplier Lead Time (days)", 1, value=7)

    # Get last SKU record
    sku = inv_df[inv_df["item_id"] == item_id].tail(1)

    if st.button("Run Optimization"):
        # ---- Correct Stockout Features ----
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

        # Prepare input for stockout model
        X = sku.copy()
        X["current_stock"] = current_stock

        # Ensure future_demand_30 exists (fallback if missing)
        if "future_demand_30" not in X.columns:
            X["future_demand_30"] = sku["forecast_30"].values[0]

        # Ensure all features exist
        for f in stockout_features:
            if f not in X.columns:
                X[f] = 0

        # Select features in correct order
        X = X[stockout_features]

        # ---- Stockout Risk Prediction ----
        pred = int(stock_model.predict(X)[0])
        risk = risk_map[pred]

        # ---- Inventory Calculations ----
        future_demand_30 = X["future_demand_30"].values[0]
        rolling_std_30 = sku["rolling_std_30"].values[0]

        daily = future_demand_30 / 30
        safety = rolling_std_30 * 1.65
        reorder_point = daily * lead_time + safety
        reorder_qty = max(0, reorder_point - current_stock)

        # ---- DISPLAY ----
        st.subheader("📈 Stockout Risk")
        if pred == 2:
            st.error(f"HIGH RISK — {risk}")
        elif pred == 1:
            st.warning(f"MEDIUM RISK — {risk}")
        else:
            st.success(f"LOW RISK — {risk}")

        st.markdown("---")
        st.subheader("📦 Inventory Recommendation")

        st.write(f"Daily Demand: **{daily:.2f} units**")
        st.write(f"Safety Stock: **{safety:.0f} units**")
        st.write(f"Reorder Point: **{reorder_point:.0f} units**")

        if reorder_qty > 0:
            st.error(f"REORDER REQUIRED → **Order {reorder_qty:.0f} units**")
        else:
            st.success("Stock sufficient. No reorder required.")

        # Chart
        chart = pd.DataFrame({
            "Units": [current_stock, reorder_point, future_demand_30]
        }, index=["Current Stock", "Reorder Point", "30-day Forecast"])
        st.bar_chart(chart)


# --------------------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------------------
def login_page():

    st.title("Login")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    if st.button("Login"):
        if user in USER_CREDENTIALS and USER_CREDENTIALS[user] == pw:
            st.session_state["authenticated"] = True
            st.session_state["page"] = "forecasting"
            st.session_state["username"] = user
            st.rerun()
        else:
            st.error("Invalid credentials")


# --------------------------------------------------------------
# MAIN NAVIGATION
# --------------------------------------------------------------
if st.session_state["authenticated"]:

    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose Page", ["Forecasting", "Inventory Optimization"])

    if choice == "Forecasting":
        forecasting_dashboard()

    elif choice == "Inventory Optimization":
        inventory_optimization()

else:
    login_page()
