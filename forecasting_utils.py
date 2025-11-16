# forecasting_utils.py
import pandas as pd
import numpy as np

def recursive_forecast(model, feature_list, history_df, horizon=30):
    """
    Recursive multi-step forecasting for one item-store history.
    - model: trained regression model (sklearn/xgboost/lightgbm)
    - feature_list: list of features used in training (must match model)
    - history_df: historical rows (pandas DataFrame) for one item-store sorted by date ascending
    - horizon: days to forecast

    Returns: DataFrame with columns ['date','forecast'] length = horizon
    """
    if isinstance(history_df, list):
        history_df = pd.DataFrame(history_df)
    
    hist = history_df.copy().reset_index(drop=True)
    # Ensure numeric index and columns exist
    forecasts = []
    # We'll keep a rolling buffer of past sales to update lags/rolling
    for step in range(horizon):
        # Build next-day feature row using last row of hist
        last = hist.iloc[-1].copy()
        # compute next date if 'date' exists
        if 'date' in last.index:
            next_date = pd.to_datetime(last['date']) + pd.Timedelta(days=1)
        else:
            next_date = pd.NaT

        # Prepare new_row dict - start with last row so static fields carry over
        new_row = last.to_dict()
        if pd.notna(next_date):
            new_row['date'] = next_date
            # update time features if present
            if 'wday' in feature_list:
                new_row['wday'] = next_date.weekday()
            if 'weekday' in feature_list:
                # keep string weekday if used
                new_row['weekday'] = next_date.strftime('%A')
            if 'month' in feature_list:
                new_row['month'] = next_date.month
            if 'year' in feature_list:
                new_row['year'] = next_date.year
            if 'wm_yr_wk' in feature_list:
                new_row['wm_yr_wk'] = int(next_date.strftime("%U"))

        # Update lag features using hist sales
        # Require that hist has 'sales' column present
        sales_series = hist['sales'].astype(float).reset_index(drop=True)
        new_row['lag_1'] = float(sales_series.iloc[-1]) if len(sales_series) >= 1 else 0.0
        new_row['lag_7'] = float(sales_series.iloc[-7]) if len(sales_series) >= 7 else new_row['lag_1']
        new_row['lag_14'] = float(sales_series.iloc[-14]) if len(sales_series) >= 14 else new_row['lag_1']
        new_row['lag_28'] = float(sales_series.iloc[-28]) if len(sales_series) >= 28 else new_row['lag_1']

        # Rolling windows
        for win in (7,30,60):
            col_mean = f'rolling_mean_{win}'
            col_std  = f'rolling_std_{win}'
            if len(sales_series) >= 1:
                new_row[col_mean] = float(sales_series.tail(win).mean()) if len(sales_series) >= 1 else 0.0
                new_row[col_std]  = float(sales_series.tail(win).std()) if len(sales_series) >= 2 else 0.0
            else:
                new_row[col_mean] = 0.0
                new_row[col_std] = 0.0

        # Price rolling mean and ratio if present
        if 'sell_price' in hist.columns:
            price_series = hist['sell_price'].astype(float).reset_index(drop=True)
            new_row['rolling_price_mean_30'] = float(price_series.tail(30).mean()) if len(price_series) >= 1 else new_row.get('sell_price',0.0)
            denom = new_row['rolling_price_mean_30'] if new_row['rolling_price_mean_30'] != 0 else 1.0
            new_row['price_ratio'] = float(new_row.get('sell_price', denom)) / denom

        # Ensure all features in feature_list exist, fill missing with 0
        model_input = {}
        for f in feature_list:
            model_input[f] = new_row.get(f, 0.0)

        # Convert to DataFrame row for model
        X = pd.DataFrame([model_input])

        # Predict
        yhat = model.predict(X)[0]
        # Store prediction
        forecasts.append({'date': next_date, 'forecast': float(yhat)})

        # Append predicted row to hist for next iteration: create a new row with 'sales'=pred
        appended = new_row.copy()
        appended['sales'] = float(yhat)
        hist = pd.concat([hist, pd.DataFrame([appended])], ignore_index=True)

    return pd.DataFrame(forecasts)
