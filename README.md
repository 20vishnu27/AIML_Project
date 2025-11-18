1. Project Title & Objective
Project Title:

Smart Retail Analytics – Inventory Optimization and Stock Prediction

Objective:

- To build an AI-powered retail inventory management system that:

- Accurately forecasts 30-day SKU-level demand

- Predicts stockout risk levels (Low/Medium/High)

- Detects overstock conditions

- Recommends optimal reorder quantities to minimize losses

Purpose:

- Retailers face major challenges like unpredictable demand, stockouts, overstocks, and high holding costs.
- This system uses machine learning to provide data-driven, proactive, and automated inventory decisions.

2. Dataset Details
Source Datasets

- M5 Walmart Sales Dataset – Daily SKU-level sales

- Calendar Dataset – Events, holidays, SNAP

- Sell Price Dataset

- Dataset Preparation Pipeline

- Filtered for one store (CA_1)

- Merged sales + calendar + price

- Transformed 1,941 wide-format sales columns (d_1 to d_1941) into time-series

- Cleaned missing values and unified date format

- Engineered 40+ meaningful features including:

 - Lag features (lag_7, lag_28)

- Rolling means (r7, r30, r60)

- Prices, promotions, events

- Forecasted 30-day demand

- Current stock

- Risk labels (Low/Medium/High)

3. Algorithms / Models Used
Demand Forecasting

- Decision Tree Regressor – Baseline model
- XGBoost Regressor 
- Random Forest Regressor – Reduced overfitting, improved stability (Best Performer)

  

- Handles seasonality & non-linear behavior

- Achieved lowest MAE/RMSE

- Stockout Risk Classification

- Logistic Regression – Baseline classification

XGBoost Classifier (Best Performer)

- Robust to imbalance

- High predictive accuracy (~99%)

Overstock Detection

 - Rule-based comparison of current stock vs. 30-day forecasted demand

Inventory Optimization Engine

Calculates:

- Safety Stock (SS)

- Reorder Point (ROP)

-  Reorder Quantity (ROQ)

Provides actionable SKU-wise recommendations

4. Results 
Model Performance

Forecasting Performance (XGBoost Regressor):

- Lowest errors among all models

- Captured patterns, trends, seasonality effectively

 Classification Performance (XGBoost Classifier):

- ~99% accuracy

- Strong separation of risk classes

- High recall for stockout scenarios

- System Output Includes:

- Accurate 30-day demand forecast

- Stockout Risk Label – Low / Medium / High

- Reorder Point (ROP)

- Reorder Quantity (ROQ)

Overstock Alerts

- Line charts for SKU demand trends

- Inventory vs. ROP Visualization

  <img width="1651" height="622" alt="image" src="https://github.com/user-attachments/assets/bf93765a-5067-4e9f-85cf-2ba736dc09ad" />


5. Conclusion

- Developed a fully functional ML-powered inventory optimization pipeline

- Generated highly accurate forecasting results and robust stockout risk predictions

- Automated the generation of reorder quantities, reducing manual errors

- Provides significant benefits such as:

- Fewer stockouts

 - Lower inventory carrying costs

 - Improved SKU availability

- Better planning for promotions/events

6. Future Scope

 - Reinforcement Learning (RL) for dynamic ordering based on costs

- Deep Learning (LSTM/Transformer Models) for real-time demand prediction

- Live POS Data Integration

 - Cloud Deployment for enterprise-wide multi-store management

- Dashboard Integration (Streamlit/React) for interactive insights

7. References

- Walmart M5 Forecasting Competition Dataset

- XGBoost Documentation

- Scikit-learn ML Library

- Papers on Retail Demand Forecasting & Inventory Control

- Academic references used in project PDF
