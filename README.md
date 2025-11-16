# AIML_Project
AI/ML solution for product-level demand forecasting and inventory optimization.

It is designed for retail businesses to predict future product demand, assess stockout risks, and recommend optimal inventory levels.
The system leverages historical sales data, product attributes, and external features (like holidays, price changes, and promotions) to provide actionable insights that help reduce overstocking and stockouts.

Tech Stack

    Language: Python 3.x
  
    Libraries: Pandas, NumPy, Plotly, Streamlit, XGBoost, LightGBM, scikit-learn
  
    Deployment: Streamlit Dashboard for interactive use
  
    Data: Historical sales datasets (M5-style + custom retail data)



Project Structure:

    ├── app.py                  # Main Streamlit application
  
    ├── forecasting_utils.py    # Recursive forecast functions
  
    ├── inventory_utils.py      # Stockout and inventory optimization functions
  
    ├── models/                 # Pre-trained ML models 
  
    ├── data/                   # Raw and processed sales datasets
  
    ├── requirements.txt        # Python dependencies
  
    └── README.md               # Project description and instructions
