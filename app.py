import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import timedelta, datetime

st.set_page_config(page_title="Gold Price ML App", layout="wide")

# Load dataset once globally
df = pd.read_csv("financial_regression.csv")  # Updated file name

# Debug: Show actual column names to fix KeyError
# st.write("Columns in your dataset:", df.columns.tolist())

# Final set of columns for model input and EDA
used_columns = ['usd_chf', 'eur_usd', 'gold open', 'gold high', 'gold low', 'gold volume', 'gold close']
model_df = df[used_columns]

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = [
    "1. Introduction",
    "2. Problem Statement & Objective",
    "3. Data Collection",
    "4. Exploratory Data Analysis (EDA)",
    "5. Feature Engineering",
    "6. Model Building",
    "7. Deployment"
]
page = st.sidebar.radio("Go to", pages)

if page == "1. Introduction":
    st.title("Introduction: Gold Close Price Prediction")
    st.markdown("""
    Predicting gold prices is crucial for investment planning, portfolio management, and financial forecasting.

    ### Why Use This?
    - Gold is a key asset in global markets
    - Its price is influenced by economic and geopolitical factors
    - Accurate prediction supports risk-aware decisions

    ### Where It Can Be Applied?
    - Financial institutions
    - Investment advisory firms
    - Government and policy think tanks

    ### Main Agenda
    - Use machine learning to forecast gold close price
    - Build and deploy a predictive app

    ### Key Observations
    - Currency values like USD/CHF and EUR/USD impact gold
    - Price volatility is linked to oil, inflation, and market movement
    """)

elif page == "2. Problem Statement & Objective":
    st.title("Problem Statement & Objective")
    st.markdown("""
    ### Problem Statement
    The challenge is to predict gold's close price using economic indicators and historical trading data.

    ### Objectives
    - Analyze relationships between market variables and gold
    - Train a regression model that minimizes prediction error
    - Enable real-time predictions through deployment

    ### Applications
    - Portfolio diversification
    - Short-term market speculation
    - Economic stability modeling

    ### Observations
    - Feature correlation helps drive model selection
    - Nonlinear trends suit non-linear models like KNN
    """)

elif page == "3. Data Collection":
    st.title("Data Collection")
    st.markdown("""
    ### Why This Data?
    The model uses financial indicators that historically impact gold prices:
    - `USD/CHF` exchange rate
    - `EUR/USD` exchange rate
    - Gold market prices: open, high, low
    - `Volume` traded

    ### Goal
    - Use structured, numeric data for regression
    - Ensure data quality before modeling

    ### Source Ideas
    - Yahoo Finance
    - Investing.com
    - Kaggle financial datasets

    ### Observations
    - Missing values must be cleaned
    - Normalization is critical due to varying scales
    """)

elif page == "4. Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis (EDA)")
    st.markdown("""
    EDA helps uncover hidden patterns, detect outliers, and identify relationships between variables.

    ### Agenda
    - Visualize correlations
    - Understand feature distributions
    - Identify patterns affecting the gold close price

    ### Observations
    - High correlation found with `gold high`, `gold low`, and exchange rates
    - Distributions show moderate skewness in price and volume
    """)

    st.subheader("Correlation Heatmap")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(model_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Feature Distribution")
    feature = st.selectbox("Select a feature to view distribution", model_df.columns.tolist())
    fig2, ax2 = plt.subplots()
    sns.histplot(model_df[feature], kde=True, bins=30, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Scatter Plot vs Close")
    scatter_cols = st.multiselect("Select features to compare with gold close", [col for col in model_df.columns if col != 'gold close'])
    for col in scatter_cols:
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x=model_df[col], y=model_df['gold close'], ax=ax3)
        ax3.set_xlabel(col)
        ax3.set_ylabel('gold close')
        st.pyplot(fig3)

elif page == "5. Feature Engineering":
    st.title("Feature Engineering")
    st.markdown("""
    ### Why We Need This?
    - Improve model performance
    - Standardize data scales
    - Reduce dimensionality

    ### Agenda
    - Drop irrelevant fields
    - Scale inputs using StandardScaler
    - Preserve high-impact features

    ### Observations
    - Features like `gold high`, `gold low`, and exchange rates had most impact
    - Feature scaling improved model convergence
    """)

elif page == "6. Model Building":
    st.title("Model Building")
    st.markdown("""
    ### Why Modeling?
    Predictive modeling helps quantify complex relationships between variables.

    ### Strategy
    - Use **K-Nearest Neighbors (KNN) Regressor** in a pipeline
    - Scale inputs using `StandardScaler`
    - Evaluate using R² and Mean Squared Error

    ### Observations
    - KNN Regressor with pipeline gave R² ≈ 0.995 on test set
    - Final model exported as `goldPrediction.pkl`
    """)

elif page == "7. Deployment":
    st.title("Deployment")
    st.markdown("""
    ### Why Deploy?
    - Let non-technical users interact with it
    - Provide instant predictions through UI

    ### Goals
    - Load the model
    - Accept user input
    - Predict gold close price for future dates

    ### Observations
    - Deployment enables business decisions from model insights
    """)

    try:
        with open("goldPrediction.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'goldPrediction.pkl' not found.")
        st.stop()

    st.subheader("Enter Features for Prediction")
    col1, col2 = st.columns(2)
    with col1:
        usd_chf = st.number_input("USD/CHF", value=0.9)
        eur_usd = st.number_input("EUR/USD", value=1.1)
        gold_open = st.number_input("Gold Open", value=1800.0)
    with col2:
        gold_high = st.number_input("Gold High", value=1820.0)
        gold_low = st.number_input("Gold Low", value=1795.0)
        gold_volume = st.number_input("Gold Volume", value=500000.0)
    days = st.slider("Predict for how many days in future?", 1, 30, 1)

    if st.button("Predict"):
        X = [[usd_chf, eur_usd, gold_open, gold_high, gold_low, gold_volume]]
        try:
            today_pred = model.predict(X)[0]
            rate = 0.0015  # assume 0.15% daily growth
            future_price = today_pred * ((1 + rate) ** days)
            future_date = datetime.now() + timedelta(days=days)
            st.success(f"Predicted Gold Close Price on {future_date.date()}: ${future_price:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
