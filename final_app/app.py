import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

# Load Scalers
rob_scaler = RobustScaler()

# Set page layout and title
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size:32px; 
        color:#0047AB; 
        font-weight: bold; 
        text-align:center;
        margin-bottom: 20px;
    }
    .description-text {
        font-size:18px; 
        color:#333333; 
        text-align: justify;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 24px; 
        font-weight: bold; 
        color: #8B0000; 
        margin-top: 30px;
        border-bottom: 2px solid #8B0000; 
        padding-bottom: 10px;
    }
    .stButton>button {
        background-color: #FFA07A;
        color: white;
        padding: 10px 25px;
        border: none;
        border-radius: 8px;
        margin-right: 10px;
        margin-top: 10px;
    }
    .output-box {
        background-color: #F8F9FA; 
        padding: 20px;
        border-radius: 10px; 
        margin-top: 20px;
        border: 1px solid #E9ECEF;
        font-size: 18px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App header
st.markdown('<div class="main-header">Credit Card Fraud Detection Web App</div>', unsafe_allow_html=True)

# Updated App description
st.markdown("""
<div class="description-text">
    Credit card fraud detection involves identifying fraudulent credit card transactions based on patterns and behaviors within the transaction data. 
    The goal of this project is to predict whether a given transaction is fraudulent or not using machine learning algorithms. We have used 
    Logistic Regression, Decision Tree, and Random Forest classifiers with techniques such as oversampling and undersampling to balance the dataset.
</div>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    df.drop_duplicates(inplace=True)
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    scaled_amount = df['scaled_amount']
    df.drop(['scaled_amount'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    return df

df = load_data()
X = df.drop(['Class'], axis=1)
model = joblib.load("final_model.pkl")

# Data Visualization section
st.markdown('<div class="section-header">Data Visualizations</div>', unsafe_allow_html=True)

# 1. Correlation Heatmap
st.markdown('<div class="description-text">Correlation heatmap showing relationships between the features in the dataset.</div>', unsafe_allow_html=True)
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
st.pyplot(plt)
plt.clf()  # Clear the figure after plotting

# 2. Fraud vs. Normal Transactions
st.markdown('<div class="description-text">Distribution of fraudulent and normal transactions in the dataset.</div>', unsafe_allow_html=True)
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df, palette=['#1f77b4', '#ff7f0e'])
plt.title('Fraud vs. Normal Transactions')
plt.xlabel('Transaction Class (0: Normal, 1: Fraud)')
plt.ylabel('Count')
st.pyplot(plt)
plt.clf()



# Input form
st.markdown('<div class="section-header">Input Transaction Details</div>', unsafe_allow_html=True)

# Collect user input in 3 columns
col1, col2, col3 = st.columns(3)

user_input = []

# First column inputs
with col1:
    amount = st.number_input("Enter amount", format="%.2f")
    user_input.append(amount)
    for i in range(1, 10):
        v_value = st.number_input(f"Enter value of V{i}", format="%.5f")
        user_input.append(v_value)

# Second column inputs
with col2:
    for i in range(10, 19):
        v_value = st.number_input(f"Enter value of V{i}", format="%.5f")
        user_input.append(v_value)

# Third column inputs
with col3:
    for i in range(19, 29):
        v_value = st.number_input(f"Enter value of V{i}", format="%.5f")
        user_input.append(v_value)

user_input = list(np.float64(user_input))

# Buttons placed side by side with gaps
st.markdown('<div class="section-header">Run Default Checks</div>', unsafe_allow_html=True)

button_col1, button_col2, button_col3 = st.columns(3)
with button_col1:
    if st.button("Check with default fraud transaction"):
        prediction = model.predict([X.loc[61787]])[0]
        output_msg = 'Fraudulent Transaction' if prediction else 'Normal Transaction'
        st.markdown(f'<div class="output-box">{output_msg}</div>', unsafe_allow_html=True)

with button_col3:
    if st.button("Check with default legit transaction"):
        prediction = model.predict([X.loc[20156]])[0]
        output_msg = 'Fraudulent Transaction' if prediction else 'Normal Transaction'
        st.markdown(f'<div class="output-box">{output_msg}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)
button_col1, button_col2, button_col3 = st.columns(3)
with button_col2:
    if st.button("Predict Based on Input"):
        user_input_scaled = user_input.copy()
        # Scale the amount using the same scaler
        # user_input_scaled[0] = rob_scaler.transform([[user_input_scaled]])
        prediction = model.predict([user_input_scaled])
        output_msg = 'Fraudulent Transaction' if prediction else 'Normal Transaction'
        st.markdown(f'<div class="output-box">{output_msg}</div>', unsafe_allow_html=True)

# Footer description
st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)

st.markdown("""
<div class="description-text">
    This web app demonstrates how machine learning models can be used to detect fraudulent transactions by analyzing patterns in transaction data. 
    The dataset used consists of anonymized features due to the sensitive nature of the information. For this project, we applied scaling using 
    the Robust Scaler to minimize the effect of outliers on the models.
</div>
""", unsafe_allow_html=True)
