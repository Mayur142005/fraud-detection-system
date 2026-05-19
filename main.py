import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Fraud Detection System", 
    layout="wide"
)

st.title("Credit Card Fraud Detection System")

st.write("Upload transaction dataset for fraud prediction")

# Load Model
model = pickle.load(open('models/fraud_model.pkl', 'rb'))

uploaded_file = st.file_uploader(
    "Choose a CSV file", 
    type="csv"
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Upload Data")
    st.dataframe(data.head())

    if "Class" in data.columns:
        data = data.drop("Class", axis=1)
    
    if hasattr(model, "feature_names_in_"):
        data = data[model.feature_names_in_]
        
    predictions = model.predict(data)
    data['Prediction'] = predictions
    st.subheader("Predictions Results")
    st.dataframe(data.head())

    fraud_count = data['Prediction'].sum()

    st.metric(
        label="Detected Fraud Transactions",
        value=int(fraud_count)
    )

    csv = data.to_csv(index=False)

    st.download_button(
        label="Download Results",
        data=csv,
        file_name='fraud_predictions.csv',
        mime='text/csv'
    )


