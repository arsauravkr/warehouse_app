import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.model_shap import train_and_explain_with_shap
from src.preprocess import Preprocessor

st.set_page_config(page_title="Warehouse Demand Forecaster", layout="wide")
st.title("🚚 Supply Chain Demand Forecaster")

st.markdown("""
Upload your warehouse CSV or let me load the default one from `data/raw_data.csv`.  
I’ll train an XGBoost model under the hood and show you predictions and a SHAP plot of feature importance.
""")

@st.cache_data(show_spinner=False)
def load_data(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    return pd.read_csv("data/raw_data.csv")

uploaded = st.file_uploader("🔄 Upload CSV", type="csv")
df = load_data(uploaded)

if st.button("Train & Explain"):
    with st.spinner("Training XGBoost and computing SHAP…"):
        model, prep, explainer = train_and_explain_with_shap(
            csv_path=None  # we’ll bypass file read inside; pass df directly
        )
    st.success("✅ Done!")

    # Prepare data for prediction
    X = df.drop(columns=["product_wg_ton"])
    X_proc = prep.transform(X)
    preds = model.predict(X_proc)
    df_out = df.copy()
    df_out["Predicted Demand"] = preds

    st.subheader("Sample of Predictions")
    st.dataframe(df_out.head(), use_container_width=True)

    st.subheader("SHAP Feature Importance")
    # SHAP plot was saved as 'shap_summary.png' by model_shap.py
    st.image("shap_summary.png", use_column_width=True)