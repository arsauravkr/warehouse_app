import streamlit as st
import pandas as pd

from src.model_shap import train_and_explain_with_shap

st.set_page_config(page_title="Warehouse Demand Forecaster", layout="wide")
st.title("🚚 Warehouse Demand Forecaster")

st.markdown("""
Upload your warehouse CSV or let me load the default one from `data/raw_data.csv`.  
I’ll train an XGBoost model under the hood and show you predictions and a SHAP plot of feature importance.
""")

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv("data/raw_data.csv")

uploaded = st.file_uploader("🔄 Upload CSV", type="csv")
df = load_data(uploaded)

if st.button("Train & Explain"):
    with st.spinner("Training XGBoost and computing SHAP…"):
        model, prep, explainer = train_and_explain_with_shap(df)

    st.success("✅ Done!")

    # Predictions
    X = df.drop(columns=["product_wg_ton"])
    X_proc = prep.transform(X)
    preds = model.predict(X_proc)
    df_out = df.copy()
    df_out["Predicted Demand"] = preds

    st.subheader("Sample of Predictions")
    st.dataframe(df_out.head(), use_container_width=True)

    st.subheader("SHAP Feature Importance")
    st.image("shap_summary.png", use_container_width=True)