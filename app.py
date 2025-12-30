import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="UPI Fraud Detection System",
    layout="wide"
)

# ============================================================
# BASIC STYLING
# ============================================================
st.markdown("""
<style>
.main { background-color: #F9FAFB; }
h1, h2, h3 { color: #111827; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ARTIFACTS (SAFE)
# ============================================================
@st.cache_resource
def load_artifacts():
    model_path = "fraud_model (1).pkl"
    scaler_path = "scaler.pkl"
    encoder_path = "label_encoders.pkl"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error("Scaler file not found")
        st.stop()
    if not os.path.exists(encoder_path):
        st.error("Label encoders file not found")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoder_path)

    return model, scaler, label_encoders

model, scaler, label_encoders = load_artifacts()

# ============================================================
# LOAD DATA (FOR UI OPTIONS)
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("upi_transactions_2024.csv")

df = load_data()

TARGET_COL = "fraud_flag"

# ============================================================
# FEATURE GROUPS
# ============================================================
raw_features = [c for c in df.columns if c != TARGET_COL]

numeric_features = df[raw_features].select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = list(label_encoders.keys())

trained_features = model.feature_names_in_.tolist()


# ============================================================
# HEADER
# ============================================================
st.title("UPI Fraud Detection System")
st.markdown(
    "Predict whether a UPI transaction is **Fraudulent** or **Legitimate** "
    "using a trained Machine Learning model."
)
st.markdown("---")

left_col, right_col = st.columns([1.3, 1])

# ============================================================
# INPUT FORM
# ============================================================
with left_col:
    st.subheader("Transaction Details")

    with st.form("transaction_form"):
        user_input = {}

        # Numeric inputs
        for col in numeric_features:
            user_input[col] = st.number_input(
                label=col,
                value=float(df[col].median())
            )

        # Categorical inputs
        for col in categorical_features:
            user_input[col] = st.selectbox(
                label=col,
                options=sorted(df[col].dropna().unique())
            )

        submit = st.form_submit_button("Predict Transaction")

# ============================================================
# PREDICTION
# ============================================================
with right_col:
    st.subheader("Prediction Result")

    if submit:
        # Create DataFrame
        input_df = pd.DataFrame([user_input])

        # ---------------- Label Encoding (SAFE) ----------------
        for col in categorical_features:
            le = label_encoders[col]
            value = input_df.at[0, col]

            if value not in le.classes_:
                st.warning(
                    f"Unseen category '{value}' for '{col}'. Using most frequent value."
                )
                value = le.classes_[0]

            input_df[col] = le.transform([value])[0]

        # ---------------- Feature Scaling ----------------
        input_df[numeric_features] = scaler.transform(
            input_df[numeric_features]
        )

        # ---------------- Align Feature Order ----------------
        input_df = input_df[trained_features]

        # ---------------- Prediction ----------------
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")

        # ================= CORE RESULT =================
        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

        st.metric("Fraud Probability", f"{probability:.2%}")

        # ================= ADDITIONAL ANALYSIS =================

        # Risk Level
        if probability < 0.05:
            risk_level = "Low Risk"
            action = "Transaction can be safely approved."
        elif probability < 0.20:
            risk_level = "Medium Risk"
            action = "Transaction should be monitored or verified."
        else:
            risk_level = "High Risk"
            action = "Transaction should be blocked or manually reviewed."

        st.markdown("### Risk Assessment")
        st.write(f"**Risk Level:** {risk_level}")
        st.write(f"**Recommended Action:** {action}")

        # Confidence Score
        confidence = abs(probability - 0.5) * 2
        st.markdown("### Model Confidence")
        st.progress(min(confidence, 1.0))
        st.write(f"Confidence Score: {confidence:.2%}")

        # Transaction Summary
        st.markdown("### Transaction Summary")
        summary_df = pd.DataFrame(
            user_input.items(),
            columns=["Feature", "Value"]
        )
        st.dataframe(summary_df, use_container_width=True)

    else:
        st.info("Fill the form and click **Predict Transaction**")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("UPI Fraud Detection | Streamlit Deployment")
