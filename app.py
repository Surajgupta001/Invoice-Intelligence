import streamlit as st
import pandas as pd

from Inference.predict_freight import predict_freight_cost
from Inference.predict_invoice_flag import predict_invoice_flag

# Set page configuration
st.set_page_config(
    page_title="Invoice Intelligence Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

# Header Section
st.title("📊 Invoice Intelligence Dashboard")

st.markdown("""
### 🤖 AI-Powered Invoice Analytics Platform

Leverage Machine Learning to improve financial decision-making.

### 🚀 Key Features

- 📦 Predict Freight Cost
- 🚩 Detect High-Risk Vendor Invoices
- 💰 Reduce Financial Leakage
- ⚡ Speed Up Invoice Processing
- 📈 Improve Cost Forecasting
""")

st.divider()

# Sidebar
st.sidebar.title("🧠 ML Model Selection")
selected_model = st.sidebar.radio(
    "📌 Select Prediction Module",
    ["🚚 Freight Cost Prediction", "🚩 Invoice Manual Approval"],
)

st.sidebar.markdown("---")

st.sidebar.success("""
## 💼 Business Impact

✔ Better Cost Forecasting

✔ Detect Invoice Fraud

✔ Reduce Manual Review

✔ Faster Financial Decisions

✔ Improve Operational Efficiency
""")

# Freight Cost Prediction Section
if selected_model == "🚚 Freight Cost Prediction":
    st.header("🚚 Freight Cost Prediction")
    st.info("""
Enter shipment information below to estimate the expected freight cost.
""")

    with st.form("freight_cost_form"):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input("📦 Quantity", min_value=1, value=1200)

        with col2:
            dollars = st.number_input(
                "💰 Invoice Amount ($)", min_value=1.0, value=18500.0
            )

        submitted = st.form_submit_button("🚀 Predict Freight Cost")

    if submitted:
        input_data = {
            "Dollars": [dollars],
        }

        prediction = predict_freight_cost(input_data)[0]
        st.success("✅ Prediction Completed Successfully!")

        st.metric("💰 Estimated Freight Cost", f"${prediction:,.2f}")
        st.balloons()

        st.markdown("### 📋 Input Summary")

        summary = pd.DataFrame(
            {"Feature": ["Quantity", "Invoice Amount"], "Value": [quantity, dollars]}
        )

        st.dataframe(summary, use_container_width=True)

# Invoice Flagging Section
else:
    st.header("🚩 Invoice Manual Approval Prediction")

    st.info("""
Predict whether an invoice should be automatically approved or sent for manual review.
""")

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(
                "📦 Invoice Quantity", min_value=1, value=100
            )
            invoice_dollars = st.number_input(
                "💰 Invoice Amount ($)", min_value=1.0, value=5000.0
            )

        with col2:
            freight = st.number_input("🚚 Freight", min_value=0.0, value=150.0)
            total_item_quantity = st.number_input(
                "📊 Total Item Quantity", min_value=1, value=100
            )

        with col3:
            total_item_dollars = st.number_input(
                "💰 Total Item Amount ($)", min_value=1.0, value=5000.0
            )

        submitted_flag = st.form_submit_button("Predict Invoice Flag")

    if submitted_flag:
        input_data_flag = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars],
        }

        prediction_flag = predict_invoice_flag(input_data_flag)["Predicted_Invoice_Flag"].iloc[0]
        if prediction_flag == 1:
            st.error("🚩 Manual Approval Required")

            st.metric("Invoice Status", "🚩 High Risk")

        else:
            st.success("✅ Invoice Approved")
            st.metric("Invoice Status", "✅ Low Risk")

        st.markdown("### 📋 Input Summary")

        summary = pd.DataFrame(
            {
                "Feature": [
                    "Invoice Quantity",
                    "Invoice Amount",
                    "Freight",
                    "Total Item Quantity",
                    "Total Item Amount",
                ],
                "Value": [
                    invoice_quantity,
                    invoice_dollars,
                    freight,
                    total_item_quantity,
                    total_item_dollars,
                ],
            }
        )

        st.dataframe(summary, use_container_width=True)

st.divider()

st.caption(
    "📊 Invoice Intelligence Dashboard | Built with Streamlit, Scikit-learn & Plotly ❤️"
)
