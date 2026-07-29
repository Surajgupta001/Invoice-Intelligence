# Invoice Intelligence Dashboard

An AI-powered platform that leverages machine learning for intelligent invoice processing and freight cost prediction to improve financial decision-making.

## 🚀 Key Features

- **Predict Freight Cost**: Automatically calculate expected shipping costs based on invoice values
- **Detect High-Risk Vendor Invoices**: Identify invoices that require manual approval vs automatic approval
- **Reduce Financial Leakage**: Minimize manual errors and fraud through ML-powered intelligent processing
- **Speed Up Invoice Processing**: Automated decision-making for faster financial workflows
- **Improve Cost Forecasting**: Data-driven insights for better financial planning

## 📋 Overview

The Invoice Intelligence Dashboard is a web-based application built with Streamlit that provides:

- **Predict Freight Cost**: Use the trained Linear Regression model to predict freight costs based on invoice amounts
- **Invoice Risk Assessment**: Apply a Random Forest Classifier to determine if invoices should be manually reviewed or automatically approved based on multiple risk factors

## 🛠 Libraries & Dependencies

- **Streamlit**: Web application framework for the user interface
- **Scikit-learn**: Machine learning algorithms (Linear Regression, Random Forest)
- **Plotly**: Interactive data visualizations and charts
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib/Seaborn**: Statistical plotting and data visualization
- **SciPy**: Scientific computing utilities
- **IPython/Jupyter**: Interactive notebook support

## 📁 Project Structure

```
Invoice-Intelligence/
├── app.py                    # Streamlit web application
├── data/
│   ├── inventory.db         # SQLite database with vendor invoice data
│   └── README.md           # Database documentation
├── Inference/                # ML model inference utilities
│   ├── predict_freight.py   # Freight cost prediction module
│   └── predict_invoice_flag.py # Invoice flagging prediction module
├── models/                   # Trained ML models
│   ├── predict_freight_cost_model.pkl  # Trained regression model
│   ├── invoice_flagging_model.pkl      # Trained classification model
│   └── scaler.pkl          # Data preprocessing scaler
├── notebooks/                # Jupyter notebooks for analysis and experimentation
├── pyproject.toml           # Project configuration and dependencies
├── uv.lock                  # Dependency lock file
└── .python-version          # Required Python version
```

## 🏃 How to Use

### Prerequisites
- Python 3.11 or higher
- virtual environment (recommended)

### Installation
1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Unix/Linux/macOS
   ```
3. Install dependencies:
   ```bash
   pip install -r pyproject.toml
   ```

### Running the Application
```bash
streamlit run app.py
```

### Application Usage

1. **Select Prediction Module** from the sidebar:
   - 🚚 **Freight Cost Prediction**: Predict shipping costs based on invoice amounts
   - 🚩 **Invoice Manual Approval**: Assess invoice risk and determine approval status

2. **Freight Cost Prediction**:
   - Enter an **Invoice Amount ($)** to get the predicted freight cost
   - Click **Predict Freight Cost** to get the estimate

3. **Invoice Manual Approval**:
   - Fill in the five input parameters:
     - 📦 **Invoice Quantity**: Number of items in the invoice
     - 💰 **Invoice Amount ($)**: Total value of the invoice
     - 🚚 **Freight**: Shipping costs
     - 📊 **Total Item Quantity**: Combined quantity of all items
     - 💰 **Total Item Amount ($)**: Total value of all items
   - Click **Predict Invoice Flag** to check if manual approval is required

### Models and Training

The application uses two pre-trained machine learning models:

1. **Linear Regression Model**: Predicts freight costs based on invoice amounts
2. **Random Forest Classifier**: Flagging invoices for manual review based on multiple risk factors

Both models were trained using data from the `data/inventory.db` SQLite database and saved using joblib for deployment.

## 📊 Example Predictions

### Freight Cost Prediction
- Input: Invoice Amount = $18,500
- Predicted Freight Cost: $98.00

### Invoice Manual Approval
- Input: Various quantities and amounts
- Output: **Approved** (✅) or **Manual Approval Required** (🚩)

## 💡 Technical Notes

- The database contains vendor invoice information with features like quantity, amounts, and approval status
- Models were selected based on best performance metrics (MAE for regression, F1-score for classification)
- The application provides business impact analysis and risk assessment for financial decision-making
- All predictions are rounded to the nearest whole number for practical use

## 🤝 Contributing


This project was developed as part of a financial technology initiative to automate invoice processing and improve cost visibility. Contributions are welcome to enhance the ML models, add new features, or improve the user interface.

## 📄 License

Commercial license for enterprise use.

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure all .pkl files are present in the models folder
2. **Database Connection**: Verify the data/inventory.db file exists
3. **Performance**: The application requires Python 3.11+ for optimal performance

### Getting Help

- Check the models folder for any missing prediction model files
- Verify virtual environment activation
- Review error messages in the browser console

## 📈 Future Enhancements

- Add time-series forecasting for trend analysis
- Implement anomaly detection for outlier identification
- Add ensemble models for improved accuracy
- Include explainability features for model decisions
- Add API endpoints for integration with existing systems

## 📞 Contact

For technical support or questions, please refer to the project repository or contact the development team.
