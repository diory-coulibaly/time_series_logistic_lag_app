
# Final Version: Streamlit App for GOOG Lag Prediction
# Includes: TimeSeriesSplit, VIF, SHAP, LIME, Autocorrelation, ADF Test, Seasonality
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import shap
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="GOOG Lag Prediction", layout="wide")
st.title("📈 GOOG Direction Prediction with Enhanced Time Series Analysis")
# 📷 Quick preview of expected CSV format
from PIL import Image
image = Image.open("Goog.JPG")
st.image(image, caption="CSV Format: Stocks,SP500", use_container_width=True, output_format="JPEG")


with st.sidebar:
    st.header("1. Upload Files")
    goog_file = st.file_uploader("GOOG CSV (semicolon-separated)", type="csv")
    sp500_file = st.file_uploader("S&P500 CSV (semicolon-separated)", type="csv")
    st.header("2. Model Options")
    lags = st.multiselect("Select lags (days)", [1, 2, 3, 5, 10], default=[1, 2, 5])
    model_type = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
    splits = st.slider("Time Series Splits", min_value=3, max_value=10, value=5)
    run_button = st.button("Run Model")

def load_data(gfile, sfile):
    goog = pd.read_csv(gfile, sep=";")[["Date", "Adj.Close"]].rename(columns={"Adj.Close": "goog_price"})
    sp500 = pd.read_csv(sfile, sep=";")[["Date", "Adj.Close"]].rename(columns={"Adj.Close": "sp_price"})
    for df in [goog, sp500]:
        for col in df.columns:
            if col != 'Date':
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    df = pd.merge(goog, sp500, on="Date")
    df["Date"] = pd.to_datetime(df["Date"], format='%d/%m/%Y')
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    df["goog_ret"] = df["goog_price"].pct_change()
    df["sp_ret"] = df["sp_price"].pct_change()
    return df.dropna().reset_index(drop=True)

def prepare_features(df, lags):
    df = df.copy()
    for lag in lags:
        df[f"goog_lag{lag}"] = df["goog_ret"].shift(-lag)
        df[f"sp_lag{lag}"] = df["sp_ret"].shift(-lag)
    df["goog_up"] = (df["goog_ret"] >= 0).astype(int)
    df = df.dropna().reset_index(drop=True)
    features = [f"goog_lag{lag}" for lag in lags] + [f"sp_lag{lag}" for lag in lags]
    return df, features

def select_model(name):
    if name == "Logistic Regression":
        return LogisticRegression()
    elif name == "Decision Tree":
        return DecisionTreeClassifier()
    elif name == "Random Forest":
        return RandomForestClassifier()

if run_button and goog_file and sp500_file:
    df_raw = load_data(goog_file, sp500_file)

    st.header("🔍 Time Series Pattern Detection")

    st.subheader("📉 Mean Reversion (ADF Test)")
    adf_result = adfuller(df_raw["goog_ret"].dropna())
    st.write(f"ADF Statistic: {adf_result[0]:.3f}")
    st.write(f"p-value: {adf_result[1]:.3f}")
    if adf_result[1] < 0.05:
        st.success("Likely Mean Reverting")
    else:
        st.warning("Not Mean Reverting")

    st.subheader("⚡ Momentum (Autocorrelation)")
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(df_raw["goog_ret"].dropna(), lags=20, ax=ax)
    st.pyplot(fig)

    st.subheader("📅 Seasonality Patterns")
    df_raw["Weekday"] = df_raw["Date"].dt.dayofweek
    df_raw["Month"] = df_raw["Date"].dt.month
    col1, col2 = st.columns(2)
    with col1:
        weekday_avg = df_raw.groupby("Weekday")["goog_ret"].mean()
        weekday_avg.plot(kind="bar", title="Avg Return by Weekday")
        st.pyplot(plt.gcf())
    with col2:
        month_avg = df_raw.groupby("Month")["goog_ret"].mean()
        month_avg.plot(kind="bar", title="Avg Return by Month")
        st.pyplot(plt.gcf())

    df_lagged, features = prepare_features(df_raw, lags)
    X = df_lagged[features]
    y = df_lagged["goog_up"]

    st.subheader("🔢 Multicollinearity (VIF)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    st.write(vif_data)

    model = select_model(model_type)
    tscv = TimeSeriesSplit(n_splits=splits)
    metrics = []
    preds_df = pd.DataFrame()
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        preds_df = pd.concat([preds_df, pd.DataFrame({"Actual": y_test, "Predicted": y_pred, "Probability": y_proba})], ignore_index=True)
        metrics.append({
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_proba)
        })
    df_metrics = pd.DataFrame(metrics).mean().round(3)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Metrics", "📉 ROC Curve", "📌 Feature Importance", "🤔 SHAP", "🔍 LIME"])

    with tab1:
        st.subheader("Average Cross-Validation Metrics")
        st.write(df_metrics.T)
        st.dataframe(preds_df.head(10))
        st.download_button("⬇️ Download Predictions", preds_df.to_csv(index=False), "predictions.csv")

    with tab2:
        st.subheader("Mean ROC Curve")
        model.fit(X, y)
        proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
        fpr, tpr, _ = roc_curve(y, proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, proba):.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        st.pyplot(plt.gcf())

    with tab3:
        st.subheader("Feature Importance")
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
            sns.barplot(x="Importance", y="Feature", data=fi)
            st.pyplot(plt.gcf())
        elif hasattr(model, "coef_"):
            coefs = pd.DataFrame({"Feature": features, "Coefficient": model.coef_[0]})
            sns.barplot(x="Coefficient", y="Feature", data=coefs)
            st.pyplot(plt.gcf())
        else:
            st.info("Feature importance not available for this model.")

    with tab4:
        st.subheader("SHAP Explanation")
        explainer = shap.Explainer(model, X)
        shap_values = explainer(pd.DataFrame(X, columns=features))
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(plt.gcf())

    with tab5:
        st.subheader("LIME Explanation (first sample)")
        explainer = LimeTabularExplainer(X.values, feature_names=features, class_names=["Down", "Up"], discretize_continuous=True)
        exp = explainer.explain_instance(X.iloc[0].values, lambda x: model.predict_proba(pd.DataFrame(x, columns=features)), num_features=5)
        st.text(exp.as_list())
else:
    st.info("⬅️ Upload files, choose settings, and run the model.")
