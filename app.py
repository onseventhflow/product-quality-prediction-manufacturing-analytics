import streamlit as st
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


# PAGE CONFIG


st.set_page_config(
    page_title="Product Quality & Manufacturing Analytics",
    page_icon="",
    layout="wide"
)


# LOAD DATA


@st.cache_data
def load_data():
    return pd.read_csv("product_quality_manufacturing_analytics.csv")

df = load_data()


# CACHED MODEL TRAINING FUNCTIONS


@st.cache_resource
def train_prediction_model(_df):
    X = _df.drop("Product_Quality", axis=1)
    y = _df["Product_Quality"]
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X, y)
    return rf


@st.cache_resource
def train_eval_model(_df):
    X = _df.drop("Product_Quality", axis=1)
    y = _df["Product_Quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    return rf, X_test, y_test



# SIDEBAR


page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "EDA",
        "Model Training",
        "Prediction",
        "Dashboard Overview"
    ]
)


if page != "Dashboard Overview":
    st.title("🏭Product Quality & Manufacturing Analytics")


# HOME


if page == "Home":

    st.header("Manufacturing Quality Prediction System")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Target", "Product_Quality")

    st.write(df.head())


# EDA


elif page == "EDA":

    st.header("Exploratory Data Analysis")

    st.subheader("Product Quality Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(
        x="Product_Quality",
        data=df,
        palette="viridis",
        hue="Product_Quality",
        legend=False,
        ax=ax
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Good (0)", "Defective (1)"])
    fig.tight_layout()

    st.pyplot(fig)

    st.info(
        "**Insight:**\n\n"
        "The dataset is highly imbalanced, with approximately 84% defective products and 16% good products. "
        "This indicates that quality issues are common in the manufacturing process and require proactive monitoring.",
        icon="💡"
    )
    plt.close(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )
    fig.tight_layout()

    st.pyplot(fig)

    st.info(
        "**Insight:**\n\n"
        "Equipment Maintenance Count, Defect Rate, and Quality Score show the strongest relationship with product quality, "
        "making them important indicators for quality control.",
        icon="💡"
    )
    plt.close(fig)
    
    st.subheader("Defect Rate Analysis")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.boxplot(
        x="Product_Quality",
        y="Defect_Rate",
        data=df,
        hue="Product_Quality",
        legend=False,
        ax=ax
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Good (0)", "Defective (1)"])
    fig.tight_layout()

    st.pyplot(fig)

    st.info(
        "**Insight:**\n\n"
        "Products with higher defect rates are more likely to be classified as defective, "
        "making defect rate one of the most critical quality indicators.",
        icon="💡"
    )
    plt.close(fig)

    st.subheader("Quality Score Analysis")

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.boxplot(
        x="Product_Quality",
        y="Quality_Score",
        data=df,
        hue="Product_Quality",
        legend=False,
        ax=ax
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Good (0)", "Defective (1)"])
    fig.tight_layout()

    st.pyplot(fig)

    st.info(
        "**Insight:**\n\n"
        "Good products consistently achieve higher quality scores, "
        "indicating that quality score is an effective performance metric.",
        icon="💡"
    )
    plt.close(fig)

# MODEL TRAINING


elif page == "Model Training":

    st.header("Random Forest Model")

    # Use the cached training function to avoid fitting on every run
    rf, X_test, y_test = train_eval_model(df)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.metric("Accuracy", f"{accuracy:.4f}")

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Good (0)", "Defective (1)"],
        yticklabels=["Good (0)", "Defective (1)"]
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    fig.tight_layout()

    st.pyplot(fig)

    st.subheader("ROC Curve")

    y_prob = rf.predict_proba(X_test)[:,1]

    auc_score = roc_auc_score(y_test, y_prob)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_score:.4f}",
        color="darkorange",
        lw=2
    )

    ax.plot([0,1],[0,1], color="navy", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()

    st.pyplot(fig)

    st.success(f"ROC AUC Score = {auc_score:.4f}")


# PREDICTION


elif page == "Prediction":

    st.header("Live Product Quality Prediction")

    X = df.drop("Product_Quality", axis=1)

    # Use cached model to predict instantly
    rf = train_prediction_model(df)

    inputs = {}

    # Arrange inputs in a 3-column grid for clean visuals and alignment
    cols = st.columns(3)
    for idx, col in enumerate(X.columns):
        with cols[idx % 3]:
            inputs[col] = st.number_input(
                col,
                value=float(df[col].mean())
            )

    if st.button("Predict Product Quality"):

        # Ensure correct column alignment
        input_df = pd.DataFrame([inputs])[X.columns]

        prediction = rf.predict(input_df)[0]

        if prediction == 0:
            st.success("✅ Good Product")
        else:
            st.error("❌ Defective Product")


# DASHBOARD OVERVIEW


elif page == "Dashboard Overview":

    st.title("📊 Manufacturing Analytics Dashboard")

    st.markdown("""
This Power BI dashboard provides executive-level visibility into
manufacturing quality performance, defect trends, operational metrics,
and machine learning insights.
""")
    st.image("Manufacturing Dashboard.png")

    st.info("""
Key Findings:
• Equipment Maintenance Count is the strongest quality driver.
• Defect Rate significantly impacts product quality.
• Higher Quality Scores correlate with better products.
• Production Volume influences manufacturing performance.
""")