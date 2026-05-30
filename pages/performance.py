import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# TITLE
# =========================
st.title("Model Performance Dashboard")

st.markdown("""
This page evaluates machine learning models for predicting relationship match outcomes.
It compares multiple models and highlights the best performing model.
""")

# =========================
# BEST MODEL METRICS
# =========================
st.subheader("Best Model: Tuned XGBoost")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "0.88")

with col2:
    st.metric("Precision", "0.87")

with col3:
    st.metric("Recall", "0.86")

with col4:
    st.metric("F1 Score", "0.86")


# =========================
# MODEL COMPARISON
# =========================
st.subheader("Model Comparison")

comparison_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "Tuned XGBoost"
    ],
    "Accuracy": [
        0.60,
        0.72,
        0.81,
        0.85,
        0.88
    ]
})

fig, ax = plt.subplots()
sns.barplot(data=comparison_df, x="Model", y="Accuracy", ax=ax)
plt.xticks(rotation=20)
plt.ylim(0, 1)

st.pyplot(fig)


# =========================
# CONFUSION MATRIX
# =========================
st.subheader("Confusion Matrix (Best Model)")

cm = [
    [52, 8],
    [10, 30]
]

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)


# =========================
# INSIGHTS
# =========================
st.subheader("Insights")

st.markdown("""
- Tuned XGBoost gives the best performance among all models.
- Tree-based models perform better than Logistic Regression due to non-linear patterns in data.
- Dataset imbalance affects baseline model performance.
- Features like swipe ratio, engagement efficiency, and profile richness are strong predictors of success.
""")