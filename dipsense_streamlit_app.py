# dipsense_streamlit_app.py
"""
A lightweight Streamlit dashboard for your DipSense LightGBM model
-----------------------------------------------------------------

* Single‑row prediction from manual inputs ➜ probability + label
* Batch scoring from a CSV upload ➜ download + preview
* Threshold slider to experiment with precision/recall trade‑off
* Feature importance chart (built‑in LightGBM gain)
* Basic model / environment metadata panel

Run locally:
    pip install streamlit lightgbm pandas scikit-learn numpy matplotlib seaborn
    streamlit run dipsense_streamlit_app.py

"""

from __future__ import annotations
from pathlib import Path
import joblib
import lightgbm as lgb
import streamlit as st
# (other imports…)


import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (classification_report, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

MODEL_PATH = Path("lightgbm_best_model.pkl")  # adjust for your deployment env

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_model(path: Path) -> lgb.LGBMClassifier:
    """Load the Joblib-dumped LightGBM model from disk (cached)."""
    return joblib.load(path)



def ensure_column_order(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Reindex df to match training column order (raises if columns missing)."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[columns]


# -----------------------------------------------------------------------------
# UI – sidebar settings
# -----------------------------------------------------------------------------

st.set_page_config(page_title="DipSense Demo", page_icon="📉", layout="wide")
st.title("📈 DipSense ‑ Bounce‑Back Dip Classifier")

model = load_model(MODEL_PATH)
feature_names = list(model.feature_name_)  # preserves training order

# Threshold slider (shared for single + batch)
st.sidebar.header("🔧 Inference Settings")
threshold = st.sidebar.slider(
    "Probability threshold for positive class (dip recovery)", 0.05, 0.95, 0.50, 0.01
)

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Model Details")
st.sidebar.text(f"Type : {type(model).__name__}")
st.sidebar.text(f"n_estimators : {model.n_estimators}")
st.sidebar.text(f"num_leaves : {model.num_leaves}")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab_single, tab_batch, tab_insights = st.tabs(["🔍 Single Prediction", "📑 Batch Scoring", "📊 Model Insights"])

# ————————————————————————————————————————————————————————————————
# Single‑row prediction tab
# ————————————————————————————————————————————————————————————————
with tab_single:
    st.subheader("Enter feature values")

    # Build inputs dynamically based on feature list (numeric only)
    col1, col2 = st.columns(2)
    input_dict = {}
    for idx, feat in enumerate(feature_names):
        default_val = 0.0
        col = col1 if idx % 2 == 0 else col2
        input_dict[feat] = col.number_input(feat, value=default_val, format="%f")

    if st.button("Predict", type="primary"):
        sample_df = pd.DataFrame([input_dict])
        sample_ordered = ensure_column_order(sample_df, feature_names)
        prob = float(model.predict_proba(sample_ordered)[:, 1])
        label = int(prob >= threshold)

        st.metric("Probability of recovery", f"{prob:.3f}")
        st.metric("Prediction", "✅ Recovery" if label else "🚫 No‑recovery", delta=None)

# ————————————————————————————————————————————————————————————————
# Batch scoring tab
# ————————————————————————————————————————————————————————————————
with tab_batch:
    st.subheader("Upload a CSV for batch predictions")
    uploaded = st.file_uploader("CSV file with the same feature columns (any order)", type="csv")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        try:
            data_prepped = ensure_column_order(data, feature_names)
        except ValueError as e:
            st.error(str(e))
        else:
            proba = model.predict_proba(data_prepped)[:, 1]
            pred = (proba >= threshold).astype(int)
            output = data.copy()
            output["probability"] = proba
            output["prediction"] = pred

            st.success(f"Scored {len(output)} rows 🏁")
            st.dataframe(output.head(100))

            buf = io.StringIO()
            output.to_csv(buf, index=False)
            st.download_button("📥 Download predictions", buf.getvalue(), "dipsense_predictions.csv", "text/csv")

# ————————————————————————————————————————————————————————————————
# Insights tab – feature importance + threshold exploration
# ————————————————————————————————————————————————————————————————
with tab_insights:
    st.subheader("Feature Importance (gain)")
    fi = pd.DataFrame({"feature": feature_names, "importance": model.booster_.feature_importance(importance_type="gain")})
    fi_sorted = fi.sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 0.35 * len(fi_sorted)))
    sns.barplot(data=fi_sorted, y="feature", x="importance", palette="viridis", ax=ax)
    ax.set_xlabel("Total gain")
    ax.set_ylabel("")
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Threshold explorer (validation data)")
    st.markdown("Upload a labelled validation dataset to inspect precision/recall trade‑offs if desired.")
    val_file = st.file_uploader("Validation CSV (with 'target' col)", type="csv", key="val")

    if val_file:
        val_df = pd.read_csv(val_file)
        if "target" not in val_df.columns:
            st.error("CSV must contain a 'target' column for ground‑truth labels.")
        else:
            X_val = ensure_column_order(val_df.drop(columns=["target"]), feature_names)
            y_val = val_df["target"].values
            proba_val = model.predict_proba(X_val)[:, 1]
            fpr, tpr, thr = roc_curve(y_val, proba_val)
            prec, rec, thr_pr = precision_recall_curve(y_val, proba_val)
            auc = roc_auc_score(y_val, proba_val)

            st.caption(f"AUC‑ROC = {auc:.3f}")
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label="ROC curve")
            ax2.plot([0, 1], [0, 1], "--", color="grey")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            st.pyplot(fig2)

            fig3, ax3 = plt.subplots()
            ax3.plot(rec, prec)
            ax3.set_xlabel("Recall")
            ax3.set_ylabel("Precision")
            st.pyplot(fig3)
