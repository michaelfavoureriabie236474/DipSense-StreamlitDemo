import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import pickle
from scipy import stats
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ML Drift Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
CSS = """
<style>
[data-testid='stSidebar'] {
    background-color: #1a1a2e;
}
[data-testid='stSidebar'] * {
    color: #e0e0e0 !important;
}
[data-testid='metric-container'] {
    background: #1e1e2e;
    border: 1px solid #333355;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid='stMetricValue'] {
    font-size: 1.8rem !important;
    font-weight: 700;
}
h2 {
    border-bottom: 2px solid #e74c3c;
    padding-bottom: 6px;
    margin-top: 1.2rem;
}
h3 {
    color: #a0c4ff;
}
.health-green {
    background: #0d2b1a;
    border-left: 5px solid #2ecc71;
    padding: 14px 18px;
    border-radius: 8px;
    margin: 8px 0;
}
.health-yellow {
    background: #2b250d;
    border-left: 5px solid #f39c12;
    padding: 14px 18px;
    border-radius: 8px;
    margin: 8px 0;
}
.health-red {
    background: #2b0d0d;
    border-left: 5px solid #e74c3c;
    padding: 14px 18px;
    border-radius: 8px;
    margin: 8px 0;
}
.retrain-box {
    background: #2b1a0d;
    border: 2px solid #e67e22;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.stable-box {
    background: #0d2b1a;
    border: 2px solid #2ecc71;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
TRAIN_COLS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

# --------------------------------------------------
# DATA LOADERS
# --------------------------------------------------
@st.cache_data
def load_data():
    # Use test_set_baseline.csv as the baseline reference for drift comparison
    baseline_df = pd.read_csv("Dataset/test_set_baseline.csv")
    drift_dfs = {
        "Drift 1": pd.read_csv("Dataset/drift_1.csv"),
        "Drift 2": pd.read_csv("Dataset/drift_2.csv"),
        "Drift 3": pd.read_csv("Dataset/drift_3.csv"),
        "Drift 4": pd.read_csv("Dataset/drift_4.csv"),
        "Drift 5": pd.read_csv("Dataset/drift_5.csv"),
    }
    return baseline_df, drift_dfs


@st.cache_resource
def load_model():
    with open("Models/baseline_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("Models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


@st.cache_data
def load_baseline_metrics():
    with open("Models/baseline_metrics.json", "r") as f:
        return json.load(f)


def avail_cols(df):
    return [c for c in TRAIN_COLS if c in df.columns]


def run_ks_tests(baseline_df, drift_df, alpha_val=0.05):
    results = []

    for feat in TRAIN_COLS:
        if feat not in baseline_df.columns or feat not in drift_df.columns:
            continue

        base_s = baseline_df[feat].dropna()
        prod_s = drift_df[feat].dropna()

        if base_s.empty or prod_s.empty:
            continue

        stat, pval = stats.ks_2samp(base_s, prod_s)

        results.append(
            {
                "feature": feat,
                "ks_statistic": stat,
                "p_value": pval,
                "drifted": pval < alpha_val,
                "mean_baseline": float(base_s.mean()),
                "mean_prod": float(prod_s.mean()),
                "mean_diff": float(prod_s.mean() - base_s.mean()),
            }
        )

    if not results:
        return pd.DataFrame(
            columns=[
                "feature",
                "ks_statistic",
                "p_value",
                "drifted",
                "mean_baseline",
                "mean_prod",
                "mean_diff",
            ]
        )

    return (
        pd.DataFrame(results)
        .sort_values("ks_statistic", ascending=False)
        .reset_index(drop=True)
    )


def compute_psi(ref, prod, buckets=10):
    ref = np.asarray(ref)
    prod = np.asarray(prod)

    if len(ref) == 0 or len(prod) == 0:
        return np.nan

    lo, hi = np.min(ref), np.max(ref)
    if lo == hi:
        return 0.0

    bins = np.linspace(lo, hi, buckets + 1)
    bins[0] -= 1e-6
    bins[-1] += 1e-6

    ref_pct = np.histogram(ref, bins=bins)[0] / max(len(ref), 1)
    prod_pct = np.histogram(prod, bins=bins)[0] / max(len(prod), 1)

    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    prod_pct = np.where(prod_pct == 0, 1e-6, prod_pct)

    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))


@st.cache_data
def build_psi_matrix(baseline_df, drift_dfs):
    rows = {}
    for name, df in drift_dfs.items():
        row = {}
        for feat in TRAIN_COLS:
            if feat in baseline_df.columns and feat in df.columns:
                row[feat] = compute_psi(
                    baseline_df[feat].dropna().values,
                    df[feat].dropna().values,
                )
        rows[name] = row

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).T


def evaluate_scenario(drift_df, model, scaler):
    if "Class" not in drift_df.columns:
        return None

    cols = avail_cols(drift_df)
    if not cols:
        return None

    try:
        X = drift_df[cols]
        y = drift_df["Class"]

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]

        try:
            auc = roc_auc_score(y, y_proba)
        except ValueError:
            auc = float("nan")

        return {
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "auc_roc": auc,
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "cm": confusion_matrix(y, y_pred, labels=[0, 1]),
            "fraud_rate": float(y.mean()),
        }
    except Exception:
        return None


def retraining_decision(
    ks_df,
    perf,
    baseline_metrics,
    drift_pct_thresh,
    f1_drop_thresh,
    roc_drop_thresh,
):
    reasons = []
    alerts = []

    n_drift = int(ks_df["drifted"].sum()) if not ks_df.empty else 0
    pct = n_drift / max(len(TRAIN_COLS), 1)

    if pct > drift_pct_thresh:
        reasons.append(
            f"{n_drift}/{len(TRAIN_COLS)} features drifted ({pct*100:.1f}% > {drift_pct_thresh*100:.0f}%)"
        )
        alerts.append(f"Warning: {pct*100:.1f}% of features show significant drift")

    if perf and not np.isnan(perf.get("f1_score", float("nan"))):
        f1_drop = baseline_metrics["f1_score"] - perf["f1_score"]
        roc_drop = baseline_metrics["auc_roc"] - perf.get("auc_roc", baseline_metrics["auc_roc"])

        if f1_drop > f1_drop_thresh:
            reasons.append(f"F1 dropped {f1_drop:.4f} (threshold {f1_drop_thresh})")
            alerts.append(f"Warning: F1 degraded by {f1_drop:.4f}")

        if not np.isnan(roc_drop) and roc_drop > roc_drop_thresh:
            reasons.append(f"AUC-ROC dropped {roc_drop:.4f} (threshold {roc_drop_thresh})")
            alerts.append(f"Warning: AUC-ROC degraded by {roc_drop:.4f}")

    retrain = bool(reasons)

    if not retrain:
        alerts.append("All metrics within acceptable range - no retraining needed.")

    return retrain, reasons, alerts


# --------------------------------------------------
# LOAD EVERYTHING
# --------------------------------------------------
try:
    baseline_df, drift_dfs = load_data()
    model, scaler = load_model()
    baseline_report = load_baseline_metrics()
    baseline_metrics = baseline_report.get("metrics", baseline_report)
    baseline_model_name = baseline_report.get("baseline_model", "Baseline Model")
    DATA_LOADED = True
except Exception as e:
    DATA_LOADED = False
    load_error = str(e)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## Drift Monitor")
    st.caption("Credit Card Fraud Detection - MLOps Assignment")
    st.divider()

    if DATA_LOADED:
        st.success("Data loaded")
        st.info(f"Baseline: {baseline_model_name}")
    else:
        st.error("Data not loaded")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Drift Analysis",
            "Impact Analysis",
            "Feature Deep Dive",
            "Monitoring & Retraining",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    selected_scenario = st.selectbox(
        "Select Scenario",
        list(drift_dfs.keys()) if DATA_LOADED else [],
    )

    alpha = st.slider("KS significance threshold (alpha)", 0.01, 0.10, 0.05, 0.01)

    if page == "Monitoring & Retraining":
        st.divider()
        st.markdown("**Alert Thresholds**")
        psi_mod_thresh = st.slider("PSI moderate", 0.05, 0.15, 0.10, 0.01)
        psi_high_thresh = st.slider("PSI high (retrain)", 0.10, 0.30, 0.20, 0.01)
        drift_pct_thresh = st.slider("% features retrain", 10, 60, 30, 5) / 100
        f1_drop_thresh = st.slider("F1 drop retrain", 0.01, 0.15, 0.05, 0.01)
        roc_drop_thresh = st.slider("AUC drop retrain", 0.01, 0.10, 0.03, 0.01)
    else:
        psi_mod_thresh = 0.10
        psi_high_thresh = 0.20
        drift_pct_thresh = 0.30
        f1_drop_thresh = 0.05
        roc_drop_thresh = 0.03

    st.divider()
    st.caption("Breda University of Applied Sciences")

if not DATA_LOADED:
    st.error(f"Could not load data: {load_error}")
    st.stop()

# --------------------------------------------------
# PAGE 1 - OVERVIEW
# --------------------------------------------------
if page == "Overview":
    st.title("Model Monitoring Dashboard")
    st.caption("Comparing production drift scenarios against the baseline model")

    st.subheader("Baseline Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1 Score", f"{baseline_metrics['f1_score']:.4f}")
    c2.metric("AUC-ROC", f"{baseline_metrics['auc_roc']:.4f}")
    c3.metric("Precision", f"{baseline_metrics['precision']:.4f}")
    c4.metric("Recall", f"{baseline_metrics['recall']:.4f}")

    st.divider()
    st.subheader("All Scenarios at a Glance")

    summary_rows = []
    for name, df in drift_dfs.items():
        ks_df = run_ks_tests(baseline_df, df, alpha_val=alpha)
        perf = evaluate_scenario(df, model, scaler)
        n_drift = int(ks_df["drifted"].sum()) if not ks_df.empty else 0

        summary_rows.append(
            {
                "Scenario": name,
                "Features Drifted": f"{n_drift}/{len(TRAIN_COLS)}",
                "Drift %": round(n_drift / len(TRAIN_COLS) * 100, 1),
                "F1 Score": round(perf["f1_score"], 4) if perf else np.nan,
                "dF1": round(perf["f1_score"] - baseline_metrics["f1_score"], 4) if perf else np.nan,
                "AUC-ROC": round(perf["auc_roc"], 4) if perf and not np.isnan(perf["auc_roc"]) else np.nan,
                "Fraud Rate %": round(perf["fraud_rate"] * 100, 3) if perf else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    def colour_delta(val):
        if pd.isna(val):
            return ""
        return f"color: {'#2ecc71' if val >= 0 else '#e74c3c'}; font-weight: bold"

    st.dataframe(
        summary_df.style.map(colour_delta, subset=["dF1"]),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("F1 Score Trend Across Scenarios")

    names_plot = ["Baseline"]
    f1_vals_plot = [baseline_metrics["f1_score"]]

    for r in summary_rows:
        names_plot.append(r["Scenario"])
        f1_vals_plot.append(0 if pd.isna(r["F1 Score"]) else float(r["F1 Score"]))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor("#0e0e1a")
    fig.patch.set_facecolor("#0e0e1a")

    ax.plot(
        names_plot,
        f1_vals_plot,
        marker="o",
        linewidth=2.5,
        color="#3498db",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="#3498db",
        markeredgewidth=2,
    )
    ax.axhline(
        y=baseline_metrics["f1_score"],
        color="#888",
        linestyle="--",
        alpha=0.6,
        label="Baseline",
    )

    for x, y_val in enumerate(f1_vals_plot):
        ax.annotate(
            f"{y_val:.3f}",
            (x, y_val),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    ax.set_ylabel("F1 Score", color="#ccc")
    ax.set_ylim(max(0, min(f1_vals_plot) - 0.1), 1.08)
    ax.tick_params(colors="#ccc")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(alpha=0.2, color="#444")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# PAGE 2 - DRIFT ANALYSIS
# --------------------------------------------------
elif page == "Drift Analysis":
    st.title(f"Drift Analysis - {selected_scenario}")

    ks_df = run_ks_tests(baseline_df, drift_dfs[selected_scenario], alpha_val=alpha)
    n_drift = int(ks_df["drifted"].sum()) if not ks_df.empty else 0
    max_ks = ks_df["ks_statistic"].max() if not ks_df.empty else 0.0
    top_feat = ks_df.iloc[0]["feature"] if not ks_df.empty else "N/A"

    c1, c2, c3 = st.columns(3)
    c1.metric("Features Drifted", f"{n_drift} / {len(TRAIN_COLS)}")
    c2.metric("Max KS Statistic", f"{max_ks:.4f}")
    c3.metric("Top Drifted Feature", top_feat)

    st.divider()
    st.subheader("KS Statistic Heatmap - All Scenarios")

    heatmap_frames = []
    for name, df in drift_dfs.items():
        temp = run_ks_tests(baseline_df, df, alpha_val=alpha)
        if not temp.empty:
            heatmap_frames.append(temp.set_index("feature")["ks_statistic"].rename(name))

    if heatmap_frames:
        hmap = pd.concat(heatmap_frames, axis=1).fillna(0)
        hmap = hmap.loc[hmap.max(axis=1).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(10, 12))
        fig.patch.set_facecolor("#0e0e1a")
        ax.set_facecolor("#0e0e1a")

        sns.heatmap(
            hmap,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            linewidths=0.4,
            vmin=0,
            vmax=1,
            ax=ax,
            annot_kws={"size": 8, "color": "black"},
        )

        ax.set_title(
            "KS Statistic per Feature per Scenario  (higher = more drift)",
            pad=12,
            color="white",
        )
        ax.tick_params(colors="#ccc")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("No KS heatmap data available.")

    st.divider()

    if ks_df.empty:
        st.warning("No KS drift results for this scenario.")
    else:
        st.subheader(f"Top Drifted Features - {selected_scenario}")
        top15 = ks_df.head(15).copy()

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#0e0e1a")
        ax.set_facecolor("#151525")

        bar_colors = ["#e74c3c" if d else "#2ecc71" for d in top15["drifted"]]
        ax.barh(
            top15["feature"][::-1],
            top15["ks_statistic"][::-1],
            color=bar_colors[::-1],
            edgecolor="#333",
            linewidth=0.5,
        )
        ax.axvline(0.1, color="#888", linestyle="--", alpha=0.6, label="KS = 0.10 guide")
        ax.set_xlabel("KS Statistic", color="#ccc")
        ax.tick_params(colors="#ccc")
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.grid(axis="x", alpha=0.2, color="#444")

        red_p = mpatches.Patch(color="#e74c3c", label=f"Drifted (p < {alpha})")
        green_p = mpatches.Patch(color="#2ecc71", label="Stable")
        ax.legend(handles=[red_p, green_p], facecolor="#1a1a2e", labelcolor="white")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# --------------------------------------------------
# PAGE 3 - IMPACT ANALYSIS
# --------------------------------------------------
elif page == "Impact Analysis":
    st.title("Impact Analysis - Drift Effect on Model Performance")

    perf_rows = []
    for name, df in drift_dfs.items():
        p = evaluate_scenario(df, model, scaler)
        perf_rows.append(
            {
                "scenario": name,
                "f1_score": p["f1_score"] if p else 0.0,
                "auc_roc": p["auc_roc"] if p and not np.isnan(p["auc_roc"]) else 0.0,
                "precision": p["precision"] if p else 0.0,
                "recall": p["recall"] if p else 0.0,
                "cm": p["cm"] if p else None,
            }
        )

    perf_df = pd.DataFrame(perf_rows)

    st.subheader("Performance Change vs Baseline")
    delta_df = perf_df[["scenario", "f1_score", "auc_roc", "precision", "recall"]].set_index("scenario").copy()

    for col in delta_df.columns:
        delta_df[col] = delta_df[col] - baseline_metrics[col]

    delta_df.columns = ["dF1", "dAUC-ROC", "dPrecision", "dRecall"]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0e0e1a")
    ax.set_facecolor("#0e0e1a")

    sns.heatmap(
        delta_df,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
        vmin=-0.3,
        vmax=0.1,
        annot_kws={"size": 10},
    )

    ax.set_title("Green = improvement  /  Red = degradation vs baseline", color="white", pad=10)
    ax.tick_params(colors="#ccc")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Metric Scores per Scenario")

    metric_cols = ["f1_score", "auc_roc", "precision", "recall"]
    metric_names = ["F1 Score", "AUC-ROC", "Precision", "Recall"]
    all_scen = ["Baseline"] + perf_df["scenario"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("#0e0e1a")
    axes = axes.flatten()

    for i, (col, name) in enumerate(zip(metric_cols, metric_names)):
        ax = axes[i]
        ax.set_facecolor("#151525")
        vals = [baseline_metrics[col]] + perf_df[col].tolist()
        clrs = ["#3498db"] + [
            "#e74c3c" if v < baseline_metrics[col] - 0.001 else "#2ecc71"
            for v in perf_df[col].tolist()
        ]

        bars = ax.bar(all_scen, vals, color=clrs, alpha=0.88, edgecolor="#555", linewidth=0.6)
        ax.axhline(
            baseline_metrics[col],
            color="#3498db",
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline ({baseline_metrics[col]:.3f})",
        )

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="white",
            )

        ax.set_title(name, fontsize=11, fontweight="bold", color="white")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
        ax.tick_params(colors="#ccc", axis="both")
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.grid(axis="y", alpha=0.2, color="#444")

    plt.suptitle("Model Performance: Baseline vs Drift Scenarios", fontsize=13, fontweight="bold", color="white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader(f"Confusion Matrix - {selected_scenario}")

    perf = evaluate_scenario(drift_dfs[selected_scenario], model, scaler)

    if perf is None:
        st.warning("No valid performance data available for this scenario.")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0b1020")
        ax.set_facecolor("#111827")

        disp = ConfusionMatrixDisplay(
            confusion_matrix=perf["cm"],
            display_labels=["Legit", "Fraud"],
        )

        disp.plot(
            ax=ax,
            colorbar=False,
            cmap="Blues",
            values_format="d",
        )

        ax.set_title(
            f"{selected_scenario} | F1 = {perf['f1_score']:.4f}",
            fontsize=20,
            fontweight="bold",
            color="white",
            pad=18,
        )
        ax.set_xlabel("Predicted label", fontsize=14, color="white", labelpad=12)
        ax.set_ylabel("True label", fontsize=14, color="white", labelpad=12)
        ax.tick_params(axis="x", colors="white", labelsize=13)
        ax.tick_params(axis="y", colors="white", labelsize=13)

        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        threshold = perf["cm"].max() / 2 if perf["cm"].size > 0 else 0
        if disp.text_ is not None:
            for i, row in enumerate(disp.text_):
                for j, txt in enumerate(row):
                    if txt is not None:
                        cell_value = perf["cm"][i, j]
                        txt.set_color("white" if cell_value > threshold else "black")
                        txt.set_fontsize(20)
                        txt.set_fontweight("bold")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# --------------------------------------------------
# PAGE 4 - FEATURE DEEP DIVE
# --------------------------------------------------
elif page == "Feature Deep Dive":
    st.title(f"Feature Deep Dive - {selected_scenario}")

    ks_df = run_ks_tests(baseline_df, drift_dfs[selected_scenario], alpha_val=alpha)

    if ks_df.empty:
        st.warning("No common features found between baseline and this drift batch.")
        st.stop()

    selected_feature = st.selectbox(
        "Select Feature",
        ks_df["feature"].tolist(),
        help="Sorted by KS statistic - most drifted first",
    )

    feat_row = ks_df[ks_df["feature"] == selected_feature].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("KS Statistic", f"{feat_row['ks_statistic']:.4f}")
    c2.metric("p-value", f"{feat_row['p_value']:.2e}")
    c3.metric("Drifted?", "Yes" if feat_row["drifted"] else "No")
    c4.metric("Mean Shift", f"{feat_row['mean_diff']:+.4f}")

    st.divider()
    st.subheader(f"Distribution: {selected_feature}")

    q01 = baseline_df[selected_feature].quantile(0.01)
    q99 = baseline_df[selected_feature].quantile(0.99)

    base_clipped = baseline_df[selected_feature].clip(q01, q99)
    prod_clipped = drift_dfs[selected_scenario][selected_feature].clip(q01, q99)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0e0e1a")
    ax.set_facecolor("#151525")

    ax.hist(
        base_clipped,
        bins=60,
        alpha=0.55,
        color="#3498db",
        density=True,
        label="Baseline (training)",
    )
    ax.hist(
        prod_clipped,
        bins=60,
        alpha=0.55,
        color="#e74c3c",
        density=True,
        label=selected_scenario,
    )
    ax.axvline(
        feat_row["mean_baseline"],
        color="#3498db",
        linestyle="--",
        linewidth=1.8,
        label=f"Baseline mean ({feat_row['mean_baseline']:.3f})",
    )
    ax.axvline(
        feat_row["mean_prod"],
        color="#e74c3c",
        linestyle="--",
        linewidth=1.8,
        label=f"Prod mean ({feat_row['mean_prod']:.3f})",
    )

    ax.set_xlabel(selected_feature, color="#ccc")
    ax.set_ylabel("Density", color="#ccc")
    ax.tick_params(colors="#ccc")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(alpha=0.2, color="#444")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")

    drift_label = "DRIFTED" if feat_row["drifted"] else "Stable"
    ax.set_title(
        f"{selected_feature}  |  KS={feat_row['ks_statistic']:.3f}  p={feat_row['p_value']:.2e}  |  {drift_label}",
        color="white",
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Full KS Test Results")

    display_ks = ks_df[["feature", "ks_statistic", "p_value", "drifted", "mean_diff"]].copy()
    display_ks.columns = ["Feature", "KS Statistic", "p-value", "Drifted", "Mean Shift"]

    def highlight_drifted(row):
        if row["Drifted"]:
            return ["background-color: #3d1515; color: #ff8080"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_ks.style.apply(highlight_drifted, axis=1).format(
            {"KS Statistic": "{:.4f}", "p-value": "{:.2e}", "Mean Shift": "{:+.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

# --------------------------------------------------
# PAGE 5 - MONITORING & RETRAINING
# --------------------------------------------------
elif page == "Monitoring & Retraining":
    st.title("Monitoring & Retraining Strategy")
    st.caption("Task 5 - Define thresholds, alert conditions, and automated drift checks")

    ks_df = run_ks_tests(baseline_df, drift_dfs[selected_scenario], alpha_val=alpha)
    perf = evaluate_scenario(drift_dfs[selected_scenario], model, scaler)
    psi_matrix = build_psi_matrix(baseline_df, drift_dfs)

    n_drift = int(ks_df["drifted"].sum()) if not ks_df.empty else 0
    drift_pct = n_drift / len(TRAIN_COLS)

    avg_psi = (
        float(psi_matrix.loc[selected_scenario].mean())
        if not psi_matrix.empty and selected_scenario in psi_matrix.index
        else float("nan")
    )

    retrain, reasons, alerts = retraining_decision(
        ks_df,
        perf,
        baseline_metrics,
        drift_pct_thresh,
        f1_drop_thresh,
        roc_drop_thresh,
    )

    if drift_pct == 0 and not retrain:
        css_cls = "health-green"
        label = "HEALTHY - No significant drift detected"
    elif drift_pct < drift_pct_thresh and not retrain:
        css_cls = "health-yellow"
        label = f"MODERATE DRIFT - {n_drift} feature(s) drifting"
    else:
        css_cls = "health-red"
        label = "HIGH DRIFT / PERFORMANCE DROP - action required"

    st.markdown(
        f'<div class="{css_cls}"><b>Model Health ({selected_scenario}): {label}</b>'
        f"<br>Drifted features: {n_drift}/{len(TRAIN_COLS)} ({drift_pct*100:.1f}%)</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Features Drifted",
        f"{n_drift}/{len(TRAIN_COLS)}",
        f"{drift_pct*100:.1f}%",
        delta_color="inverse",
    )
    k2.metric(
        "Avg PSI",
        f"{avg_psi:.3f}" if not np.isnan(avg_psi) else "N/A",
        f"threshold {psi_high_thresh:.2f}",
        delta_color="off",
    )

    if perf:
        k3.metric(
            "F1 Score",
            f"{perf['f1_score']:.4f}",
            f"{perf['f1_score'] - baseline_metrics['f1_score']:+.4f} vs baseline",
            delta_color="inverse",
        )
        auc_val = perf["auc_roc"] if not np.isnan(perf["auc_roc"]) else 0.0
        k4.metric(
            "AUC-ROC",
            f"{auc_val:.4f}",
            f"{auc_val - baseline_metrics['auc_roc']:+.4f} vs baseline",
            delta_color="inverse",
        )
    else:
        k3.metric("F1 Score", "N/A")
        k4.metric("AUC-ROC", "N/A")

    k5.metric("Retrain?", "YES" if retrain else "NO")

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("PSI per Feature - Alert Thresholds")

        if not psi_matrix.empty and selected_scenario in psi_matrix.index:
            psi_row = psi_matrix.loc[selected_scenario].sort_values(ascending=False)

            bar_clrs = [
                "#e74c3c" if v > psi_high_thresh else "#f39c12" if v > psi_mod_thresh else "#2ecc71"
                for v in psi_row.values
            ]

            fig, ax = plt.subplots(figsize=(8, 7))
            fig.patch.set_facecolor("#0e0e1a")
            ax.set_facecolor("#151525")

            ax.barh(
                psi_row.index[::-1],
                psi_row.values[::-1],
                color=bar_clrs[::-1],
                edgecolor="#333",
                linewidth=0.4,
            )
            ax.axvline(
                psi_mod_thresh,
                color="#f39c12",
                linestyle="--",
                linewidth=1.3,
                label=f"Moderate ({psi_mod_thresh})",
            )
            ax.axvline(
                psi_high_thresh,
                color="#e74c3c",
                linestyle="--",
                linewidth=1.3,
                label=f"High/Retrain ({psi_high_thresh})",
            )

            ax.set_xlabel("PSI", color="#ccc")
            ax.tick_params(colors="#ccc")
            for spine in ax.spines.values():
                spine.set_color("#333")
            ax.grid(axis="x", alpha=0.2, color="#444")
            ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
            ax.set_title(f"Population Stability Index - {selected_scenario}", color="white")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No PSI data available for this scenario.")

    with col_right:
        st.subheader("Retraining Decision")

        if retrain:
            st.error("RETRAINING RECOMMENDED")
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.success("Model stable - no retraining needed")

        st.markdown("")
        st.subheader("Active Alerts")
        for alert in alerts:
            if "Warning" in alert:
                st.warning(alert)
            else:
                st.success(alert)

        st.markdown("")
        st.subheader("Configured Thresholds")
        threshold_df = pd.DataFrame(
            [
                ["KS p-value", f"< {alpha}", "Feature drifted"],
                ["PSI moderate", f"> {psi_mod_thresh}", "Watch closely"],
                ["PSI high", f"> {psi_high_thresh}", "Trigger retrain"],
                ["% features drifted", f"> {drift_pct_thresh*100:.0f}%", "Trigger retrain"],
                ["F1 drop", f"> {f1_drop_thresh}", "Trigger retrain"],
                ["AUC-ROC drop", f"> {roc_drop_thresh}", "Trigger retrain"],
            ],
            columns=["Metric", "Threshold", "Action"],
        )
        st.dataframe(threshold_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("PSI Heatmap - All Scenarios vs All Features")
    if not psi_matrix.empty:
        sorted_feats = psi_matrix.mean(axis=0).sort_values(ascending=False).index

        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor("#0e0e1a")
        ax.set_facecolor("#0e0e1a")

        sns.heatmap(
            psi_matrix[sorted_feats],
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r",
            linewidths=0.3,
            vmin=0,
            vmax=0.30,
            ax=ax,
            annot_kws={"size": 7},
        )
        ax.set_title(
            "PSI per Feature per Scenario  (green = stable | red = high drift)",
            color="white",
            pad=10,
        )
        ax.tick_params(colors="#ccc")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.subheader("Model Performance Trend - All Scenarios")
    trend_rows = []
    for name, df in drift_dfs.items():
        p = evaluate_scenario(df, model, scaler)
        trend_rows.append(
            {
                "Scenario": name,
                "F1": p["f1_score"] if p else 0,
                "AUC": p["auc_roc"] if p and not np.isnan(p["auc_roc"]) else 0,
            }
        )

    trend_df = pd.DataFrame(trend_rows)
    x_labels = ["Baseline"] + trend_df["Scenario"].tolist()
    f1_vals = [baseline_metrics["f1_score"]] + trend_df["F1"].tolist()
    roc_vals = [baseline_metrics["auc_roc"]] + trend_df["AUC"].tolist()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0e0e1a")
    ax.set_facecolor("#151525")

    ax.plot(
        x_labels,
        f1_vals,
        marker="o",
        lw=2.5,
        color="#3498db",
        markersize=9,
        markerfacecolor="white",
        label="F1 Score",
    )
    ax.plot(
        x_labels,
        roc_vals,
        marker="s",
        lw=2.5,
        color="#e67e22",
        markersize=9,
        markerfacecolor="white",
        label="AUC-ROC",
    )
    ax.axhline(
        baseline_metrics["f1_score"] - f1_drop_thresh,
        color="#3498db",
        linestyle=":",
        lw=1.4,
        alpha=0.8,
        label=f"F1 retrain line (-{f1_drop_thresh})",
    )
    ax.axhline(
        baseline_metrics["auc_roc"] - roc_drop_thresh,
        color="#e67e22",
        linestyle=":",
        lw=1.4,
        alpha=0.8,
        label=f"AUC retrain line (-{roc_drop_thresh})",
    )
    ax.fill_between(
        range(len(x_labels)),
        0,
        baseline_metrics["f1_score"] - f1_drop_thresh,
        alpha=0.06,
        color="#e74c3c",
    )

    ax.set_ylabel("Score", color="#ccc")
    ax.set_ylim(max(0, min(f1_vals + roc_vals) - 0.05), 1.08)
    ax.tick_params(colors="#ccc")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(alpha=0.2, color="#444")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.set_title("Performance Over Scenarios - dotted lines = retraining triggers", color="white")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Automated Drift Check - How It Works")
    st.code(
        """# Automated drift check (runs on every new production batch)
from scipy import stats
import numpy as np

KS_ALPHA        = 0.05
PSI_HIGH        = 0.20
DRIFT_PCT_LIMIT = 0.30
F1_DROP_LIMIT   = 0.05
ROC_DROP_LIMIT  = 0.03

def check_and_trigger(reference_df, production_df, model, scaler,
                      baseline_f1, baseline_roc, feature_cols):
    drifted = sum(
        1 for col in feature_cols
        if stats.ks_2samp(reference_df[col], production_df[col])[1] < KS_ALPHA
    )
    drift_pct = drifted / len(feature_cols)

    X = scaler.transform(production_df[feature_cols])
    y = production_df['Class']
    yp = model.predict(X)
    ypr = model.predict_proba(X)[:, 1]

    from sklearn.metrics import f1_score, roc_auc_score
    prod_f1 = f1_score(y, yp, zero_division=0)
    prod_roc = roc_auc_score(y, ypr)

    retrain = (
        drift_pct > DRIFT_PCT_LIMIT or
        baseline_f1 - prod_f1 > F1_DROP_LIMIT or
        baseline_roc - prod_roc > ROC_DROP_LIMIT
    )
    return retrain, drift_pct, prod_f1, prod_roc
""",
        language="python",
    )

    st.subheader("Retraining Strategy")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
**When to retrain**

| Trigger | Condition |
|---------|-----------|
| Feature drift | > 30% of features (KS p < 0.05) |
| Performance drop | F1 drop > 5 pp vs baseline |
| Distribution shift | PSI > 0.20 on features |
| AUC degradation | AUC-ROC drop > 3 pp |

**Monitoring schedule**

| Frequency | Check |
|-----------|-------|
| Per batch | KS test + PSI on incoming data |
| Daily | Full performance evaluation |
| On alert | Immediate retraining pipeline |
"""
        )

    with col_b:
        st.markdown(
            """
**Retraining approach**

1. **Sliding window** - retrain on the most recent N days of production data combined with 30% of original training set to avoid catastrophic forgetting.

2. **Full retrain** - if PSI > 0.40 across most features, retrain from scratch on fresh labelled data.

3. **Champion / Challenger** - keep current model live while the retrained model runs in shadow mode. Promote only if F1 improves by 1 pp without AUC regression.

4. **Evaluation gate** - automated test suite must pass before any model is promoted to production.
"""
        )

    st.divider()

    export_data = {
        "scenario": selected_scenario,
        "drift_pct": round(drift_pct * 100, 1),
        "n_drifted": n_drift,
        "avg_psi": round(float(avg_psi), 4) if not np.isnan(avg_psi) else None,
        "f1": round(perf["f1_score"], 4) if perf else None,
        "auc_roc": round(float(perf["auc_roc"]), 4) if perf and not np.isnan(perf["auc_roc"]) else None,
        "retrain": retrain,
        "reasons": reasons,
        "thresholds": {
            "ks_alpha": alpha,
            "psi_moderate": psi_mod_thresh,
            "psi_high": psi_high_thresh,
            "drift_pct_trigger": drift_pct_thresh,
            "f1_drop_trigger": f1_drop_thresh,
            "roc_drop_trigger": roc_drop_thresh,
        },
    }

    st.download_button(
        "Export Monitoring Report (JSON)",
        data=json.dumps(export_data, indent=2),
        file_name=f"monitoring_{selected_scenario.replace(' ', '_').lower()}.json",
        mime="application/json",
    )