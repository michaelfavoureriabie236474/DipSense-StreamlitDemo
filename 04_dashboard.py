"""
Task 4: Streamlit Monitoring Dashboard
Run with: python -m streamlit run Notebooks/dashboard.py
(from project root: Data-Drift-Challange-main)
"""

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
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(page_title="ML Drift Monitor", layout="wide", initial_sidebar_state="expanded")
# v2

TRAIN_COLS = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
              'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
              'V21','V22','V23','V24','V25','V26','V27','V28','Amount']

@st.cache_data
def load_data():
    baseline_df = pd.read_csv('Dataset/creditcard.csv')
    drift_dfs = {
        'Drift 1': pd.read_csv('Dataset/drift_1.csv'),
        'Drift 2': pd.read_csv('Dataset/drift_2.csv'),
        'Drift 3': pd.read_csv('Dataset/drift_3.csv'),
        'Drift 4': pd.read_csv('Dataset/drift_4.csv'),
        'Drift 5': pd.read_csv('Dataset/drift_5.csv'),
    }
    return baseline_df, drift_dfs

@st.cache_resource
def load_model():
    with open('Models/baseline_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('Models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_baseline_metrics():
    with open('Models/baseline_metrics.json') as f:
        return json.load(f)

def run_ks_tests(baseline_df, drift_df, alpha_val=0.05):
    results = []
    for feat in TRAIN_COLS:
        if feat not in baseline_df.columns or feat not in drift_df.columns:
            continue
        stat, pval = stats.ks_2samp(baseline_df[feat].dropna(), drift_df[feat].dropna())
        results.append({
            'feature':       feat,
            'ks_statistic':  stat,
            'p_value':       pval,
            'drifted':       pval < alpha_val,
            'mean_baseline': baseline_df[feat].mean(),
            'mean_prod':     drift_df[feat].mean(),
            'mean_diff':     drift_df[feat].mean() - baseline_df[feat].mean(),
        })
    if not results:
        return pd.DataFrame(columns=['feature','ks_statistic','p_value','drifted','mean_baseline','mean_prod','mean_diff'])
    return pd.DataFrame(results).sort_values('ks_statistic', ascending=False).reset_index(drop=True)

def evaluate_scenario(drift_df, model, scaler):
    X = drift_df[TRAIN_COLS]
    y = drift_df['Class']
    X_scaled = scaler.transform(X)
    y_pred   = model.predict(X_scaled)
    y_proba  = model.predict_proba(X_scaled)[:, 1]
    return {
        'f1_score':   f1_score(y, y_pred, zero_division=0),
        'auc_roc':    roc_auc_score(y, y_proba),
        'precision':  precision_score(y, y_pred, zero_division=0),
        'recall':     recall_score(y, y_pred, zero_division=0),
        'cm':         confusion_matrix(y, y_pred),
        'fraud_rate': float(y.mean()),
    }

try:
    baseline_df, drift_dfs = load_data()
    model, scaler           = load_model()
    baseline_report         = load_baseline_metrics()
    baseline_metrics        = baseline_report['metrics']
    DATA_LOADED             = True
except Exception as e:
    DATA_LOADED = False
    load_error  = str(e)

with st.sidebar:
    st.title("Drift Monitor")
    st.caption("Credit Card Fraud Detection — MLOps Assignment")
    st.divider()
    if DATA_LOADED:
        st.success("Data loaded")
        st.info(f"Baseline: **{baseline_report['baseline_model']}**")
    else:
        st.error("Data not loaded")

    page = st.radio("Navigate",
        ["Overview", "Drift Analysis", "Impact Analysis", "Feature Deep Dive"],
        label_visibility="collapsed")
    st.divider()
    selected_scenario = st.selectbox("Select Scenario", list(drift_dfs.keys()) if DATA_LOADED else [])
    alpha = st.slider("KS significance threshold (a)", 0.01, 0.10, 0.05, 0.01)

if not DATA_LOADED:
    st.error(f"Could not load data: {load_error}")
    st.stop()

if page == "Overview":
    st.title("Model Monitoring Dashboard")
    st.caption("Comparing production drift scenarios against the baseline model")

    st.subheader("Baseline Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1 Score",  f"{baseline_metrics['f1_score']:.4f}")
    c2.metric("AUC-ROC",   f"{baseline_metrics['auc_roc']:.4f}")
    c3.metric("Precision", f"{baseline_metrics['precision']:.4f}")
    c4.metric("Recall",    f"{baseline_metrics['recall']:.4f}")

    st.divider()
    st.subheader("All Scenarios at a Glance")

    summary_rows = []
    for name, df in drift_dfs.items():
        ks_df   = run_ks_tests(baseline_df, df, alpha_val=alpha)
        perf    = evaluate_scenario(df, model, scaler)
        n_drift = ks_df['drifted'].sum()
        summary_rows.append({
            'Scenario':         name,
            'Features Drifted': f"{n_drift}/{len(TRAIN_COLS)}",
            'Drift %':          round(n_drift / len(TRAIN_COLS) * 100, 1),
            'F1 Score':         round(perf['f1_score'], 4),
            'dF1':              round(perf['f1_score'] - baseline_metrics['f1_score'], 4),
            'AUC-ROC':          round(perf['auc_roc'], 4),
            'Fraud Rate %':     round(perf['fraud_rate'] * 100, 3),
        })

    summary_df = pd.DataFrame(summary_rows)

    def colour_delta(val):
        color = '#2ecc71' if val >= 0 else '#e74c3c'
        return f'color: {color}; font-weight: bold'

    st.dataframe(summary_df.style.map(colour_delta, subset=['dF1']),
                 use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("F1 Score Trend Across Scenarios")

    all_names = ['Baseline'] + list(drift_dfs.keys())
    all_f1    = [baseline_metrics['f1_score']] + [r['F1 Score'] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(all_names, all_f1, marker='o', linewidth=2.5, color='#3498db', markersize=9)
    ax.axhline(y=baseline_metrics['f1_score'], color='gray', linestyle='--', alpha=0.5, label='Baseline')
    for x, y_val in enumerate(all_f1):
        ax.annotate(f'{y_val:.3f}', (x, y_val), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(max(0, min(all_f1) - 0.1), 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

elif page == "Drift Analysis":
    st.title(f"Drift Analysis — {selected_scenario}")

    ks_df   = run_ks_tests(baseline_df, drift_dfs[selected_scenario], alpha_val=alpha)
    n_drift = ks_df['drifted'].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Features Drifted",    f"{n_drift} / {len(TRAIN_COLS)}")
    c2.metric("Max KS Statistic",    f"{ks_df['ks_statistic'].max():.4f}")
    c3.metric("Top Drifted Feature", ks_df.iloc[0]['feature'])

    st.divider()
    st.subheader("KS Statistic Heatmap — All Scenarios")

    heatmap_data = pd.DataFrame({
        name: run_ks_tests(baseline_df, df, alpha_val=alpha).set_index('feature')['ks_statistic']
        for name, df in drift_dfs.items()
    })
    heatmap_data = heatmap_data.loc[heatmap_data.max(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                linewidths=0.4, vmin=0, vmax=1, ax=ax)
    ax.set_title('KS Statistic per Feature per Scenario\n(higher = more drift)', pad=12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader(f"Top 15 Drifted Features — {selected_scenario}")

    top15 = ks_df.head(15).copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    bar_colors = ['#e74c3c' if d else '#2ecc71' for d in top15['drifted']]
    ax.barh(top15['feature'][::-1], top15['ks_statistic'][::-1], color=bar_colors[::-1])
    ax.axvline(0.1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('KS Statistic')
    red_p   = mpatches.Patch(color='#e74c3c', label=f'Drifted (p < {alpha})')
    green_p = mpatches.Patch(color='#2ecc71', label='Stable')
    ax.legend(handles=[red_p, green_p])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

elif page == "Impact Analysis":
    st.title("Impact Analysis — Drift Effect on Model Performance")

    perf_rows = []
    for name, df in drift_dfs.items():
        p = evaluate_scenario(df, model, scaler)
        perf_rows.append({
            'scenario':  name,
            'f1_score':  p['f1_score'],
            'auc_roc':   p['auc_roc'],
            'precision': p['precision'],
            'recall':    p['recall'],
            'cm':        p['cm'],
        })
    perf_df = pd.DataFrame(perf_rows)

    st.subheader("Performance Change vs Baseline")
    delta_df = perf_df[['scenario','f1_score','auc_roc','precision','recall']].set_index('scenario').copy()
    for col in delta_df.columns:
        delta_df[col] = delta_df[col] - baseline_metrics[col]
    delta_df.columns = ['dF1', 'dAUC-ROC', 'dPrecision', 'dRecall']

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(delta_df, annot=True, fmt='.4f', cmap='RdYlGn',
                center=0, linewidths=0.5, ax=ax, vmin=-0.3, vmax=0.1)
    ax.set_title('Green = improvement  /  Red = degradation vs baseline')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Metric Scores per Scenario")

    metric_cols  = ['f1_score', 'auc_roc', 'precision', 'recall']
    metric_names = ['F1 Score', 'AUC-ROC', 'Precision', 'Recall']
    all_scen     = ['Baseline'] + perf_df['scenario'].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for i, (col, name) in enumerate(zip(metric_cols, metric_names)):
        ax   = axes[i]
        vals = [baseline_metrics[col]] + perf_df[col].tolist()
        clrs = ['#3498db'] + [
            '#e74c3c' if v < baseline_metrics[col] - 0.001 else '#2ecc71'
            for v in perf_df[col].tolist()
        ]
        bars = ax.bar(all_scen, vals, color=clrs, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.axhline(baseline_metrics[col], color='navy', linestyle='--', linewidth=1.3,
                   label=f"Baseline ({baseline_metrics[col]:.3f})")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Model Performance: Baseline vs Drift Scenarios', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader(f"Confusion Matrix — {selected_scenario}")
    perf = evaluate_scenario(drift_dfs[selected_scenario], model, scaler)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(perf['cm'], display_labels=['Legit', 'Fraud']).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{selected_scenario}  |  F1={perf['f1_score']:.4f}", fontweight='bold')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

elif page == "Feature Deep Dive":
    st.title(f"Feature Deep Dive — {selected_scenario}")

    ks_df = run_ks_tests(baseline_df, drift_dfs[selected_scenario], alpha_val=alpha)
    selected_feature = st.selectbox("Select Feature", ks_df['feature'].tolist(),
                                    help="Sorted by KS statistic — most drifted first")

    feat_row = ks_df[ks_df['feature'] == selected_feature].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("KS Statistic", f"{feat_row['ks_statistic']:.4f}")
    c2.metric("p-value",      f"{feat_row['p_value']:.2e}")
    c3.metric("Drifted?",     "Yes" if feat_row['drifted'] else "No")
    c4.metric("Mean Shift",   f"{feat_row['mean_diff']:+.4f}")

    st.divider()
    st.subheader(f"Distribution: {selected_feature}")

    q01 = baseline_df[selected_feature].quantile(0.01)
    q99 = baseline_df[selected_feature].quantile(0.99)
    base_clipped = baseline_df[selected_feature].clip(q01, q99)
    prod_clipped = drift_dfs[selected_scenario][selected_feature].clip(q01, q99)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(base_clipped, bins=60, alpha=0.5, color='steelblue', density=True, label='Baseline')
    ax.hist(prod_clipped, bins=60, alpha=0.5, color='tomato',    density=True, label=selected_scenario)
    ax.axvline(feat_row['mean_baseline'], color='steelblue', linestyle='--', linewidth=1.5,
               label=f"Baseline mean ({feat_row['mean_baseline']:.3f})")
    ax.axvline(feat_row['mean_prod'], color='tomato', linestyle='--', linewidth=1.5,
               label=f"Prod mean ({feat_row['mean_prod']:.3f})")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title(f'{selected_feature}  |  KS={feat_row["ks_statistic"]:.3f}  p={feat_row["p_value"]:.2e}')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Full KS Test Results")

    display_ks = ks_df[['feature','ks_statistic','p_value','drifted','mean_diff']].copy()
    display_ks.columns = ['Feature', 'KS Statistic', 'p-value', 'Drifted', 'Mean Shift']

    def highlight_drifted(row):
        if row['Drifted']:
            return ['background-color: #3d1515'] * len(row)
        return [''] * len(row)

    st.dataframe(
        display_ks.style.apply(highlight_drifted, axis=1)
                        .format({'KS Statistic': '{:.4f}', 'p-value': '{:.2e}', 'Mean Shift': '{:+.4f}'}),
        use_container_width=True, hide_index=True
    )