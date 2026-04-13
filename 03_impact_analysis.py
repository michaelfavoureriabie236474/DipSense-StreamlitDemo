"""
Task 3: Impact Analysis — How Does Drift Affect Model Performance?
===================================================================
Load the trained baseline model and evaluate it on each drift scenario.
Compare metrics against the baseline to quantify performance degradation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import pickle
import json

sns.set_style("whitegrid")

print("TASK 3: IMPACT ANALYSIS — DRIFT EFFECT ON MODEL PERFORMANCE")

# 1. LOAD MODEL, SCALER, AND BASELINE METRICS
print("\n[1] Loading model and baseline metrics...")

with open('Models/baseline_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('Models/baseline_metrics.json', 'r') as f:
    baseline_report = json.load(f)

baseline_metrics = baseline_report['metrics']
print(f"   Model: {baseline_report['baseline_model']}")
print(f"   Baseline F1:        {baseline_metrics['f1_score']:.4f}")
print(f"   Baseline AUC-ROC:   {baseline_metrics['auc_roc']:.4f}")
print(f"   Baseline Precision: {baseline_metrics['precision']:.4f}")
print(f"   Baseline Recall:    {baseline_metrics['recall']:.4f}")

# 2. EVALUATE ON EACH DRIFT SCENARIO
print("\n[2] Evaluating model on drift scenarios...")

drift_files  = [
    'Dataset/drift_1.csv',
    'Dataset/drift_2.csv',
    'Dataset/drift_3.csv',
    'Dataset/drift_4.csv',
    'Dataset/drift_5.csv',
]
drift_labels = ['Drift 1', 'Drift 2', 'Drift 3', 'Drift 4', 'Drift 5']

results = []

train_cols = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
              'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
              'V21','V22','V23','V24','V25','V26','V27','V28','Amount']

for drift_file, drift_label in zip(drift_files, drift_labels):
    df = pd.read_csv(drift_file)
    X  = df.drop('Class', axis=1)
    y  = df['Class']
    X  = X[train_cols]

    X_scaled = scaler.transform(X)
    y_pred   = model.predict(X_scaled)
    y_proba  = model.predict_proba(X_scaled)[:, 1]

    f1        = f1_score(y, y_pred, zero_division=0)
    auc       = roc_auc_score(y, y_proba)
    precision = precision_score(y, y_pred, zero_division=0)
    recall    = recall_score(y, y_pred, zero_division=0)
    cm        = confusion_matrix(y, y_pred)

    results.append({
        'scenario':   drift_label,
        'f1_score':   f1,
        'auc_roc':    auc,
        'precision':  precision,
        'recall':     recall,
        'fraud_rate': float(y.mean()),
        'n_samples':  len(y),
        'cm':         cm,
        'y_true':     y.values,
        'y_proba':    y_proba,
    })

    print(f"\n   {drift_label}:")
    print(f"      F1={f1:.4f}  AUC={auc:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")

# 3. BUILD COMPARISON DATAFRAME
print("\n[3] Building comparison table...")

rows = [{'scenario': 'Baseline',
         'f1_score': baseline_metrics['f1_score'],
         'auc_roc':  baseline_metrics['auc_roc'],
         'precision':baseline_metrics['precision'],
         'recall':   baseline_metrics['recall']}]

for r in results:
    rows.append({k: r[k] for k in ['scenario', 'f1_score', 'auc_roc', 'precision', 'recall']})

metrics_df = pd.DataFrame(rows)

for col in ['f1_score', 'auc_roc', 'precision', 'recall']:
    metrics_df[f'{col}_delta'] = metrics_df[col] - baseline_metrics[col]

print("\n   Performance vs Baseline:")
print(metrics_df[['scenario','f1_score','f1_score_delta',
                   'auc_roc','auc_roc_delta',
                   'precision','precision_delta',
                   'recall','recall_delta']].to_string(index=False, float_format='{:.4f}'.format))

metrics_df.to_csv('Evaluations/impact_analysis_results.csv', index=False)
print("\n   Saved: Evaluations/impact_analysis_results.csv")

# 4. METRICS COMPARISON BAR CHART
print("\n[4] Creating metrics comparison chart...")

metric_cols  = ['f1_score', 'auc_roc', 'precision', 'recall']
metric_names = ['F1 Score', 'AUC-ROC', 'Precision', 'Recall']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (col, name) in enumerate(zip(metric_cols, metric_names)):
    ax     = axes[i]
    values = metrics_df[col].tolist()
    colors = ['#3498db'] + [
        '#e74c3c' if v < baseline_metrics[col] - 0.001 else '#2ecc71'
        for v in values[1:]
    ]
    bars = ax.bar(metrics_df['scenario'], values, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    ax.axhline(y=baseline_metrics[col], color='navy', linestyle='--',
               linewidth=1.5, label=f"Baseline ({baseline_metrics[col]:.3f})")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

plt.suptitle('Model Performance: Baseline vs Drift Scenarios', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Evaluations/impact_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("   Saved: Evaluations/impact_metrics_comparison.png")
plt.close()

# 5. DELTA HEATMAP
print("\n[5] Creating performance delta heatmap...")

delta_df = (metrics_df[metrics_df['scenario'] != 'Baseline']
            [['scenario','f1_score_delta','auc_roc_delta','precision_delta','recall_delta']]
            .set_index('scenario'))
delta_df.columns = ['ΔF1', 'ΔAUC-ROC', 'ΔPrecision', 'ΔRecall']

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(delta_df.astype(float), annot=True, fmt='.4f',
            cmap='RdYlGn', center=0, linewidths=0.5, ax=ax, vmin=-0.3, vmax=0.1)
ax.set_title('Performance Change vs Baseline\n(green = improvement, red = degradation)',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('Evaluations/impact_delta_heatmap.png', dpi=300, bbox_inches='tight')
print("   Saved: Evaluations/impact_delta_heatmap.png")
plt.close()

# 6. CONFUSION MATRICES
print("\n[6] Creating confusion matrices...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, r in enumerate(results):
    disp = ConfusionMatrixDisplay(r['cm'], display_labels=['Legit', 'Fraud'])
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(f"{r['scenario']}  |  F1={r['f1_score']:.4f}",
                      fontsize=11, fontweight='bold')

axes[-1].axis('off')
plt.suptitle('Confusion Matrices — Drift Scenarios', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Evaluations/impact_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("   Saved: Evaluations/impact_confusion_matrices.png")
plt.close()

# 7. F1 & AUC TREND LINE
print("\n[7] Creating trend chart...")

all_scenarios = ['Baseline'] + drift_labels
all_f1  = [baseline_metrics['f1_score']]  + [r['f1_score'] for r in results]
all_auc = [baseline_metrics['auc_roc']]   + [r['auc_roc']  for r in results]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(all_scenarios, all_f1,  marker='o', linewidth=2.5, color='#e74c3c', label='F1 Score',  markersize=9)
ax.plot(all_scenarios, all_auc, marker='s', linewidth=2.5, color='#3498db', label='AUC-ROC',   markersize=9)
ax.axhline(y=baseline_metrics['f1_score'], color='#e74c3c', linestyle='--', alpha=0.35)
ax.axhline(y=baseline_metrics['auc_roc'],  color='#3498db', linestyle='--', alpha=0.35)

for x_pos, (f1, auc) in enumerate(zip(all_f1, all_auc)):
    ax.annotate(f'{f1:.3f}',  (x_pos, f1),  xytext=(0, 10),  textcoords='offset points',
                ha='center', fontsize=9, color='#e74c3c', fontweight='bold')
    ax.annotate(f'{auc:.3f}', (x_pos, auc), xytext=(0, -15), textcoords='offset points',
                ha='center', fontsize=9, color='#3498db', fontweight='bold')

ax.set_title('F1 Score & AUC-ROC Across Drift Scenarios', fontsize=13, fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim(0.5, 1.05)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Evaluations/impact_trend.png', dpi=300, bbox_inches='tight')
print("   Saved: Evaluations/impact_trend.png")
plt.close()

# 8. SAVE IMPACT REPORT JSON
print("\n[8] Saving impact report JSON...")

impact_report = {
    'baseline': {'model': baseline_report['baseline_model'], 'metrics': baseline_metrics},
    'drift_scenarios': [
        {
            'scenario':   r['scenario'],
            'n_samples':  r['n_samples'],
            'fraud_rate': r['fraud_rate'],
            'metrics': {
                'f1_score':  round(r['f1_score'], 6),
                'auc_roc':   round(r['auc_roc'], 6),
                'precision': round(r['precision'], 6),
                'recall':    round(r['recall'], 6),
            },
            'deltas': {
                'f1_score':  round(r['f1_score']  - baseline_metrics['f1_score'], 6),
                'auc_roc':   round(r['auc_roc']   - baseline_metrics['auc_roc'], 6),
                'precision': round(r['precision'] - baseline_metrics['precision'], 6),
                'recall':    round(r['recall']    - baseline_metrics['recall'], 6),
            }
        }
        for r in results
    ]
}

with open('Models/impact_analysis_report.json', 'w') as f:
    json.dump(impact_report, f, indent=2)
print("   Saved: Models/impact_analysis_report.json")

# 9. FINAL SUMMARY
print("\nIMPACT ANALYSIS SUMMARY")
print(f"\nBaseline F1: {baseline_metrics['f1_score']:.4f}")
print(f"\nF1 per scenario:")
for r in results:
    delta     = r['f1_score'] - baseline_metrics['f1_score']
    direction = "down" if delta < 0 else "up"
    print(f"   {r['scenario']}: {r['f1_score']:.4f}  ({direction} {abs(delta):.4f})")

worst = min(results, key=lambda x: x['f1_score'])
print(f"\nWorst scenario: {worst['scenario']} (F1={worst['f1_score']:.4f})")