"""
Task 1: Train Baseline Fraud Classifier
========================================
Train a fraud detection model on the Kaggle credit card dataset.
Log baseline metrics: F1, AUC-ROC, precision, recall.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import pickle
import json
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("TASK 1: BASELINE FRAUD CLASSIFIER")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading data...")
df = pd.read_csv('creditcard.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Fraud cases: {df['Class'].sum()} ({100*df['Class'].mean():.2f}%)")
print(f"   Legitimate cases: {len(df) - df['Class'].sum()} ({100*(1-df['Class'].mean()):.2f}%)")

# ============================================================================
# 2. EXPLORE BASIC STATS
# ============================================================================
print("\n[2] Data overview:")
print(f"   Columns: {df.columns.tolist()}")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Data types: {df.dtypes.unique()}")

# ============================================================================
# 3. PREPARE DATA
# ============================================================================
print("\n[3] Preparing training and test sets...")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-validation-test split (60-20-20)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"   Train set: {X_train.shape[0]} samples ({100*y_train.mean():.2f}% fraud)")
print(f"   Validation set: {X_val.shape[0]} samples ({100*y_val.mean():.2f}% fraud)")
print(f"   Test set: {X_test.shape[0]} samples ({100*y_test.mean():.2f}% fraud)")

# Standardize features (fit on train, apply to val/test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("   Features standardized (mean=0, std=1)")

# ============================================================================
# 4. TRAIN MODELS
# ============================================================================
print("\n[4] Training baseline models...")

# Logistic Regression
print("   Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train_scaled, y_train)

# Random Forest
print("   Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# ============================================================================
# 5. EVALUATE ON TEST SET
# ============================================================================
print("\n[5] Evaluating models on test set...")

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics dictionary"""
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    
    return metrics, y_pred, y_pred_proba

# Evaluate Logistic Regression
lr_metrics, lr_pred, lr_proba = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
print(f"\n   Logistic Regression:")
print(f"      F1 Score:    {lr_metrics['f1']:.4f}")
print(f"      AUC-ROC:     {lr_metrics['auc_roc']:.4f}")
print(f"      Precision:   {lr_metrics['precision']:.4f}")
print(f"      Recall:      {lr_metrics['recall']:.4f}")

# Evaluate Random Forest
rf_metrics, rf_pred, rf_proba = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
print(f"\n   Random Forest:")
print(f"      F1 Score:    {rf_metrics['f1']:.4f}")
print(f"      AUC-ROC:     {rf_metrics['auc_roc']:.4f}")
print(f"      Precision:   {rf_metrics['precision']:.4f}")
print(f"      Recall:      {rf_metrics['recall']:.4f}")

# Choose best model
if rf_metrics['f1'] >= lr_metrics['f1']:
    best_model = rf_model
    best_metrics = rf_metrics
    best_pred = rf_pred
    best_proba = rf_proba
    print(f"\n   ✓ Random Forest selected as baseline (F1: {rf_metrics['f1']:.4f})")
else:
    best_model = lr_model
    best_metrics = lr_metrics
    best_pred = lr_pred
    best_proba = best_proba
    print(f"\n   ✓ Logistic Regression selected as baseline (F1: {lr_metrics['f1']:.4f})")

# ============================================================================
# 6. SAVE BASELINE METRICS
# ============================================================================
print("\n[6] Saving baseline metrics...")

baseline_report = {
    'timestamp': datetime.now().isoformat(),
    'dataset': 'creditcard.csv',
    'baseline_model': best_metrics['model_name'],
    'test_set_size': len(y_test),
    'metrics': {
        'f1_score': float(best_metrics['f1']),
        'auc_roc': float(best_metrics['auc_roc']),
        'precision': float(best_metrics['precision']),
        'recall': float(best_metrics['recall']),
    },
    'confusion_matrix': best_metrics['confusion_matrix'],
    'class_distribution': {
        'legitimate': int((y_test == 0).sum()),
        'fraud': int((y_test == 1).sum()),
    }
}

# Save as JSON
with open('baseline_metrics.json', 'w') as f:
    json.dump(baseline_report, f, indent=2)
print("   ✓ Saved: baseline_metrics.json")

# Save model and scaler
with open('baseline_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("   ✓ Saved: baseline_model.pkl")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ Saved: scaler.pkl")

# Save test set for later drift comparison
test_data = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_data['Class'] = y_test.values
test_data['Prediction'] = best_pred
test_data['Probability'] = best_proba
test_data.to_csv('test_set_baseline.csv', index=False)
print("   ✓ Saved: test_set_baseline.csv")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n[7] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion Matrix
cm = best_metrics['confusion_matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
axes[0, 0].set_title(f"Confusion Matrix - {best_metrics['model_name']}", fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')
axes[0, 0].set_xticklabels(['Legitimate', 'Fraud'])
axes[0, 0].set_yticklabels(['Legitimate', 'Fraud'])

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best_proba)
axes[0, 1].plot(fpr, tpr, lw=2, label=f"AUC = {best_metrics['auc_roc']:.3f}")
axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Metrics Bar Chart
metrics_names = ['F1', 'AUC-ROC', 'Precision', 'Recall']
metrics_values = [
    best_metrics['f1'],
    best_metrics['auc_roc'],
    best_metrics['precision'],
    best_metrics['recall']
]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
axes[1, 0].bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Baseline Model Performance Metrics', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(metrics_values):
    axes[1, 0].text(i, v + 0.03, f'{v:.3f}', ha='center', fontweight='bold')

# Prediction Distribution
axes[1, 1].hist(best_proba[y_test == 0], bins=50, alpha=0.6, label='Legitimate', color='blue')
axes[1, 1].hist(best_proba[y_test == 1], bins=50, alpha=0.6, label='Fraud', color='red')
axes[1, 1].set_xlabel('Predicted Fraud Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Prediction Distribution by Class', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_model_evaluation.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: baseline_model_evaluation.png")
plt.show()

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("BASELINE MODEL SUMMARY")
print("=" * 70)
print(f"\nModel: {best_metrics['model_name']}")
print(f"Test Set Size: {len(y_test)}")
print(f"\nKey Metrics:")
print(f"  • F1 Score:    {best_metrics['f1']:.4f}")
print(f"  • AUC-ROC:     {best_metrics['auc_roc']:.4f}")
print(f"  • Precision:   {best_metrics['precision']:.4f}")
print(f"  • Recall:      {best_metrics['recall']:.4f}")
print(f"\nConfusion Matrix:")
print(f"  • True Negatives:  {cm[0][0]}")
print(f"  • False Positives: {cm[0][1]}")
print(f"  • False Negatives: {cm[1][0]}")
print(f"  • True Positives:  {cm[1][1]}")
print(f"\nSaved artifacts:")
print(f"  ✓ baseline_metrics.json")
print(f"  ✓ baseline_model.pkl")
print(f"  ✓ scaler.pkl")
print(f"  ✓ test_set_baseline.csv")
print(f"  ✓ baseline_model_evaluation.png")
print("\n" + "=" * 70)
