"""
Task 2: Compare Training vs Production Data
==============================================
Analyse and visualise distributional differences between baseline training data
and production drift scenarios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("=" * 80)
print("TASK 2: COMPARE TRAINING VS PRODUCTION DATA")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading data...")

# Load training data (original creditcard.csv)
df_train = pd.read_csv('creditcard.csv')
X_train = df_train.drop('Class', axis=1)
y_train = df_train['Class']

print(f"   Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"   Fraud rate (train): {100*y_train.mean():.2f}%")

# Load drift scenarios
drift_files = ['drift_1.csv', 'drift_2.csv', 'drift_3.csv', 'drift_4.csv', 'drift_5.csv']
drift_data = {}

for drift_file in drift_files:
    df_drift = pd.read_csv(drift_file)
    X_drift = df_drift.drop('Class', axis=1)
    y_drift = df_drift['Class']
    
    drift_data[drift_file] = {
        'X': X_drift,
        'y': y_drift,
        'n_samples': X_drift.shape[0],
        'fraud_rate': 100 * y_drift.mean()
    }
    
    print(f"   {drift_file}: {X_drift.shape[0]} samples, fraud rate: {100*y_drift.mean():.2f}%")

# ============================================================================
# 2. STATISTICAL COMPARISON
# ============================================================================
print("\n[2] Computing statistical differences...")

# Calculate summary statistics for train vs drift
def compute_stats(X, name="Dataset"):
    """Compute mean, std, min, max for all features"""
    stats_dict = {
        'name': name,
        'mean': X.mean(),
        'std': X.std(),
        'min': X.min(),
        'max': X.max(),
        'median': X.median(),
    }
    return stats_dict

train_stats = compute_stats(X_train, "Train")

comparison_results = {}

for drift_name, drift_dict in drift_data.items():
    X_drift = drift_dict['X']
    drift_stats = compute_stats(X_drift, drift_name)
    
    # Calculate KS test and effect size for each feature
    ks_tests = {}
    for col in X_train.columns:
        ks_stat, ks_pval = stats.ks_2samp(X_train[col], X_drift[col])
        
        # Cohen's d effect size
        pooled_std = np.sqrt((X_train[col].std()**2 + X_drift[col].std()**2) / 2)
        cohens_d = (X_train[col].mean() - X_drift[col].mean()) / pooled_std if pooled_std > 0 else 0
        
        ks_tests[col] = {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'cohens_d': cohens_d,
            'mean_diff': X_train[col].mean() - X_drift[col].mean(),
            'std_diff': X_train[col].std() - X_drift[col].std(),
        }
    
    comparison_results[drift_name] = {
        'stats': drift_stats,
        'ks_tests': ks_tests,
    }

print(f"   ✓ Computed KS tests and effect sizes for all features across all drift scenarios")

# ============================================================================
# 3. SUMMARIZE KEY FINDINGS
# ============================================================================
print("\n[3] Key findings summary:")

for drift_name, results in comparison_results.items():
    print(f"\n   {drift_name}:")
    
    ks_tests = results['ks_tests']
    
    # Count significant differences (p < 0.05)
    significant_features = sum(1 for col, test in ks_tests.items() if test['ks_pvalue'] < 0.05)
    
    # Find top drifted features (by Cohen's d)
    top_features = sorted(ks_tests.items(), key=lambda x: abs(x[1]['cohens_d']), reverse=True)[:3]
    
    print(f"      • Significant features (p < 0.05): {significant_features}/{len(ks_tests)}")
    print(f"      • Top drifted features:")
    for feat, test in top_features:
        print(f"         - {feat}: Cohen's d = {test['cohens_d']:.3f}, KS p-value = {test['ks_pvalue']:.4f}")

# ============================================================================
# 4. SAVE COMPARISON REPORT
# ============================================================================
print("\n[4] Saving comparison report...")

# Convert to JSON-serializable format
report_data = {}
for drift_name, results in comparison_results.items():
    report_data[drift_name] = {
        'ks_tests': {
            col: {
                'ks_statistic': float(test['ks_statistic']),
                'ks_pvalue': float(test['ks_pvalue']),
                'cohens_d': float(test['cohens_d']),
                'mean_diff': float(test['mean_diff']),
                'std_diff': float(test['std_diff']),
            }
            for col, test in results['ks_tests'].items()
        }
    }

with open('distribution_comparison_report.json', 'w') as f:
    json.dump(report_data, f, indent=2)
print("   ✓ Saved: distribution_comparison_report.json")

# ============================================================================
# 5. VISUALIZE DISTRIBUTIONS - PCA FEATURES
# ============================================================================
print("\n[5] Creating distribution visualizations...")

# Select key features to visualize (V1-V10 and Amount)
key_features = [f'V{i}' for i in range(1, 11)] + ['Amount']

# Create subplots for each drift scenario
for drift_name, drift_dict in drift_data.items():
    X_drift = drift_dict['X']
    
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    fig.suptitle(f'Distribution Comparison: Train vs {drift_name}', fontsize=16, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    for idx, feature in enumerate(key_features):
        ax = axes[idx]
        
        # KDE plots
        X_train[feature].plot(kind='density', ax=ax, label='Train', linewidth=2.5, color='#2ecc71')
        X_drift[feature].plot(kind='density', ax=ax, label=f'{drift_name}', linewidth=2.5, color='#e74c3c')
        
        ax.set_title(f'{feature}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
    
    # Hide the last unused subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'dist_comparison_{drift_name.replace(".csv", "")}.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: dist_comparison_{drift_name.replace('.csv', '')}.png")
    plt.close()

# ============================================================================
# 6. VISUALIZE KS STATISTICS - ALL FEATURES
# ============================================================================
print("\n[6] Creating KS statistic heatmap...")

# Create a matrix of KS p-values
features_list = list(X_train.columns)
ks_pvalue_matrix = np.zeros((len(drift_files), len(features_list)))

for i, drift_name in enumerate(drift_files):
    for j, feature in enumerate(features_list):
        ks_pvalue_matrix[i, j] = comparison_results[drift_name]['ks_tests'][feature]['ks_pvalue']

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    ks_pvalue_matrix,
    annot=False,
    cmap='RdYlGn_r',
    center=0.05,
    cbar_kws={'label': 'KS p-value (lower = more drift)'},
    xticklabels=features_list,
    yticklabels=[f.replace('.csv', '') for f in drift_files],
    ax=ax,
    vmin=0,
    vmax=0.1
)
ax.set_title('KS Test p-values: Train vs Production Drift\n(Red = significant drift, p < 0.05)', 
             fontweight='bold', fontsize=13)
ax.set_xlabel('Features', fontweight='bold')
ax.set_ylabel('Drift Scenario', fontweight='bold')
plt.tight_layout()
plt.savefig('ks_pvalue_heatmap.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: ks_pvalue_heatmap.png")
plt.close()

# ============================================================================
# 7. VISUALIZE EFFECT SIZES (COHEN'S D)
# ============================================================================
print("\n[7] Creating effect size heatmap...")

# Create a matrix of Cohen's d values
cohens_d_matrix = np.zeros((len(drift_files), len(features_list)))

for i, drift_name in enumerate(drift_files):
    for j, feature in enumerate(features_list):
        cohens_d_matrix[i, j] = comparison_results[drift_name]['ks_tests'][feature]['cohens_d']

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    cohens_d_matrix,
    annot=False,
    cmap='coolwarm',
    center=0,
    cbar_kws={'label': "Cohen's d (effect size)"},
    xticklabels=features_list,
    yticklabels=[f.replace('.csv', '') for f in drift_files],
    ax=ax,
    vmin=-1,
    vmax=1
)
ax.set_title("Cohen's d Effect Sizes: Train vs Production Drift\n(Red = large positive shift, Blue = large negative shift)", 
             fontweight='bold', fontsize=13)
ax.set_xlabel('Features', fontweight='bold')
ax.set_ylabel('Drift Scenario', fontweight='bold')
plt.tight_layout()
plt.savefig('cohens_d_heatmap.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: cohens_d_heatmap.png")
plt.close()

# ============================================================================
# 8. SUMMARY STATISTICS TABLE
# ============================================================================
print("\n[8] Creating summary table...")

# Create summary dataframe
summary_list = []
for drift_name in drift_files:
    results = comparison_results[drift_name]
    ks_tests = results['ks_tests']
    
    significant_count = sum(1 for test in ks_tests.values() if test['ks_pvalue'] < 0.05)
    max_effect = max(abs(test['cohens_d']) for test in ks_tests.values())
    
    summary_list.append({
        'Drift Scenario': drift_name.replace('.csv', ''),
        'N Samples': drift_data[drift_name]['n_samples'],
        'Fraud Rate (%)': f"{drift_data[drift_name]['fraud_rate']:.2f}",
        'Significant Features': f"{significant_count}/{len(ks_tests)}",
        'Max Effect Size': f"{max_effect:.3f}",
    })

summary_df = pd.DataFrame(summary_list)
print("\n   Drift Scenario Summary:")
print(summary_df.to_string(index=False))

# Save to CSV
summary_df.to_csv('distribution_summary.csv', index=False)
print("\n   ✓ Saved: distribution_summary.csv")

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DISTRIBUTION COMPARISON SUMMARY")
print("=" * 80)
print(f"\nTraining Data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"\nDrift Scenarios Analyzed: {len(drift_files)}")

print(f"\nKey Outputs:")
print(f"  ✓ distribution_comparison_report.json — detailed statistical tests")
print(f"  ✓ distribution_summary.csv — quick reference table")
print(f"  ✓ dist_comparison_drift_*.png — KDE plots for each scenario")
print(f"  ✓ ks_pvalue_heatmap.png — KS test significance across all features")
print(f"  ✓ cohens_d_heatmap.png — effect sizes across all features")

print(f"\nNext Step: Task 3 - Drift Detection with PSI and other methods")
print("=" * 80)
