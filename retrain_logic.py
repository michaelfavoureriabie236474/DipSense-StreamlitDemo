import pandas as pd

def determine_status(n_drifted, total_features, current_metrics, baseline_metrics):
    f1_drop = baseline_metrics["f1_score"] - current_metrics["f1_score"]
    auc_drop = baseline_metrics["auc_roc"] - current_metrics["auc_roc"]
    drift_ratio = n_drifted / total_features if total_features else 0

    reasons = []
    
    if n_drifted >= 8:
        reasons.append(f"{n_drifted} features drifted")
    if f1_drop > 0.10:
        reasons.append(f"F1 dropped by {f1_drop:.4f}")
    if auc_drop > 0.05:
        reasons.append(f"AUC-ROC dropped by {auc_drop:.4f}")

    if reasons:
        return {
            "status": "RETRAIN",
            "color": "red",
            "f1_drop": f1_drop,
            "auc_drop": auc_drop,
            "drift_ratio": drift_ratio,
            "reasons": reasons
        }
    warning_reasons = []
    if n_drifted >= 4:
        warning_reasons.append(f"{n_drifted} features drifted")
    if f1_drop > 0.05:
        warning_reasons.append(f"F1 dropped by {f1_drop:.4f}")
    if auc_drop > 0.02:
        warning_reasons.append(f"AUC-ROC dropped by {auc_drop:.4f}")

    if warning_reasons:
        return {
            "status": "WARNING",
            "color": "orange",
            "f1_drop": f1_drop,
            "auc_drop": auc_drop,
            "drift_ratio": drift_ratio,
            "reasons": warning_reasons
        }

    return {
        "status": "OK",
        "color": "green",
        "f1_drop": f1_drop,
        "auc_drop": auc_drop,
        "drift_ratio": drift_ratio,
        "reasons": ["No retraining trigger exceeded"]
    }

def build_retraining_recommendation(selected_scenario, ks_df, current_metrics, baseline_metrics):
    n_drifted = int(ks_df["drifted"].sum())
    total_features = len(ks_df)

    result = determine_status(
        n_drifted=n_drifted,
        total_features=total_features,
        current_metrics=current_metrics,
        baseline_metrics=baseline_metrics
    )
    status = result["status"]

    if status == "RETRAIN":
        recommendation = (
            f"Retraining is recommended for {selected_scenario}. "
            f"The model shows meaningful degradation and/or drift beyond threshold."
        )
    elif status == "WARNING":
        recommendation = (
            f"{selected_scenario} should be monitored closely. "
            f"Retraining is not urgent yet, but warning thresholds were exceeded."
        )
    else:
        recommendation = (
            f"{selected_scenario} is currently stable. "
            f"No immediate retraining is required."
        )
        
    result["recommendation"] = recommendation
    return result