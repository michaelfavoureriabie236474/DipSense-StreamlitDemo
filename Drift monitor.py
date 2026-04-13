elif page == "Monitoring & Retraining":
     st.title("Monitoring & Retraining Strategy")
    st.caption("Task 5 - Define thresholds, alert conditions, and automated drift checks")

    ks_df = run_ks_tests(baseline_df, drift_dfs[selected_scenario], alpha_val=alpha)
    perf  = evaluate_scenario(drift_dfs[selected_scenario], model, scaler)

    n_drifted  = int(ks_df['drifted'].sum()) if not ks_df.empty else 0
    drift_pct  = n_drifted / max(len(TRAIN_COLS), 1)

    # Inline retraining logic (no external import needed)
    reasons = []
    if n_drifted >= 8:
        reasons.append(f"{n_drifted} features drifted (>= 8 threshold)")
    if perf and (baseline_metrics['f1_score'] - perf['f1_score']) > 0.10:
        reasons.append(f"F1 dropped by {baseline_metrics['f1_score'] - perf['f1_score']:.4f}")
    if perf and not np.isnan(perf.get('auc_roc', float('nan'))):
        roc_drop = baseline_metrics['auc_roc'] - perf['auc_roc']
        if roc_drop > 0.05:
            reasons.append(f"AUC-ROC dropped by {roc_drop:.4f}")

    warnings = []
    if n_drifted >= 4:
        warnings.append(f"{n_drifted} features drifted (>= 4 warning threshold)")
    if perf and (baseline_metrics['f1_score'] - perf['f1_score']) > 0.05:
        warnings.append(f"F1 dropped by {baseline_metrics['f1_score'] - perf['f1_score']:.4f}")

    if reasons:
        status = "RETRAIN"
        recommendation = "Retraining is recommended based on drift and/or performance degradation."
    elif warnings:
        status = "WARNING"
        recommendation = "Monitor closely. Performance is degrading but has not yet crossed retrain threshold."
    else:
        status = "STABLE"
        recommendation = "Model is stable. No action required."

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenario", selected_scenario)
    c2.metric("Features Drifted", f"{n_drifted}/{len(TRAIN_COLS)}")
    if perf:
        c3.metric("F1 Score",  f"{perf['f1_score']:.4f}",
                  delta=f"{perf['f1_score'] - baseline_metrics['f1_score']:.4f}")
        auc = perf['auc_roc'] if not np.isnan(perf['auc_roc']) else 0.0
        c4.metric("AUC-ROC",  f"{auc:.4f}",
                  delta=f"{auc - baseline_metrics['auc_roc']:.4f}")
    else:
        c3.metric("F1 Score", "N/A")
        c4.metric("AUC-ROC",  "N/A")
        
    st.divider()
    if status == "RETRAIN":
        st.error(f"Status: {status}")
    elif status == "WARNING":
        st.warning(f"Status: {status}")
    else:
        st.success(f"Status: {status}")

    st.subheader("Recommendation")
    st.write(recommendation)

    st.subheader("Trigger Reasons")
    if reasons:
        for r in reasons:
            st.write(f"- {r}")
    elif warnings:
        for w in warnings:
            st.write(f"- {w}")
    else:
        st.write("No triggers activated.")

    st.divider()
    st.subheader("Monitoring Rules Used")
    st.markdown("""
**Warning conditions**
- 4 or more features drifted
- F1 score drops by more than 0.05
- AUC-ROC drops by more than 0.02

**Retraining conditions**
- 8 or more features drifted
- F1 score drops by more than 0.10
- AUC-ROC drops by more than 0.05
    """)

    st.divider()
    st.subheader("Top Drifted Features for This Scenario")
    if ks_df.empty:
        st.warning("No drifted features available for this scenario.")
    else:
        st.dataframe(
            ks_df[['feature', 'ks_statistic', 'p_value', 'drifted', 'mean_diff']]
            .sort_values('ks_statistic', ascending=False)
            .head(10),
            use_container_width=True,
            hide_index=True
        )