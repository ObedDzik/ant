import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def bootstrap_auc(y_true, y_score, n_bootstrap=1000, ci=95):
    y_true, y_score = np.array(y_true), np.array(y_score)
    if len(np.unique(y_true)) < 2:
        return None, None, None

    rng = np.random.RandomState(42)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))

    if not aucs:
        return None, None, None

    alpha = (100 - ci) / 2
    return float(np.mean(aucs)), float(np.percentile(aucs, alpha)), float(np.percentile(aucs, 100 - alpha))


def bootstrap_sensitivity_at_specificity(y_true, y_score,
                                          target_specificity,
                                          n_bootstrap=1000, ci=95):
    """Mirrors calculate_binary_classification_metrics exactly:
       sensitivity = tpr[np.argmax(fpr > 1 - specificity)]
    """
    y_true, y_score = np.array(y_true), np.array(y_score)
    if len(np.unique(y_true)) < 2:
        return None, None, None

    def _sens_at_spec(yt, ys, target_spec):
        fpr, tpr, _ = roc_curve(yt, ys)
        return float(tpr[np.argmax(fpr > 1 - target_spec)])  # matches original exactly

    rng = np.random.RandomState(42)
    sens_vals = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        sens_vals.append(_sens_at_spec(y_true[idx], y_score[idx], target_specificity))

    if not sens_vals:
        return None, None, None

    alpha = (100 - ci) / 2
    return float(np.mean(sens_vals)), float(np.percentile(sens_vals, alpha)), float(np.percentile(sens_vals, 100 - alpha))
    

def compute_bootstrap_metrics(results_table, desc, n_bootstrap=1000, ci=95):
    """
    Column mappings derived from ProstNFoundMetricsCalculator:

    metric key                    | label                      | score
    ------------------------------|----------------------------|------------------------------
    auc                           | label                      | average_needle_heatmap_value
    auc_heatmap_cspca             | grade_group > 2            | average_needle_heatmap_value
    sens_at_80_spe_heatmap_cspca  | grade_group > 2            | average_needle_heatmap_value
    sens_at_60_spe_heatmap_cspca  | grade_group > 2            | average_needle_heatmap_value
    auc_image_level_cspca         | grade_group > 2            | image_level_cancer_logits
    """
    if results_table is None or len(results_table) == 0:
        return {}

    heatmap_scores = results_table["average_needle_heatmap_value"].values
    pca_labels     = results_table["label"].values
    cspca_labels   = (results_table["grade_group"].values > 2).astype(int)

    # Drop NaNs from heatmap scores (mirrors calculate_binary_classification_metrics)
    valid = ~np.isnan(heatmap_scores)
    heatmap_scores = heatmap_scores[valid]
    pca_labels     = pca_labels[valid]
    cspca_labels   = cspca_labels[valid]

    bootstrap_metrics = {}

    def _add(metric_name, mean, lower, upper):
        if mean is None:
            return
        bootstrap_metrics[f"{desc}/bootstrap_{metric_name}_mean"]  = mean
        bootstrap_metrics[f"{desc}/bootstrap_{metric_name}_lower"] = lower
        bootstrap_metrics[f"{desc}/bootstrap_{metric_name}_upper"] = upper

    # 1. Core-level PCa AUC
    _add("auc",
         *bootstrap_auc(pca_labels, heatmap_scores, n_bootstrap=n_bootstrap, ci=ci))

    # 2. Core-level csPCa AUC
    _add("auc_heatmap_cspca",
         *bootstrap_auc(cspca_labels, heatmap_scores, n_bootstrap=n_bootstrap, ci=ci))

    # 3 & 4. Core-level csPCa sensitivity @ 80% / 60% specificity
    _add("sens_at_80_spe_heatmap_cspca",
         *bootstrap_sensitivity_at_specificity(
             cspca_labels, heatmap_scores,
             target_specificity=0.80, n_bootstrap=n_bootstrap, ci=ci))

    _add("sens_at_60_spe_heatmap_cspca",
         *bootstrap_sensitivity_at_specificity(
             cspca_labels, heatmap_scores,
             target_specificity=0.60, n_bootstrap=n_bootstrap, ci=ci))

    # 5. Image-level csPCa AUC — separate NaN mask for logits
    if "image_level_cancer_logits" in results_table.columns:
        logits = results_table["image_level_cancer_logits"].values
        cspca_all = (results_table["grade_group"].values > 2).astype(int)
        valid_logits = ~np.isnan(logits)
        _add("auc_image_level_cspca",
             *bootstrap_auc(
                 cspca_all[valid_logits],
                 logits[valid_logits],
                 n_bootstrap=n_bootstrap, ci=ci))

    return bootstrap_metrics