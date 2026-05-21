import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Features monitored for drift per spec Section 9.3
PSI_FEATURES = ["TransactionAmt", "amt_log", "card1_amt_mean"]
PSI_THRESHOLD = 0.2
KS_THRESHOLD = 0.05

REFERENCE_PATH = Path("data/monitoring/reference.csv")
LOG_PATH = Path("logs/inference.jsonl")


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between reference and current distributions.
    PSI < 0.1  : stable
    PSI < 0.2  : moderate shift
    PSI >= 0.2 : significant shift — trigger retraining review
    """
    combined = np.concatenate([reference, current])
    edges = np.percentile(combined, np.linspace(0, 100, bins + 1))
    edges = np.unique(edges)

    counts_ref, _ = np.histogram(reference, bins=edges)
    counts_cur, _ = np.histogram(current, bins=edges)

    prop_ref = counts_ref / counts_ref.sum()
    prop_cur = counts_cur / counts_cur.sum()

    prop_ref = np.clip(prop_ref, 1e-10, None)
    prop_cur = np.clip(prop_cur, 1e-10, None)

    psi = np.sum((prop_cur - prop_ref) * np.log(prop_cur / prop_ref))
    return float(psi)


def compute_ks(reference: np.ndarray, current: np.ndarray) -> dict:
    """
    KS test on prediction score distributions.
    p-value < 0.05 means the distributions are significantly different.
    """
    result = stats.ks_2samp(reference, current)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "drift_detected": result.pvalue < KS_THRESHOLD,
    }


def generate_drift_report(
    reference_path: Path = REFERENCE_PATH,
    log_path: Path = LOG_PATH,
    output_path: Path = None,
) -> dict:
    """
    Load reference data and inference logs, compute PSI and KS, return report.
    Run manually or on a schedule — not called during inference.
    """
    reference = pd.read_csv(reference_path)

    if not log_path.exists():
        raise FileNotFoundError(
            f"Inference log not found at {log_path}. "
            "Run /predict at least once before generating a report."
        )

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) < 30:
        raise ValueError(
            f"Only {len(records)} inference records found. "
            "Need at least 30 to compute meaningful drift metrics."
        )

    logs = pd.DataFrame(records)

    psi_results = {}
    for feature in PSI_FEATURES:
        if feature not in reference.columns:
            logger.warning("Feature %s not in reference data — skipping PSI.", feature)
            continue
        if feature not in logs.columns:
            logger.warning("Feature %s not in inference logs — skipping PSI.", feature)
            continue

        psi_value = compute_psi(
            reference[feature].dropna().values,
            logs[feature].dropna().values,
        )
        psi_results[feature] = {
            "psi": round(psi_value, 4),
            "drift_detected": psi_value >= PSI_THRESHOLD,
        }

    ks_result = compute_ks(
        reference["fraud_probability"].dropna().values
        if "fraud_probability" in reference.columns
        else np.zeros(len(reference)),
        logs["fraud_probability"].dropna().values,
    )

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_reference": len(reference),
        "n_current": len(logs),
        "psi_threshold": PSI_THRESHOLD,
        "ks_threshold": KS_THRESHOLD,
        "feature_drift": psi_results,
        "prediction_drift": ks_result,
        "action_required": (
            any(v["drift_detected"] for v in psi_results.values())
            or ks_result["drift_detected"]
        ),
    }

    output_path = Path(output_path) if output_path else Path("logs/drift_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Drift report written to %s", output_path)
    return report
