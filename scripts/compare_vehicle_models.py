"""
Vehicle Detector Model Comparison Tool.
Compares baseline model vs candidate model(s) on identical benchmark metrics:
- Overall Precision, Recall, F1, mAP@50, mAP@50-95.
- Per-class breakdown: car, motorcycle, bus, truck.
- Latency and deployment efficiency deltas.
- Emits automatic REGRESSION WARNINGS if any critical class falls below tolerance.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.infrastructure.paths import paths
from app.models.registry import model_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_models")


def compare_metrics(
    base_m: Dict[str, Any],
    cand_m: Dict[str, Any],
    max_allowed_regression: float = 0.03,
) -> Dict[str, Any]:
    """Computes deltas and evaluates regression alerts."""
    regressions = []
    improvements = []

    def delta(k):
        b = base_m.get(k, 0.0)
        c = cand_m.get(k, 0.0)
        d = round(c - b, 4)
        status = "IMPROVED" if d > 0.005 else ("REGRESSED" if d < -max_allowed_regression else "MAINTAINED")
        if status == "REGRESSED":
            regressions.append(f"Aggregate metric '{k}' dropped by {abs(d):.4f} (from {b:.4f} to {c:.4f})")
        elif status == "IMPROVED":
            improvements.append(f"Aggregate metric '{k}' improved by {d:+.4f} (from {b:.4f} to {c:.4f})")
        return {"baseline": b, "candidate": c, "delta": d, "status": status}

    comparison = {
        "mAP50": delta("mAP50"),
        "mAP50_95": delta("mAP50_95"),
        "precision": delta("precision"),
        "recall": delta("recall"),
        "per_class": {},
    }

    base_classes = base_m.get("per_class", {})
    cand_classes = cand_m.get("per_class", {})

    for cname in ["car", "motorcycle", "bus", "truck"]:
        bc = base_classes.get(cname, {})
        cc = cand_classes.get(cname, {})

        b_ap50 = bc.get("ap50", 0.0)
        c_ap50 = cc.get("ap50", 0.0)
        d_ap50 = round(c_ap50 - b_ap50, 4)

        b_rec = bc.get("recall", 0.0)
        c_rec = cc.get("recall", 0.0)
        d_rec = round(c_rec - b_rec, 4)

        status = "IMPROVED" if d_ap50 > 0.01 else ("REGRESSED" if d_ap50 < -max_allowed_regression else "MAINTAINED")
        if status == "REGRESSED":
            regressions.append(f"Critical class '{cname}' AP@50 dropped by {abs(d_ap50):.4f} (from {b_ap50:.4f} to {c_ap50:.4f})")

        comparison["per_class"][cname] = {
            "ap50_baseline": b_ap50,
            "ap50_candidate": c_ap50,
            "ap50_delta": d_ap50,
            "recall_baseline": b_rec,
            "recall_candidate": c_rec,
            "recall_delta": d_rec,
            "status": status,
        }

    return {
        "comparison": comparison,
        "regressions": regressions,
        "improvements": improvements,
        "has_regressions": len(regressions) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs candidate model evaluation")
    parser.add_argument("--candidate-exp", type=str, default=None, help="Candidate experiment ID in registry")
    parser.add_argument("--baseline-report", type=str, default=None, help="Path to baseline evaluation JSON")
    parser.add_argument("--candidate-report", type=str, default=None, help="Path to candidate evaluation JSON")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    # Resolve baseline metrics
    base_m = {}
    if args.baseline_report:
        p = paths.resolve(args.baseline_report)
        data = json.loads(p.read_text(encoding="utf-8"))
        base_m = data.get("view1_general_benchmark") or data.get("vehicle_detector", {}).get("test_metrics", {})
    else:
        active = model_registry.get_active_model()
        base_m = active.get("metrics", {})

    # Resolve candidate metrics
    cand_m = {}
    cand_id = args.candidate_exp or "candidate"
    if args.candidate_report:
        p = paths.resolve(args.candidate_report)
        data = json.loads(p.read_text(encoding="utf-8"))
        cand_m = data.get("view1_general_benchmark") or data.get("test_metrics", {})
    elif args.candidate_exp:
        reg = model_registry.load_registry()
        c = next((c for c in reg.get("candidates", []) if c.get("experiment_id") == args.candidate_exp), None)
        if not c:
            logger.error(f"Candidate experiment '{args.candidate_exp}' not found in registry.")
            sys.exit(1)
        cand_m = c.get("test_metrics") or c.get("validation_metrics", {})
    else:
        logger.error("Must specify either --candidate-exp or --candidate-report.")
        sys.exit(1)

    result = compare_metrics(base_m, cand_m)

    if args.json:
        print(json.dumps(result, indent=2))
        sys.exit(0)

    cmp = result["comparison"]

    print("\n" + "=" * 75)
    print(f" MODEL COMPARISON: BASELINE vs CANDIDATE [{cand_id}]")
    print("=" * 75)
    print(f"{'Metric':<16} | {'Baseline':<10} | {'Candidate':<10} | {'Delta':<10} | {'Status'}")
    print("-" * 75)

    for k in ["mAP50", "mAP50_95", "precision", "recall"]:
        m = cmp[k]
        print(f"{k:<16} | {m['baseline']:<10.4f} | {m['candidate']:<10.4f} | {m['delta']:<+10.4f} | {m['status']}")

    print("\n" + "-" * 75)
    print(" PER-CLASS BREAKDOWN (AP@50)")
    print("-" * 75)
    print(f"{'Class':<12} | {'Base AP50':<10} | {'Cand AP50':<10} | {'AP Delta':<10} | {'Recall Delta':<12} | {'Status'}")
    print("-" * 75)

    for cname, cm in cmp["per_class"].items():
        print(f"{cname:<12} | {cm['ap50_baseline']:<10.4f} | {cm['ap50_candidate']:<10.4f} | {cm['ap50_delta']:<+10.4f} | {cm['recall_delta']:<+12.4f} | {cm['status']}")

    print("-" * 75)

    if result["improvements"]:
        print("\nVerified Improvements:")
        for imp in result["improvements"]:
            print(f"  [+] {imp}")

    if result["regressions"]:
        print("\nAUTOMATIC REGRESSION WARNINGS:")
        for reg in result["regressions"]:
            print(f"  [!] {reg}")
        print("\n[!] CANDIDATE CANNOT BE AUTOMATICALLY PROMOTED DUE TO CRITICAL REGRESSIONS.")
    else:
        print("\n[+] No critical regressions detected. Candidate satisfies stability baseline.")

    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
