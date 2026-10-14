"""
Gated Vehicle Model Promotion and Rollback Tool.
Enforces multi-criteria promotion gates before any candidate replaces the production model.
Provides instantaneous, deterministic rollback with SHA-256 verification.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure workspace root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.models.registry import model_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("promote_model")


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Gated Model Promotion & Rollback Manager")
    parser.add_argument("--candidate", type=str, help="Experiment ID of the candidate model to promote")
    parser.add_argument("--min-map50-gain", type=float, default=0.0, help="Required minimum mAP@50 delta (default: 0.0, non-regression)")
    parser.add_argument("--max-class-regression", type=float, default=0.05, help="Maximum allowed AP regression on critical classes (default: 0.05)")
    parser.add_argument("--force", action="store_true", help="Force promotion bypassing gate checks (requires manual review)")
    parser.add_argument("--rollback", action="store_true", help="Roll back production model to previous stable version")
    parser.add_argument("--list", action="store_true", help="List registered candidate models and active version")
    parser.add_argument("--status", action="store_true", help="Print active model status and artifact checksums")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    reg_data = model_registry.load_registry()

    if args.status:
        active = model_registry.get_active_model()
        prev = reg_data.get("previous_stable_model")
        if args.json:
            print(json.dumps({"active_model": active, "previous_stable_model": prev}, indent=2))
        else:
            print_banner("Active Production Model Status")
            print(f"Model ID:            {active.get('model_id')}")
            print(f"Version:             v{active.get('version')} ({active.get('status')})")
            print(f"Promoted At:         {active.get('promoted_at')}")
            print(f"Deployment ONNX:     {active.get('onnx_path')}")
            print(f"ONNX SHA-256:        {active.get('onnx_sha256')}")
            print(f"Integrity Verified:  {active.get('integrity_verified')}")
            m = active.get("metrics", {})
            print(f"Metrics (mAP50):     {m.get('mAP50', 'N/A')}")
            print(f"Metrics (mAP50-95):  {m.get('mAP50_95', 'N/A')}")
            print(f"Rollback Available:  {'YES (v' + prev.get('version') + ')' if prev else 'NO'}")
            print("=" * 70 + "\n")
        sys.exit(0)

    if args.list:
        candidates = reg_data.get("candidates", [])
        active = reg_data.get("active_model", {})
        if args.json:
            print(json.dumps({"active_model": active, "candidates": candidates}, indent=2))
        else:
            print_banner("Registered Models & Experimental Candidates")
            print(f"[ACTIVE] v{active.get('version')} - {active.get('model_id')} (mAP50: {active.get('metrics', {}).get('mAP50')})")
            print("-" * 70)
            if not candidates:
                print("No candidate models registered yet.")
            for c in candidates:
                test_m = c.get("test_metrics", {})
                print(f"[{c.get('status')}] {c.get('experiment_id')} - Registered: {c.get('registered_at')}")
                print(f"       mAP50: {test_m.get('mAP50', 'Pending')} | mAP50-95: {test_m.get('mAP50_95', 'Pending')}")
                print(f"       Notes: {c.get('notes', 'None')}")
            print("=" * 70 + "\n")
        sys.exit(0)

    if args.rollback:
        print_banner("Initiating Production Model Rollback")
        success, msg = model_registry.rollback_model()
        if success:
            logger.info(f"[SUCCESS] {msg}")
            if args.json:
                print(json.dumps({"success": True, "message": msg}))
            else:
                print(f"\n[+] ROLLBACK COMPLETE: {msg}\n")
            sys.exit(0)
        else:
            logger.error(f"[FAILED] {msg}")
            if args.json:
                print(json.dumps({"success": False, "error": msg}))
            else:
                print(f"\n[-] ROLLBACK FAILED: {msg}\n")
            sys.exit(1)

    if args.candidate:
        print_banner(f"Evaluating Candidate Promotion: {args.candidate}")
        candidate = next((c for c in reg_data.get("candidates", []) if c.get("experiment_id") == args.candidate), None)
        if not candidate:
            logger.error(f"Candidate '{args.candidate}' not found in registry.")
            sys.exit(1)

        # Gate Evaluation
        gate_res = model_registry.evaluate_promotion_gates(
            candidate,
            min_mAP50_improvement=args.min_map50_gain,
            max_class_regression=args.max_class_regression,
        )

        print("\n--- PROMOTION GATE AUDIT REPORT ---")
        print(f"Eligible for Production: {gate_res.eligible or args.force}")
        if gate_res.metrics_summary:
            print(f"Baseline mAP@50:         {gate_res.metrics_summary.get('baseline_mAP50'):.4f}")
            print(f"Candidate mAP@50:        {gate_res.metrics_summary.get('candidate_mAP50'):.4f}")
            print(f"Delta mAP@50:            {gate_res.metrics_summary.get('delta_mAP50'):+.4f}")
            print(f"Baseline mAP@50-95:      {gate_res.metrics_summary.get('baseline_mAP50_95'):.4f}")
            print(f"Candidate mAP@50-95:     {gate_res.metrics_summary.get('candidate_mAP50_95'):.4f}")
            print(f"Delta mAP@50-95:         {gate_res.metrics_summary.get('delta_mAP50_95'):+.4f}")

        if gate_res.warnings:
            print("\nGate Warnings:")
            for w in gate_res.warnings:
                print(f"  [!] {w}")

        if not gate_res.eligible:
            print("\nBlocking Promotion Failures:")
            for r in gate_res.reasons:
                print(f"  [-] {r}")
            if not args.force:
                print("\n[!] PROMOTION BLOCKED. Use --force to override if explicitly authorized by CV lead.")
                print("=" * 70 + "\n")
                if args.json:
                    print(json.dumps({"success": False, "reasons": gate_res.reasons, "warnings": gate_res.warnings}))
                sys.exit(1)
            else:
                print("\n[!] FORCE FLAG ACTIVE: Proceeding with promotion override despite gate failures...")

        # Execute promotion
        success, msg = model_registry.promote_candidate(
            args.candidate,
            force=args.force,
            min_mAP50_improvement=args.min_map50_gain,
        )

        if success:
            print(f"\n[+] PROMOTION SUCCEEDED: {msg}")
            print(f"[+] Active deployment models updated.")
            print(f"[+] Prior model archived for instant rollback if needed.")
            print("=" * 70 + "\n")
            if args.json:
                print(json.dumps({"success": True, "message": msg}))
            sys.exit(0)
        else:
            print(f"\n[-] PROMOTION FAILED: {msg}")
            print("=" * 70 + "\n")
            if args.json:
                print(json.dumps({"success": False, "error": msg}))
            sys.exit(1)

    parser.print_help()


if __name__ == "__main__":
    main()
