"""CLI entry point for in-app diagnostics: python -m app.diagnostics --quick/--full"""

import argparse
import sys
from app.diagnostics.health import DiagnosticsRunner


def main():
    parser = argparse.ArgumentParser(description="Smart Parking System Self-Diagnostics Utility")
    parser.add_argument("--quick", action="store_true", help="Run quick non-destructive startup health checks.")
    parser.add_argument("--full", action="store_true", help="Run full diagnostic suite with image quality and resource checks.")
    args = parser.parse_args()

    run_type = "full" if args.full else "quick"
    print(f"\n[+] Executing Smart Parking {run_type.upper()} Diagnostics...\n" + "=" * 65)

    runner = DiagnosticsRunner()
    report = runner.run(run_type=run_type)

    for r in report.results:
        symbol = "+" if r.status == "PASS" else ("!" if r.status == "WARN" else "X")
        print(f"[{symbol}] {r.name:<40} [{r.status:<5}] ({r.duration_ms} ms)")
        if r.status != "PASS" and r.message:
            print(f"    -> {r.message}")
            if r.corrective_action:
                print(f"    -> Action: {r.corrective_action}")

    print("=" * 65)
    print(f"Overall Result: {report.overall_status} | Passed: {report.summary['pass']} | Warnings: {report.summary['warn']} | Failed: {report.summary['fail']}")
    print(f"Report saved to: data/test_reports/{report.report_id}.json and .md\n")

    sys.exit(0 if report.overall_status in ["PASS", "WARN"] else 1)


if __name__ == "__main__":
    main()
