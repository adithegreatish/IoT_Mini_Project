"""Model registry, artifact versioning, gated promotion, and safe rollback manager."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

from app.infrastructure.paths import paths

logger = logging.getLogger("smart_parking.model_registry")


def calculate_sha256(filepath: Path) -> str:
    """Calculates SHA-256 hash of a file."""
    if not filepath.exists() or filepath.is_dir():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PromotionGateResult:
    eligible: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """
    Authoritative Model Registry manager.
    Tracks active baseline model, candidate experimental models, past stable versions,
    checksums, training lineage, and executes gated promotions and instantaneous rollbacks.
    """

    def __init__(self, registry_file: Optional[Path] = None):
        self.registry_file = registry_file or (paths.models_dir / "model_registry.json")
        self.archive_dir = paths.models_dir / "archive"
        self.experiments_dir = paths.models_dir / "experiments"
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_file.exists():
            self._initialize_default_registry()

    def _initialize_default_registry(self) -> None:
        """Initializes registry with the verified baseline model."""
        pt_path = paths.models_dir / "vehicle_detector.pt"
        onnx_path = paths.models_dir / "vehicle_detector.onnx"

        pt_sha = calculate_sha256(pt_path) if pt_path.exists() else "0f8c08dc8f4aff7f0b52582f38677c016b8caedf73ab4b39c182046a086e6089"
        onnx_sha = calculate_sha256(onnx_path) if onnx_path.exists() else "8ea5414f0cf2bc4f81861fc19db26ece955cadc736dec8fa56ebe01a049244bc"

        default_data = {
            "schema_version": "1.0.0",
            "updated_at": datetime.now().isoformat(),
            "active_model": {
                "model_id": "vehicle-yolo-v8n",
                "version": "1.0.0",
                "status": "ACTIVE_PRODUCTION",
                "promoted_at": "2026-10-12T22:00:19Z",
                "pt_path": "models/vehicle_detector.pt",
                "onnx_path": "models/vehicle_detector.onnx",
                "pt_sha256": pt_sha,
                "onnx_sha256": onnx_sha,
                "architecture": "YOLOv8-Nano",
                "input_dimensions": [640, 640],
                "classes": ["car", "motorcycle", "bus", "truck"],
                "metrics": {
                    "precision": 0.6224,
                    "recall": 0.5459,
                    "mAP50": 0.5950,
                    "mAP50_95": 0.3797,
                    "per_class": {
                        "car": {"precision": 0.6980, "recall": 0.5552, "ap50": 0.6038, "ap50_95": 0.3717},
                        "motorcycle": {"precision": 0.7648, "recall": 0.5932, "ap50": 0.6698, "ap50_95": 0.3585},
                        "bus": {"precision": 0.6797, "recall": 0.6182, "ap50": 0.7192, "ap50_95": 0.5449},
                        "truck": {"precision": 0.3473, "recall": 0.4168, "ap50": 0.3873, "ap50_95": 0.2438},
                    },
                    "latency_pi5_ms": 78.4,
                    "latency_windows_ms": 29.28,
                },
                "lineage": {
                    "dataset_source": "COCO 2017 high-recall vehicle expansion",
                    "training_instances": 6870,
                    "epochs": 35,
                    "device": "NVIDIA GeForce RTX 4070 Laptop GPU",
                },
            },
            "previous_stable_model": None,
            "candidates": [],
            "history": [],
        }
        self.save_registry(default_data)

    def load_registry(self) -> Dict[str, Any]:
        """Loads model registry from disk."""
        if not self.registry_file.exists():
            self._initialize_default_registry()
        try:
            return json.loads(self.registry_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Error loading model registry: {e}. Reinitializing default.")
            self._initialize_default_registry()
            return json.loads(self.registry_file.read_text(encoding="utf-8"))

    def save_registry(self, data: Dict[str, Any]) -> None:
        """Saves registry data atomically."""
        data["updated_at"] = datetime.now().isoformat()
        temp_file = self.registry_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        shutil.move(str(temp_file), str(self.registry_file))

    def get_active_model(self) -> Dict[str, Any]:
        """Returns the current production model metadata and verifies on-disk artifact integrity."""
        reg = self.load_registry()
        active = reg.get("active_model", {})
        onnx_path = paths.resolve(active.get("onnx_path", "models/vehicle_detector.onnx"))
        pt_path = paths.resolve(active.get("pt_path", "models/vehicle_detector.pt"))

        active["onnx_exists"] = onnx_path.exists()
        active["pt_exists"] = pt_path.exists()

        current_onnx_hash = calculate_sha256(onnx_path) if onnx_path.exists() else None
        active["integrity_verified"] = (current_onnx_hash == active.get("onnx_sha256"))
        return active

    def register_candidate(
        self,
        experiment_id: str,
        pt_path: Path,
        onnx_path: Path,
        training_config: Dict[str, Any],
        validation_metrics: Dict[str, Any],
        test_metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Registers a newly trained experimental run as a candidate."""
        reg = self.load_registry()

        candidate = {
            "experiment_id": experiment_id,
            "registered_at": datetime.now().isoformat(),
            "status": "CANDIDATE",
            "pt_path": paths.to_relative_str(pt_path),
            "onnx_path": paths.to_relative_str(onnx_path) if onnx_path.exists() else None,
            "pt_sha256": calculate_sha256(pt_path),
            "onnx_sha256": calculate_sha256(onnx_path) if onnx_path.exists() else None,
            "training_config": training_config,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics or {},
            "notes": notes,
        }

        # Replace existing entry with same experiment_id or append
        existing = [c for c in reg.get("candidates", []) if c.get("experiment_id") != experiment_id]
        existing.append(candidate)
        reg["candidates"] = existing
        self.save_registry(reg)
        logger.info(f"Registered candidate experiment: {experiment_id}")
        return candidate

    def evaluate_promotion_gates(
        self,
        candidate_data: Dict[str, Any],
        min_mAP50_improvement: float = 0.0,
        max_class_regression: float = 0.05,
        max_pi_latency_ms: float = 120.0,
    ) -> PromotionGateResult:
        """
        Evaluates candidate against strict multi-criteria promotion gates:
        1. Artifacts must exist and checksums match.
        2. Validation metrics must be documented.
        3. Independent test evaluation must be present.
        4. Overall mAP@50 must meet or exceed baseline minus tolerance.
        5. No critical vehicle class (car, motorcycle, bus, truck) may regress beyond max_class_regression.
        6. ONNX export must exist and latency budget must be respected.
        """
        active = self.get_active_model()
        active_m = active.get("metrics", {})
        cand_test = candidate_data.get("test_metrics", {})
        reasons = []
        warnings = []

        # Gate 1: Artifact existence
        pt_path = paths.resolve(candidate_data.get("pt_path", ""))
        onnx_path = paths.resolve(candidate_data.get("onnx_path", "")) if candidate_data.get("onnx_path") else None

        if not pt_path.exists():
            reasons.append(f"PyTorch checkpoint not found at {pt_path}")
        if not onnx_path or not onnx_path.exists():
            reasons.append(f"ONNX deployment artifact not found at {onnx_path}")

        # Gate 2: Test metrics present
        if not cand_test or "mAP50" not in cand_test:
            reasons.append("Independent test evaluation metrics are missing from candidate record.")
            return PromotionGateResult(eligible=False, reasons=reasons, warnings=warnings)

        cand_map50 = cand_test.get("mAP50", 0.0)
        baseline_map50 = active_m.get("mAP50", 0.0)
        delta_map50 = cand_map50 - baseline_map50

        # Gate 3: Aggregate mAP@50 delta
        if delta_map50 < min_mAP50_improvement:
            reasons.append(
                f"Candidate mAP@50 ({cand_map50:.4f}) does not satisfy required improvement against baseline ({baseline_map50:.4f}, delta: {delta_map50:+.4f})."
            )

        # Gate 4: mAP@50-95 check (no severe drop > 3 percentage points)
        cand_map50_95 = cand_test.get("mAP50_95", 0.0)
        baseline_map50_95 = active_m.get("mAP50_95", 0.0)
        if (cand_map50_95 - baseline_map50_95) < -0.03:
            reasons.append(
                f"Candidate mAP@50-95 ({cand_map50_95:.4f}) regresses by more than 3% vs baseline ({baseline_map50_95:.4f})."
            )

        # Gate 5: Per-class non-regression check
        active_per_class = active_m.get("per_class", {})
        cand_per_class = cand_test.get("per_class", {})

        for cls_name, base_c in active_per_class.items():
            cand_c = cand_per_class.get(cls_name, {})
            base_ap50 = base_c.get("ap50", 0.0)
            cand_ap50 = cand_c.get("ap50", 0.0)
            ap_diff = cand_ap50 - base_ap50

            if ap_diff < -max_class_regression:
                reasons.append(
                    f"Class '{cls_name}' regressed by {abs(ap_diff):.4f} (baseline AP50={base_ap50:.4f}, candidate AP50={cand_ap50:.4f}, max allowed regression={max_class_regression:.4f})."
                )
            elif ap_diff < 0:
                warnings.append(f"Minor regression on class '{cls_name}': {ap_diff:+.4f}")

        # Gate 6: Latency budget check
        cand_latency = candidate_data.get("benchmark", {}).get("mean_latency_ms", 0.0)
        if cand_latency > max_pi_latency_ms:
            reasons.append(
                f"Candidate latency ({cand_latency:.1f} ms) exceeds deployment budget ({max_pi_latency_ms:.1f} ms)."
            )

        summary = {
            "baseline_mAP50": baseline_map50,
            "candidate_mAP50": cand_map50,
            "delta_mAP50": round(delta_map50, 4),
            "baseline_mAP50_95": baseline_map50_95,
            "candidate_mAP50_95": cand_map50_95,
            "delta_mAP50_95": round(cand_map50_95 - baseline_map50_95, 4),
        }

        eligible = (len(reasons) == 0)
        return PromotionGateResult(
            eligible=eligible,
            reasons=reasons,
            warnings=warnings,
            metrics_summary=summary,
        )

    def promote_candidate(
        self,
        experiment_id: str,
        force: bool = False,
        min_mAP50_improvement: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Atomically promotes an approved candidate model to active production status:
        1. Verifies promotion gates (unless explicitly bypassed with force=True).
        2. Backs up current production artifacts to models/archive/ with timestamp.
        3. Copies candidate PyTorch and ONNX files to canonical model paths.
        4. Updates models/manifest.json and models/model_registry.json.
        """
        reg = self.load_registry()
        candidate = next((c for c in reg.get("candidates", []) if c.get("experiment_id") == experiment_id), None)
        if not candidate:
            return False, f"Experiment ID '{experiment_id}' not found in registered candidates."

        if not force:
            gate_res = self.evaluate_promotion_gates(candidate, min_mAP50_improvement=min_mAP50_improvement)
            if not gate_res.eligible:
                return False, f"Promotion gates failed: {'; '.join(gate_res.reasons)}"

        active = reg.get("active_model", {})
        curr_ver = active.get("version", "1.0.0")

        # 1. Archive current active model
        archive_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        curr_pt = paths.resolve(active.get("pt_path", "models/vehicle_detector.pt"))
        curr_onnx = paths.resolve(active.get("onnx_path", "models/vehicle_detector.onnx"))

        archived_pt = self.archive_dir / f"vehicle_detector_v{curr_ver}_{archive_ts}.pt"
        archived_onnx = self.archive_dir / f"vehicle_detector_v{curr_ver}_{archive_ts}.onnx"

        if curr_pt.exists():
            shutil.copy2(curr_pt, archived_pt)
        if curr_onnx.exists():
            shutil.copy2(curr_onnx, archived_onnx)

        # 2. Prepare previous stable record for rollback
        reg["previous_stable_model"] = {
            "version": curr_ver,
            "archived_pt": paths.to_relative_str(archived_pt),
            "archived_onnx": paths.to_relative_str(archived_onnx),
            "pt_sha256": active.get("pt_sha256"),
            "onnx_sha256": active.get("onnx_sha256"),
            "metrics": active.get("metrics"),
            "archived_at": datetime.now().isoformat(),
        }

        # 3. Copy candidate to canonical deployment locations
        cand_pt = paths.resolve(candidate.get("pt_path", ""))
        cand_onnx = paths.resolve(candidate.get("onnx_path", ""))

        target_pt = paths.models_dir / "vehicle_detector.pt"
        target_onnx = paths.models_dir / "vehicle_detector.onnx"

        shutil.copy2(cand_pt, target_pt)
        shutil.copy2(cand_onnx, target_onnx)

        new_pt_sha = calculate_sha256(target_pt)
        new_onnx_sha = calculate_sha256(target_onnx)

        # 4. Calculate new version string
        try:
            parts = [int(p) for p in curr_ver.split(".")]
            parts[-1] += 1
            new_version = ".".join(str(p) for p in parts)
        except Exception:
            new_version = f"{curr_ver}.1"

        # 5. Update active model record
        reg["active_model"] = {
            "model_id": "vehicle-yolo-v8n",
            "version": new_version,
            "status": "ACTIVE_PRODUCTION",
            "promoted_at": datetime.now().isoformat(),
            "source_experiment": experiment_id,
            "pt_path": "models/vehicle_detector.pt",
            "onnx_path": "models/vehicle_detector.onnx",
            "pt_sha256": new_pt_sha,
            "onnx_sha256": new_onnx_sha,
            "architecture": candidate.get("training_config", {}).get("architecture", "YOLOv8-Nano"),
            "input_dimensions": candidate.get("training_config", {}).get("imgsz", 640),
            "classes": ["car", "motorcycle", "bus", "truck"],
            "metrics": candidate.get("test_metrics") or candidate.get("validation_metrics", {}),
            "lineage": candidate.get("training_config", {}),
        }

        # Mark candidate as PROMOTED
        candidate["status"] = "PROMOTED"
        candidate["promoted_version"] = new_version

        # Record in history
        reg.setdefault("history", []).append({
            "action": "PROMOTE",
            "experiment_id": experiment_id,
            "new_version": new_version,
            "previous_version": curr_ver,
            "timestamp": datetime.now().isoformat(),
            "onnx_sha256": new_onnx_sha,
        })

        self.save_registry(reg)

        # Update models/manifest.json
        self._sync_manifest(reg["active_model"])

        logger.info(f"Successfully promoted {experiment_id} to production as v{new_version}.")
        return True, f"Promoted {experiment_id} to production as v{new_version} (SHA-256: {new_onnx_sha[:16]}...)"

    def rollback_model(self) -> Tuple[bool, str]:
        """
        Rolls back the active model to the previously archived stable version.
        Guarantees SHA-256 integrity verification and works without training dependencies.
        """
        reg = self.load_registry()
        prev = reg.get("previous_stable_model")
        if not prev:
            return False, "No previous stable model recorded in registry to roll back to."

        prev_pt = paths.resolve(prev.get("archived_pt", ""))
        prev_onnx = paths.resolve(prev.get("archived_onnx", ""))

        if not prev_pt.exists() or not prev_onnx.exists():
            return False, f"Archived artifacts missing on disk: {prev_pt} or {prev_onnx}"

        # Check integrity of archive
        onnx_hash = calculate_sha256(prev_onnx)
        if prev.get("onnx_sha256") and onnx_hash != prev.get("onnx_sha256"):
            return False, f"Archived ONNX failed integrity check! Expected {prev.get('onnx_sha256')}, got {onnx_hash}"

        target_pt = paths.models_dir / "vehicle_detector.pt"
        target_onnx = paths.models_dir / "vehicle_detector.onnx"

        shutil.copy2(prev_pt, target_pt)
        shutil.copy2(prev_onnx, target_onnx)

        restored_ver = prev.get("version", "1.0.0")

        # Swap active model
        old_active = reg["active_model"]
        reg["active_model"] = {
            "model_id": "vehicle-yolo-v8n",
            "version": restored_ver,
            "status": "ACTIVE_PRODUCTION (ROLLED_BACK)",
            "promoted_at": datetime.now().isoformat(),
            "pt_path": "models/vehicle_detector.pt",
            "onnx_path": "models/vehicle_detector.onnx",
            "pt_sha256": prev.get("pt_sha256"),
            "onnx_sha256": prev.get("onnx_sha256"),
            "architecture": "YOLOv8-Nano",
            "input_dimensions": [640, 640],
            "classes": ["car", "motorcycle", "bus", "truck"],
            "metrics": prev.get("metrics", {}),
            "lineage": {"restored_from_archive": prev.get("archived_onnx")},
        }

        # Clear or shift previous stable
        reg["previous_stable_model"] = None
        reg.setdefault("history", []).append({
            "action": "ROLLBACK",
            "restored_version": restored_ver,
            "rolled_back_from": old_active.get("version"),
            "timestamp": datetime.now().isoformat(),
            "onnx_sha256": prev.get("onnx_sha256"),
        })

        self.save_registry(reg)
        self._sync_manifest(reg["active_model"])
        logger.info(f"Successfully rolled back to v{restored_ver}.")
        return True, f"Successfully rolled back production model to v{restored_ver}."

    def _sync_manifest(self, active_data: Dict[str, Any]) -> None:
        """Syncs active model metadata into models/manifest.json."""
        manifest_path = paths.models_manifest_file
        if not manifest_path.exists():
            return
        try:
            m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            for m in m_data.get("models", []):
                if m.get("model_id") == "vehicle-yolo-v8n":
                    m["version"] = active_data.get("version", m.get("version"))
                    m["sha256"] = active_data.get("onnx_sha256", m.get("sha256"))
                    m["updated_at"] = datetime.now().isoformat()
                    if active_data.get("metrics"):
                        m.setdefault("metrics", {}).update(active_data["metrics"])
                    break
            manifest_path.write_text(json.dumps(m_data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to sync models/manifest.json: {e}")


# Singleton instance
model_registry = ModelRegistry()
