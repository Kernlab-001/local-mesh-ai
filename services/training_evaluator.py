"""
Training and Adapter Evaluation Service for Local Mesh AI.
Evaluates fine-tuned model candidates against golden test benchmarks,
executing actual validation for JSON syntax, tool prediction, component identification,
numeric parameter accuracy within bounds, and invariant preservation.
Persists evaluation reports and gates model promotion.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

import torch
from core.path_security import get_data_dir
from core.command_schemas import ComponentEditPlan
from services.lora_training_manager import AuxiliaryTechnicalArtistClassifier

GOLDEN_EVALUATION_TASKS = [
    {
        "prompt": "Make the backrest 20% taller.",
        "expected_component": "backrest",
        "expected_tool": "scale_component",
        "expected_params": {"axis": "z", "min_factor": 1.15, "max_factor": 1.25},
        "is_held_out": False,
    },
    {
        "prompt": "Increase the thickness of the four legs.",
        "expected_component": "legs",
        "expected_tool": "scale_component",
        "expected_params": {"axis": "all", "min_factor": 1.1, "max_factor": 1.5},
        "is_held_out": False,
    },
    {
        "prompt": "Make the handle slightly longer.",
        "expected_component": "handle",
        "expected_tool": "scale_component",
        "expected_params": {"axis": "y", "min_factor": 1.05, "max_factor": 1.3},
        "is_held_out": False,
    },
    {
        "prompt": "Separate the wheels into individual objects.",
        "expected_component": "wheels",
        "expected_tool": "separate_components",
        "expected_params": {"axis": "all"},
        "is_held_out": False,
    },
    {
        "prompt": "Make the backrest taller.",
        "expected_component": "backrest",
        "expected_tool": "scale_component",
        "expected_params": {"axis": "z", "min_factor": 1.10, "max_factor": 1.30},
        "is_held_out": True,
    },
    {
        "prompt": "Separate the wheels.",
        "expected_component": "wheels",
        "expected_tool": "separate_components",
        "expected_params": {"axis": "all"},
        "is_held_out": True,
    },
]


class EvaluationTaskResult(BaseModel):
    prompt: str
    passed: bool
    json_valid: bool
    tool_match: bool = True
    component_match: bool
    bounds_safe: bool
    params_accurate: bool = True
    regression_detected: bool = False
    details: str = ""
    prediction: Optional[Dict[str, Any]] = None


class EvaluationReport(BaseModel):
    model_name: str
    adapter_hash: Optional[str] = None
    evaluation_timestamp: float = Field(default_factory=time.time)
    total_tasks: int
    passed_tasks: int
    score_percent: float
    baseline_score_percent: float = 0.0
    improvement_over_baseline: float = 0.0
    syntax_accuracy: float
    tool_accuracy: float = 100.0
    component_accuracy: float
    safety_accuracy: float
    passed_evaluation: bool
    generalization_provenance: str = "in_distribution_demonstration"
    dataset_type: str = "demonstration_preset"
    task_results: List[EvaluationTaskResult] = Field(default_factory=list)
    report_file_path: Optional[str] = None


class TrainingEvaluator:
    def __init__(self):
        self.eval_dir = get_data_dir() / "learning" / "evaluation"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_candidate(self, candidate_name: str, adapter_path: Optional[str] = None) -> EvaluationReport:
        """
        Executes genuine inference evaluation of candidate adapter against golden benchmarks.
        Requires physical checkpoint presence on disk.
        Fails corrupted or dummy checkpoints with 0 score.
        """
        cand_dir = Path(adapter_path) if adapter_path else (get_data_dir() / "models" / "candidates" / candidate_name)
        if not cand_dir.exists():
            raise FileNotFoundError(f"Candidate adapter directory '{cand_dir}' does not exist.")

        # 1. Require real checkpoint file
        ckpt_files = list(cand_dir.glob("*.bin")) + list(cand_dir.glob("*.pt")) + list(cand_dir.glob("*.safetensors")) + list(cand_dir.glob("*.gguf"))
        if not ckpt_files:
            raise ValueError(f"Candidate '{candidate_name}' contains no genuine checkpoint files (.bin, .pt, .safetensors, .gguf).")

        primary_ckpt = ckpt_files[0]
        if primary_ckpt.stat().st_size == 0:
            raise ValueError(f"Candidate checkpoint '{primary_ckpt.name}' is empty (0 bytes).")

        # Compute checkpoint SHA-256
        h = hashlib.sha256()
        with open(primary_ckpt, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        adapter_hash = h.hexdigest()

        # 2. Attempt loading checkpoint weights into neural module
        model = AuxiliaryTechnicalArtistClassifier()
        model_load_ok = False
        load_error = ""

        try:
            state_dict = torch.load(str(primary_ckpt), map_location="cpu", weights_only=True)
            if not isinstance(state_dict, dict) or len(state_dict) == 0:
                raise ValueError("Checkpoint does not contain a valid PyTorch state_dict.")
            # Verify required architecture keys
            required_keys = {"lora_A", "lora_B", "tool_head.weight", "comp_head.weight"}
            missing_reqs = required_keys.difference(state_dict.keys())
            if missing_reqs:
                raise ValueError(f"Checkpoint architecture mismatch: missing required keys {missing_reqs}")
            model.load_state_dict(state_dict, strict=True)
            model_load_ok = True
        except Exception as e:
            load_error = f"Checkpoint deserialization failed: {str(e)}"

        results: List[EvaluationTaskResult] = []

        # 3. If checkpoint failed to load or is corrupt, fail all tasks
        if not model_load_ok:
            for task in GOLDEN_EVALUATION_TASKS:
                results.append(
                    EvaluationTaskResult(
                        prompt=task["prompt"],
                        passed=False,
                        json_valid=False,
                        tool_match=False,
                        component_match=False,
                        bounds_safe=False,
                        params_accurate=False,
                        regression_detected=True,
                        details=load_error,
                        prediction=None
                    )
                )
        else:
            # Execute actual model inference for each golden task
            for task in GOLDEN_EVALUATION_TASKS:
                prompt = task["prompt"]
                exp_comp = task["expected_component"]
                exp_tool = task["expected_tool"]
                exp_params = task.get("expected_params", {})

                json_valid = False
                tool_match = False
                component_match = False
                params_accurate = False
                bounds_safe = False
                prediction_data = None
                details_list = []

                try:
                    prediction_data = model.generate_plan(prompt, device=torch.device("cpu"))
                    plan_obj = ComponentEditPlan(**prediction_data)
                    json_valid = True

                    # 1. Strict tool verification (zero permissive fallback)
                    pred_tool = plan_obj.parameters.get("tool")
                    if pred_tool == exp_tool:
                        tool_match = True
                    else:
                        details_list.append(f"Tool mismatch: expected '{exp_tool}', got '{pred_tool}'.")

                    # 2. Strict component verification
                    labels_str = " ".join(plan_obj.target_semantic_labels).lower()
                    ids_str = " ".join(plan_obj.target_component_ids).lower()
                    if exp_comp in labels_str or exp_comp in ids_str:
                        component_match = True
                    else:
                        details_list.append(f"Component mismatch: expected '{exp_comp}', got '{labels_str}'.")

                    # 3. Exact axis verification
                    exp_axis = exp_params.get("axis")
                    axis_match = True
                    if exp_axis is not None:
                        if plan_obj.axis != exp_axis:
                            axis_match = False
                            details_list.append(f"Axis mismatch: expected '{exp_axis}', got '{plan_obj.axis}'.")

                    # 4. Complete scale vector validation
                    min_factor = exp_params.get("min_factor")
                    max_factor = exp_params.get("max_factor")
                    scale_params = plan_obj.parameters.get("scale", [1.0, 1.0, 1.0])
                    scale_factor = plan_obj.scale_factor

                    if min_factor is not None and max_factor is not None:
                        # Check target axis scale and ensure non-target axes did not diverge
                        target_axis = exp_axis or "z"
                        if target_axis == "z" and len(scale_params) >= 3:
                            val_target = scale_params[2]
                            non_target = [scale_params[0], scale_params[1]]
                            matches_target = (min_factor <= val_target <= max_factor)
                            non_target_safe = all(0.90 <= s <= 1.10 for s in non_target)
                            if matches_target and non_target_safe:
                                params_accurate = True
                            else:
                                details_list.append(f"Vector mismatch: expected z in [{min_factor}, {max_factor}] with neutral x/y, got scale={scale_params}.")
                        elif target_axis == "y" and len(scale_params) >= 2:
                            val_target = scale_params[1]
                            non_target = [scale_params[0], scale_params[2] if len(scale_params) > 2 else 1.0]
                            matches_target = (min_factor <= val_target <= max_factor)
                            non_target_safe = all(0.90 <= s <= 1.10 for s in non_target)
                            if matches_target and non_target_safe:
                                params_accurate = True
                            else:
                                details_list.append(f"Vector mismatch: expected y in [{min_factor}, {max_factor}] with neutral x/z, got scale={scale_params}.")
                        elif target_axis == "all":
                            matches_target = (min_factor <= scale_factor <= max_factor) or all(min_factor <= s <= max_factor for s in scale_params[:2])
                            if matches_target:
                                params_accurate = True
                            else:
                                details_list.append(f"Vector mismatch: expected all in [{min_factor}, {max_factor}], got scale={scale_params}.")
                        else:
                            params_accurate = (min_factor <= scale_factor <= max_factor)
                    else:
                        params_accurate = True

                    # 5. Global bounds safety
                    if all(0.2 <= s <= 3.0 for s in scale_params) and (0.2 <= scale_factor <= 3.0):
                        bounds_safe = True
                    else:
                        details_list.append(f"Parameter bounds violation: scale {scale_params} outside [0.2, 3.0].")

                except Exception as eval_err:
                    details_list.append(f"Inference/schema validation failed: {eval_err}")

                task_passed = json_valid and tool_match and component_match and axis_match and params_accurate and bounds_safe
                results.append(
                    EvaluationTaskResult(
                        prompt=prompt,
                        passed=task_passed,
                        json_valid=json_valid,
                        tool_match=tool_match,
                        component_match=component_match,
                        bounds_safe=bounds_safe,
                        params_accurate=params_accurate,
                        regression_detected=not task_passed,
                        details="; ".join(details_list) if not task_passed else "All schema, tool, component, and parameter checks passed.",
                        prediction=prediction_data
                    )
                )

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        syntax = (sum(1 for r in results if r.json_valid) / total) * 100.0 if total else 0.0
        tools = (sum(1 for r in results if r.tool_match) / total) * 100.0 if total else 0.0
        comp = (sum(1 for r in results if r.component_match) / total) * 100.0 if total else 0.0
        safety = (sum(1 for r in results if r.bounds_safe) / total) * 100.0 if total else 0.0
        score = (passed / total) * 100.0 if total else 0.0
        passed_eval = (score >= 80.0) and model_load_ok

        report_file = self.eval_dir / f"{candidate_name}_evaluation.json"
        # Baseline comparison: Evaluate untrained prior
        baseline_score = 0.0
        try:
            baseline_model = AuxiliaryTechnicalArtistClassifier()
            base_passed = 0
            for bt in GOLDEN_EVALUATION_TASKS:
                b_pred = baseline_model.generate_plan(bt["prompt"], device=torch.device("cpu"))
                if b_pred.get("parameters", {}).get("tool") == bt["expected_tool"] and bt["expected_component"] in " ".join(b_pred.get("target_semantic_labels", [])):
                    base_passed += 1
            baseline_score = round((base_passed / len(GOLDEN_EVALUATION_TASKS)) * 100.0, 1)
        except Exception:
            baseline_score = 0.0

        improvement = round(max(0.0, score - baseline_score), 1)

        report = EvaluationReport(
            model_name=candidate_name,
            adapter_hash=adapter_hash,
            total_tasks=total,
            passed_tasks=passed,
            score_percent=score,
            baseline_score_percent=baseline_score,
            improvement_over_baseline=improvement,
            syntax_accuracy=syntax,
            tool_accuracy=tools,
            component_accuracy=comp,
            safety_accuracy=safety,
            passed_evaluation=passed_eval,
            generalization_provenance="in_distribution_demonstration",
            dataset_type="demonstration_preset",
            task_results=results,
            report_file_path=str(report_file),
        )

        # Persist report to disk
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

        # Update candidate metadata
        meta_file = cand_dir / "adapter_metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["evaluated"] = True
                meta["benchmark_score"] = score
                meta["passed_evaluation"] = passed_eval
                meta["evaluation_timestamp"] = time.time()
                meta["evaluation_report_file"] = str(report_file)
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

        return report

    def get_evaluation_report(self, candidate_name: str) -> Optional[EvaluationReport]:
        report_file = self.eval_dir / f"{candidate_name}_evaluation.json"
        if report_file.exists():
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    return EvaluationReport(**json.load(f))
            except Exception:
                pass
        return None


training_evaluator = TrainingEvaluator()
