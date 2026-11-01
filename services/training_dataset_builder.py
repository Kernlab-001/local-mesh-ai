"""
Training Dataset Builder for Local Mesh AI.
Transforms approved geometric editing samples into structured instruction-tuning datasets
(Alpaca and ChatML format) for supervised fine-tuning and LoRA adaptation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.path_security import get_data_dir
from core.learning_schemas import LearningSample, DatasetManifest
from services.learning_data_manager import learning_data_manager


class TrainingDatasetBuilder:
    def __init__(self):
        self.output_dir = get_data_dir() / "learning" / "prepared_training"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_sample_as_instruction(self, sample: LearningSample) -> Dict[str, Any]:
        """Formats a single LearningSample into an Alpaca-style instruction tuning pair."""
        instruction = (
            "You are Local Mesh AI 3D Technical Artist. "
            "Analyze the asset components and user modification request, "
            "and output a strictly valid JSON editing plan with bounded geometry operations."
        )

        input_context = {
            "user_request": sample.prompt,
            "target_components": sample.component_ids,
            "metrics_before": sample.metrics_before,
        }

        output_plan = {
            "plan_id": f"plan_{sample.sample_id}",
            "intent": sample.prompt,
            "operations": sample.operations_applied,
        }

        return {
            "instruction": instruction,
            "input": json.dumps(input_context, indent=2),
            "output": json.dumps(output_plan, indent=2),
            "sample_id": sample.sample_id,
            "asset_family": sample.asset_family,
        }

    def build_instruction_dataset(
        self,
        samples: Optional[List[LearningSample]] = None,
        output_filename: str = "alpaca_editing_instructions.jsonl"
    ) -> Path:
        """Builds and writes a JSONL instruction dataset."""
        if samples is None:
            from core.learning_schemas import SampleReviewStatus
            samples = learning_data_manager.get_samples(status=SampleReviewStatus.APPROVED)

        out_path = self.output_dir / output_filename
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                formatted = self.format_sample_as_instruction(s)
                f.write(json.dumps(formatted) + "\n")

        return out_path


training_dataset_builder = TrainingDatasetBuilder()
