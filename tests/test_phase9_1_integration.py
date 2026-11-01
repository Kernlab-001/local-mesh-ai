"""
Comprehensive Integration Tests for Phase 9.1:
Reliability Closure, Real Training, Candidate Lifecycle, GPU Safety & API Integrity.
Verifies all 15 core architectural and integration requirements.
"""

import os
import json
import time
import uuid
import hashlib
import threading
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from api.app import app
from core.path_security import get_staging_dir, get_data_dir
from core.gpu_broker import gpu_broker
from core.model_manager import model_manager
from core.recovery_actions import ActionType, RecoveryActionRequest
from services.lora_training_manager import (
    lora_training_manager,
    TechnicalArtistLoRAModule,
    load_and_parse_dataset,
)
from services.training_evaluator import training_evaluator, EvaluationReport
from services.model_version_manager import model_version_manager, LifecycleState
from services.model_rollback_manager import model_rollback_manager
from services.self_healing_manager import self_healing_manager
from services.editing_session_manager import editing_session_manager
from core.component_decomposer import BoundingBox3D, ComponentHierarchy, ComponentNode

client = TestClient(app)


# --- 1. Dataset Loading and Parsing ---
def test_training_worker_loads_and_consumes_real_dataset():
    """1. The training worker loads and consumes real dataset examples."""
    staging = get_staging_dir()
    ds_file = staging / f"test_instructions_{uuid.uuid4().hex[:6]}.jsonl"

    instruction_samples = [
        {"prompt": "Make the backrest 20% taller.", "expected_json_command": {"tool": "scale_component", "component": "backrest"}},
        {"prompt": "Increase the thickness of the four legs.", "expected_json_command": {"tool": "scale_component", "component": "legs"}},
        {"prompt": "Separate the wheels into individual objects.", "expected_json_command": {"tool": "separate_components", "component": "wheels"}},
    ]
    with open(ds_file, "w", encoding="utf-8") as f:
        for s in instruction_samples:
            f.write(json.dumps(s) + "\n")

    # Verify parser
    parsed = load_and_parse_dataset(str(ds_file))
    assert len(parsed) == 3
    assert parsed[0]["prompt"] == "Make the backrest 20% taller."

    # Execute training with this real dataset
    job = lora_training_manager.start_training(
        target_model="local-technical-artist-lora",
        adapter_name=f"test_lora_ds_{uuid.uuid4().hex[:6]}",
        dataset_path=str(ds_file),
        epochs=2,
        force_cpu=True
    )

    timeout = 15.0
    start = time.time()
    while job.status == "running" and time.time() - start < timeout:
        time.sleep(0.1)

    assert job.status == "completed", f"Job failed: {job.error} ({job.message})"
    assert len(job.loss_history) == 2
    assert job.current_loss is not None
    assert job.candidate_adapter_path is not None


# --- 2. Trainable Parameters Change ---
def test_real_trainable_model_parameters_change():
    """2. Real trainable model parameters change."""
    module = TechnicalArtistLoRAModule(r=8)
    init_A = module.lora_A.detach().clone()
    init_B = module.lora_B.detach().clone()

    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    input_ids = module.encode_text("Make the backrest taller.")
    target_tool = torch.tensor([0], dtype=torch.long)

    for _ in range(5):
        optimizer.zero_grad()
        tool_logits, _, _ = module(input_ids)
        loss = criterion(tool_logits, target_tool)
        loss.backward()
        optimizer.step()

    delta_A = torch.norm(module.lora_A.detach() - init_A).item()
    assert delta_A > 0.0, "LoRA matrix A parameters must update during optimization"


# --- 3. Resulting Adapter Loads with Base Model ---
def test_resulting_adapter_loads_with_base_model():
    """3. The resulting adapter can be loaded with its intended base model."""
    adapter_name = f"test_loadable_{uuid.uuid4().hex[:6]}"
    job = lora_training_manager.start_training(
        target_model="local-technical-artist-lora",
        adapter_name=adapter_name,
        epochs=1,
        force_cpu=True
    )
    start = time.time()
    while job.status == "running" and time.time() - start < 10.0:
        time.sleep(0.1)

    assert job.status == "completed"
    ckpt_file = Path(job.candidate_adapter_path) / "adapter_model.bin"
    assert ckpt_file.exists()

    # Load into fresh module
    fresh_module = TechnicalArtistLoRAModule()
    state = torch.load(str(ckpt_file), map_location="cpu", weights_only=True)
    fresh_module.load_state_dict(state, strict=False)

    plan = fresh_module.generate_plan("Make the backrest taller.")
    assert "target_semantic_labels" in plan
    assert "backrest" in plan["target_semantic_labels"]


# --- 4. Evaluation Runs Real Candidate Inference ---
def test_evaluation_runs_real_candidate_inference():
    """4. Evaluation runs real candidate inference."""
    adapter_name = f"eval_real_{uuid.uuid4().hex[:6]}"
    job = lora_training_manager.start_training(
        target_model="local-technical-artist-lora",
        adapter_name=adapter_name,
        epochs=1,
        force_cpu=True
    )
    start = time.time()
    while job.status == "running" and time.time() - start < 10.0:
        time.sleep(0.1)

    assert job.status == "completed"

    report = training_evaluator.evaluate_candidate(adapter_name)
    assert report.total_tasks > 0
    assert report.score_percent >= 80.0
    assert report.passed_evaluation is True

    # Confirm candidate prediction came from model inference
    first_res = report.task_results[0]
    assert first_res.prediction is not None
    assert "plan_id" in first_res.prediction
    assert first_res.component_match is True


# --- 5. Dummy or Corrupt Checkpoints Fail Evaluation ---
def test_dummy_or_corrupt_checkpoint_fails_evaluation():
    """5. Dummy, corrupt, or mismatched checkpoints cannot pass evaluation."""
    cand_dir = get_data_dir() / "models" / "candidates" / f"corrupt_{uuid.uuid4().hex[:6]}"
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "adapter_model.bin").write_bytes(b"CORRUPT_NOT_A_PYTORCH_STATE_DICT")

    meta = {"adapter_name": cand_dir.name, "created_at": time.time()}
    (cand_dir / "adapter_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    report = training_evaluator.evaluate_candidate(cand_dir.name)
    assert report.passed_evaluation is False
    assert report.score_percent == 0.0
    assert report.passed_tasks == 0


# --- 6. Fake Metadata Cannot Bypass Promotion Gates ---
def test_fake_metadata_cannot_bypass_promotion_gates():
    """6. Fake metadata cannot bypass promotion gates."""
    cand_id = f"fake_meta_cand_{uuid.uuid4().hex[:6]}"
    cand_dir = get_data_dir() / "models" / "candidates" / cand_id
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "adapter_model.bin").write_bytes(b"123")

    spoofed_meta = {
        "benchmark_score": 100.0,
        "passed_evaluation": True,
        "evaluated": True
    }
    model_version_manager.register_candidate(
        model_id=cand_id,
        model_name=cand_id,
        adapter_path=str(cand_dir),
        metadata=spoofed_meta
    )

    with pytest.raises(ValueError) as exc:
        model_version_manager.promote_candidate_to_active(cand_id, approved=True)

    assert "Metadata score bypasses are strictly prohibited" in str(exc.value)


# --- 7. Candidate Registration Happens Automatically After Training ---
def test_candidate_registration_happens_automatically_after_training():
    """7. Candidate registration happens automatically after successful training."""
    adapter_name = f"auto_reg_test_{uuid.uuid4().hex[:6]}"
    job = lora_training_manager.start_training(
        target_model="local-technical-artist-lora",
        adapter_name=adapter_name,
        epochs=1,
        force_cpu=True
    )
    start = time.time()
    while job.status == "running" and time.time() - start < 10.0:
        time.sleep(0.1)

    assert job.status == "completed"

    with model_version_manager._lock:
        candidates = model_version_manager._state["candidates"]
        assert adapter_name in candidates
        cand_rec = candidates[adapter_name]
        assert cand_rec["lifecycle_state"] == LifecycleState.CANDIDATE.value


# --- 8 & 9. Promotion Changes Inference Behavior and Rollback Restores It ---
def test_promotion_changes_inference_and_rollback_restores_it():
    """8. Promotion changes actual inference behavior. 9. Rollback restores previous inference behavior."""
    # Ensure known baseline state
    model_manager.unload_adapter()
    with model_version_manager._lock:
        model_version_manager._state["active_adapter_id"] = None
        base_records = [
            h for h in model_version_manager._state.get("version_history", [])
            if h.get("is_base_model") or h.get("model_id") == "qwen-7b-base"
        ]
        if not base_records:
            base_records = [{
                "model_id": "qwen-7b-base",
                "model_name": "Qwen 2.5 Coder 7B Base",
                "lifecycle_state": "active",
                "is_base_model": True,
                "benchmark_score": 100.0,
                "created_at": time.time()
            }]
        model_version_manager._state["version_history"] = base_records
        model_version_manager._save_state_atomic(model_version_manager._state)

    # Step A: Baseline inference before promotion
    base_plan = model_manager.predict_technical_artist_plan("Make the backrest taller.")
    assert base_plan.get("active_adapter_id") is None
    assert base_plan.get("model_type") == "base_model"

    # Step B: Train, evaluate, and promote candidate
    adapter_name = f"promo_cand_{uuid.uuid4().hex[:6]}"
    job = lora_training_manager.start_training(
        target_model="local-technical-artist-lora",
        adapter_name=adapter_name,
        epochs=1,
        force_cpu=True
    )
    start = time.time()
    while job.status == "running" and time.time() - start < 10.0:
        time.sleep(0.1)
    assert job.status == "completed"

    eval_report = training_evaluator.evaluate_candidate(adapter_name)
    assert eval_report.passed_evaluation is True

    model_version_manager.promote_candidate_to_active(adapter_name, approved=True)

    # Step C: Inference behavior reflects promoted adapter
    active_plan = model_manager.predict_technical_artist_plan("Make the backrest taller.")
    assert active_plan.get("active_adapter_id") == adapter_name

    # Step D: Rollback restores previous inference behavior
    res = model_rollback_manager.rollback_to_previous_active(reason="Testing inference restoration")
    assert res.success is True
    restored_plan = model_manager.predict_technical_artist_plan("Make the backrest taller.")
    assert restored_plan.get("active_adapter_id") is None
    assert restored_plan.get("model_type") == "base_model"


# --- 10. Blender Request Payloads Match Backend API Contracts ---
def test_blender_request_payloads_match_api_contracts():
    """10. Blender request payloads match the backend API contracts."""
    # Preflight contract
    resp_pref = client.post(
        "/api/v1/learning/preflight",
        json={"model_name": "local-technical-artist-lora", "force_cpu": False}
    )
    assert resp_pref.status_code == 200
    p_data = resp_pref.json()
    assert "hardware_feasibility" in p_data
    assert "free_vram_mb" in p_data

    # Models list contract
    resp_models = client.get("/api/v1/learning/models")
    assert resp_models.status_code == 200
    m_data = resp_models.json()
    assert isinstance(m_data, list)
    if len(m_data) > 0:
        assert "model_id" in m_data[0]
        assert "model_type" in m_data[0]

    # Evaluate contract
    cand_name = f"contract_eval_{uuid.uuid4().hex[:6]}"
    j = lora_training_manager.start_training(target_model="local-technical-artist-lora", adapter_name=cand_name, epochs=1, force_cpu=True)
    st = time.time()
    while j.status == "running" and time.time() - st < 10.0:
        time.sleep(0.1)

    resp_eval = client.post("/api/v1/learning/evaluate", json={"candidate_name": cand_name})
    assert resp_eval.status_code == 200
    e_data = resp_eval.json()
    assert "score_percent" in e_data
    assert "passed_evaluation" in e_data

    # Recovery contract
    resp_rec = client.post(
        "/api/v1/health/recover",
        json={"action_type": "clean_staging_dir", "approved": False, "dry_run": True}
    )
    assert resp_rec.status_code == 200
    r_data = resp_rec.json()
    assert r_data["status"] == "dry_run"
    assert "affected_files" in r_data


# --- 11. Recovery Preview Sends dry_run=True and Preserves Files ---
def test_recovery_preview_sends_dry_run_and_preserves_files():
    """11. Recovery preview sends dry_run=true and does not delete files."""
    staging = get_staging_dir()
    test_file = staging / f"preserve_preview_{uuid.uuid4().hex[:6]}.glb"
    test_file.write_bytes(b"TEST_PREVIEW_BYTES")

    old_time = time.time() - 4000
    os.utime(test_file, (old_time, old_time))

    res = self_healing_manager.execute_action(
        RecoveryActionRequest(action_type=ActionType.CLEAN_STAGING_DIR, approved=False, dry_run=True)
    )

    assert res.status == "dry_run"
    assert test_file.exists(), "Dry-run preview must NEVER delete files!"
    test_file.unlink(missing_ok=True)


# --- 12. Recovery Approval Applies Only to Reviewed Files ---
def test_recovery_approval_applies_only_to_reviewed_files():
    """12. Recovery approval applies only to the reviewed files."""
    staging = get_staging_dir()
    file_a = staging / f"target_file_{uuid.uuid4().hex[:6]}.glb"
    file_b = staging / f"unreviewed_file_{uuid.uuid4().hex[:6]}.glb"

    file_a.write_bytes(b"AAA")
    file_b.write_bytes(b"BBB")

    old_time = time.time() - 4000
    os.utime(file_a, (old_time, old_time))
    os.utime(file_b, (old_time, old_time))

    res = self_healing_manager.execute_action(
        RecoveryActionRequest(
            action_type=ActionType.CLEAN_STAGING_DIR,
            approved=True,
            dry_run=False,
            parameters={"approved_files": [str(file_a.resolve())]}
        )
    )

    assert not file_a.exists(), "Reviewed and approved file must be cleaned"
    assert file_b.exists(), "Unreviewed file must be preserved!"
    file_b.unlink(missing_ok=True)


# --- 13. GPU Lock Behavior Tested Through Actual Training Worker ---
def test_gpu_lock_behavior_tested_through_actual_training_worker():
    """13. GPU lock behavior is tested through the actual training worker."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable on host for hardware GPU lock test.")

    lock_file = Path(gpu_broker.lock_file)
    adapter_name = f"gpu_lock_worker_{uuid.uuid4().hex[:6]}"

    job = lora_training_manager.start_training(
        target_model="local-technical-artist-lora",
        adapter_name=adapter_name,
        epochs=3,
        force_cpu=False
    )

    st = time.time()
    locked_observed = False
    while time.time() - st < 10.0:
        if gpu_broker.is_locked() and lock_file.exists():
            locked_observed = True
            break
        if job.status in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert locked_observed is True, "Training worker must acquire and hold the GPU disk lock"

    while job.status == "running" and time.time() - st < 15.0:
        time.sleep(0.1)

    assert job.status == "completed"
    assert not lock_file.exists(), "GPU lock must be released when training worker finishes"

    lease2 = gpu_broker.acquire("subsequent_client", timeout=5.0)
    assert lease2 is not None
    gpu_broker.release(client_name="subsequent_client", lease_id=lease2)


# --- 14. Living Process Cannot Lose Its GPU Lease Merely on Timeout ---
def test_living_process_protects_lease_from_timeout_purge():
    """14. A live process cannot lose its GPU lease merely because an arbitrary timeout elapsed."""
    lock_file = Path(gpu_broker.lock_file)
    lease = gpu_broker.acquire("active_live_process", timeout=5.0)
    assert lease is not None
    assert lock_file.exists()

    try:
        with open(lock_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["timestamp"] = time.time() - 1200.0
        with open(lock_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        assert gpu_broker.is_locked() is True
        assert lock_file.exists()

    finally:
        gpu_broker.release(client_name="active_live_process", lease_id=lease)


# --- 15. Session Concurrency Tests Exercise Actual Mutation Endpoints ---
def test_session_concurrency_exercises_actual_mutation_endpoints():
    """15. Session concurrency tests exercise actual mutation endpoints rather than manually acquiring a lock."""
    session_id = f"concurr_endpoint_{uuid.uuid4().hex[:6]}"
    staging = get_staging_dir()
    mesh_path = staging / f"{session_id}_test_mesh.obj"
    mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    session = editing_session_manager.create_session(
        asset_name="concurr_endpoint_asset",
        source_mesh_path=str(mesh_path)
    )
    s_id = session.session_id

    errors = []
    statuses = []

    def call_rollback():
        try:
            r = client.post("/api/v1/artist/rollback", json={"session_id": s_id})
            statuses.append(r.status_code)
        except Exception as e:
            errors.append(str(e))

    def call_session():
        try:
            r = client.get(f"/api/v1/artist/session/{s_id}")
            statuses.append(r.status_code)
        except Exception as e:
            errors.append(str(e))

    threads = [
        threading.Thread(target=call_rollback),
        threading.Thread(target=call_session),
        threading.Thread(target=call_rollback),
        threading.Thread(target=call_session),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent endpoint calls caused exceptions: {errors}"
    assert all(code in (200, 400) for code in statuses), f"Endpoints returned invalid statuses: {statuses}"
