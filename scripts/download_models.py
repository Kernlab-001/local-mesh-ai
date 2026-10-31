"""
Standalone CLI Model Provisioning Tool for Local Mesh AI.
Downloads and verifies required model weights (TripoSR, DINO ViT, and Qwen 2.5 Coder)
with resume capability, disk space checks, and offline verification.

Usage:
    python scripts/download_models.py               # Downloads TripoSR + DINO weights
    python scripts/download_models.py --check       # Verifies local offline weights
    python scripts/download_models.py --qwen        # Downloads Qwen 2.5 Coder 7B GGUF
    python scripts/download_models.py --all         # Downloads all models
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.download_manager import DownloadManager, get_configured_models_dir


def main():
    parser = argparse.ArgumentParser(description="Local Mesh AI - Model Provisioning Tool")
    parser.add_argument("--triposr", action="store_true", help="Download TripoSR and DINO model weights")
    parser.add_argument("--qwen", action="store_true", help="Download Qwen 2.5 Coder 7B Instruct GGUF model")
    parser.add_argument("--all", action="store_true", help="Download all required models (TripoSR + DINO + Qwen)")
    parser.add_argument("--check", action="store_true", help="Check offline readiness of models without downloading")
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom directory for persistent model storage")
    parser.add_argument("--force", action="store_true", help="Force re-download even if already verified")

    args = parser.parse_args()

    custom_dir = Path(args.cache_dir) if args.cache_dir else None
    mgr = DownloadManager(models_dir=custom_dir)

    print("=" * 68)
    print("  LOCAL MESH AI — MODEL WEIGHT PROVISIONING")
    print(f"  Storage Location: {mgr.models_dir}")
    print(f"  Free Space:       {mgr.get_free_disk_space_gb()} GB")
    print("=" * 68)

    # If --check is specified, perform status report only
    if args.check:
        print("\n[Status: TripoSR & DINO Tokenizer]")
        t_stat = mgr.verify_triposr_status()
        print(f"  State: {t_stat['state'].upper()}")
        print(f"  Offline Ready: {t_stat['ready_for_offline_inference']}")
        for fkey, finfo in t_stat["files"].items():
            status_str = f"INSTALLED ({finfo['size_bytes'] / (1024*1024):.2f} MB)" if finfo["installed"] else "MISSING"
            print(f"  - {fkey}: {status_str}")

        print("\n[Status: Qwen 2.5 Coder LLM]")
        q_stat = mgr.verify_qwen_status()
        print(f"  State: {q_stat['state'].upper()}")
        if q_stat["model_path"]:
            print(f"  Location: {q_stat['model_path']} ({q_stat['size_bytes'] / (1024**3):.2f} GB)")
            print(f"  Source:   {q_stat['source']}")
        else:
            print("  Location: None")

        print("=" * 68)
        all_ready = t_stat["ready_for_offline_inference"]
        sys.exit(0 if all_ready else 1)

    # Determine what to download
    dl_triposr = args.triposr or args.all or (not args.triposr and not args.qwen and not args.all)
    dl_qwen = args.qwen or args.all

    success = True
    if dl_triposr:
        try:
            mgr.download_triposr_weights(force=args.force)
        except Exception as e:
            print(f"\n[FATAL] TripoSR download failed: {e}")
            success = False

    if dl_qwen and success:
        try:
            mgr.download_qwen_weights(force=args.force)
        except Exception as e:
            print(f"\n[FATAL] Qwen download failed: {e}")
            success = False

    if success:
        print("\n[SUCCESS] Model provisioning completed successfully!")
        sys.exit(0)
    else:
        print("\n[FAILED] Model provisioning encountered errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
