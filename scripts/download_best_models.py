#!/usr/bin/env python3
"""
Download BEST Satellite-Trained Models for Maximum Accuracy
===========================================================

Downloads the most accurate models specifically trained on satellite imagery:
1. Large YOLOv8 model for building/roof detection (best accuracy)
2. Best available roof damage detection model
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def install_and_import(package, import_name=None):
    """Install package if needed and import it."""
    if import_name is None:
        import_name = package
    try:
        return __import__(import_name)
    except ImportError:
        print(f"[INSTALL] {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        return __import__(import_name)


def download_model(model_id, filename, save_path, models_dir):
    """Download a specific model file from Hugging Face."""
    from huggingface_hub import hf_hub_download
    
    print(f"\n{'='*70}")
    print(f"[DOWNLOADING] {model_id}")
    print(f"[FILE] {filename}")
    print(f"{'='*70}")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        
        # Copy to simple location
        final_path = models_dir / save_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        
        if Path(downloaded_path).exists():
            shutil.copy(downloaded_path, final_path)
            size_mb = final_path.stat().st_size / (1024 * 1024)
            print(f"[SUCCESS] Saved to: {final_path}")
            print(f"[SIZE] {size_mb:.1f} MB")
            return final_path
            
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def main():
    print("=" * 70)
    print("   AI ROOF DAMAGE DETECTION - BEST MODEL DOWNLOADER")
    print("   Maximum Accuracy Models for Satellite Imagery")
    print("=" * 70)
    
    # Setup directories
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    models_dir = project_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Install dependencies
    print("\n[SETUP] Installing dependencies...")
    install_and_import("huggingface_hub", "huggingface_hub")
    
    downloaded = {}
    
    # =========================================================================
    # MODEL 1: BEST ROOF/BUILDING SEGMENTATION (Large Model for Accuracy)
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: ROOF/BUILDING SEGMENTATION (BEST ACCURACY)")
    print("Source: keremberke/yolov8m-building-segmentation")
    print("Note: Medium model - best balance of accuracy and speed")
    print("      Trained specifically on satellite imagery")
    print("=" * 70)
    
    # Try large model first (best accuracy)
    roof_model_large = download_model(
        model_id="keremberke/yolov8l-building-segmentation",  # Large model
        filename="best.pt",
        save_path="roof_detector_large.pt",
        models_dir=models_dir
    )
    
    # Also download medium model (faster, still accurate)
    roof_model = download_model(
        model_id="keremberke/yolov8m-building-segmentation",
        filename="best.pt",
        save_path="roof_detector.pt",
        models_dir=models_dir
    )
    
    if roof_model:
        downloaded["roof_detector"] = roof_model
    if roof_model_large:
        downloaded["roof_detector_large"] = roof_model_large
        print("\n[INFO] Large model available! Use roof_detector_large.pt for maximum accuracy")
    
    # =========================================================================
    # MODEL 2: ROOF DAMAGE DETECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: ROOF DAMAGE/DEFECT DETECTION")
    print("Source: JGuevara-12/yolo-roof-damage")
    print("=" * 70)
    
    damage_model = download_model(
        model_id="JGuevara-12/yolo-roof-damage",
        filename="yolov8/weights/best.pt",
        save_path="damage_detector.pt",
        models_dir=models_dir
    )
    if damage_model:
        downloaded["damage_detector"] = damage_model
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    
    if downloaded:
        print("\n[SUCCESS] Downloaded models:")
        for name, path in downloaded.items():
            print(f"  ✓ {name}: {path}")
        
        print("\n[RECOMMENDATION]")
        if "roof_detector_large" in downloaded:
            print("  Use 'roof_detector_large.pt' for MAXIMUM accuracy")
            print("  Use 'roof_detector.pt' for faster processing")
        else:
            print("  Use 'roof_detector.pt' (medium model - good balance)")
        
        print("\n[CONFIGURATION]")
        print("  Update config/settings.py or .env file:")
        if "roof_detector_large" in downloaded:
            print("    ROOF_MODEL_PATH=./models/roof_detector_large.pt  # For best accuracy")
        print("    ROOF_MODEL_PATH=./models/roof_detector.pt  # For speed")
        print("    DAMAGE_MODEL_PATH=./models/damage_detector.pt")
    else:
        print("\n[WARNING] No models downloaded. Check errors above.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

