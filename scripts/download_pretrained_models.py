#!/usr/bin/env python3
"""
Download Pre-trained Models for Roof/Building Detection and Damage Analysis

Found Models:
=============

1. BUILDING/ROOF SEGMENTATION (Hugging Face):
   - keremberke/yolov8n-building-segmentation (nano - fastest)
   - keremberke/yolov8s-building-segmentation (small - balanced)
   - keremberke/yolov8m-building-segmentation (medium - most accurate)

2. ROOF DAMAGE DETECTION (Roboflow Universe):
   - SmartRoof/Roof-Damage - Classes: Blister, Chipped Shingle, Cracked Shingle, 
                             Degranulation, Dragon Tooth, Hail Impact, Mechanical Damage, Puncture
   - hail/roof-damage - Classes: Hail Damage
   - Hayden Claims Group/Storm-Damage-ID - Storm damage detection

3. GITHUB REPOSITORIES:
   - ManishSahu53/Vector-Map-Generation-from-Aerial-Imagery-using-Deep-Learning-GeoSpatial-UNET
   - obelling/ResUnet-model-for-building-segmentation
   - gunaykrgl/buildingSegmentation
"""

import os
import subprocess
import sys
from pathlib import Path


def ensure_dependencies():
    """Ensure required packages are installed."""
    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
        import huggingface_hub
    return huggingface_hub


def download_huggingface_model(model_id: str, output_dir: Path, filename: str = None):
    """Download model from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download
    
    print(f"\n[DOWNLOAD] {model_id}")
    
    try:
        # Download the model file
        model_path = hf_hub_download(
            repo_id=model_id,
            filename="best.pt" if filename is None else filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"   [OK] Downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"   [ERROR] {e}")
        return None


def download_from_ultralytics(model_name: str, output_dir: Path):
    """Download model using ultralytics package."""
    try:
        from ultralytics import YOLO
        print(f"\n[DOWNLOAD] {model_name}")
        
        # This will auto-download if not exists
        model = YOLO(model_name)
        
        # Copy to our models directory
        import shutil
        source = Path(model_name)
        if source.exists():
            dest = output_dir / model_name
            shutil.copy(source, dest)
            print(f"   [OK] Saved to: {dest}")
            return dest
        return model_name
    except Exception as e:
        print(f"   [ERROR] {e}")
        return None


def main():
    """Main download function."""
    print("=" * 60)
    print("PRE-TRAINED MODELS FOR ROOF/BUILDING DETECTION")
    print("=" * 60)
    
    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Ensure dependencies
    hf = ensure_dependencies()
    
    # =========================================================================
    # OPTION 1: YOLOv8 Building Segmentation from Hugging Face (RECOMMENDED)
    # =========================================================================
    print("\n" + "=" * 60)
    print("OPTION 1: YOLOv8 Building Segmentation (Hugging Face)")
    print("=" * 60)
    
    hf_models = {
        "building_nano": "keremberke/yolov8n-building-segmentation",
        "building_small": "keremberke/yolov8s-building-segmentation", 
        "building_medium": "keremberke/yolov8m-building-segmentation",
    }
    
    downloaded = {}
    for name, model_id in hf_models.items():
        result = download_huggingface_model(model_id, models_dir)
        if result:
            downloaded[name] = result
    
    # =========================================================================
    # OPTION 2: Base YOLOv8 Segmentation Models
    # =========================================================================
    print("\n" + "=" * 60)
    print("OPTION 2: Base YOLOv8 Segmentation Models (Ultralytics)")
    print("=" * 60)
    
    base_models = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt"]
    for model_name in base_models:
        download_from_ultralytics(model_name, models_dir)
    
    # =========================================================================
    # ROBOFLOW API INSTRUCTIONS
    # =========================================================================
    print("\n" + "=" * 60)
    print("OPTION 3: Roboflow Models (Manual Download)")
    print("=" * 60)
    
    roboflow_info = """
To download Roboflow models, you need a free API key:

1. Sign up at: https://app.roboflow.com/
2. Get your API key from Settings

AVAILABLE ROOF DAMAGE MODELS:
-----------------------------

A) SmartRoof Roof Damage Detection
   URL: https://universe.roboflow.com/smartroof/roof-damage
   Classes: Blister, Chipped Shingle, Cracked Shingle, Degranulation,
            Dragon Tooth, Hail Impact, Mechanical Damage, Puncture
   
   Download code:
   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace("smartroof").project("roof-damage")
   model = project.version(1).model
   # Export weights
   project.version(1).download("yolov8")
   ```

B) Hail Damage Detection
   URL: https://universe.roboflow.com/hail/roof-damage
   Classes: Hail Damage
   
C) Storm Damage ID
   URL: https://universe.roboflow.com/hayden-claims-group/storm-damage-id
   Classes: Various storm damage types

D) Roof Segmentation
   URL: https://universe.roboflow.com/loperoof/roof-segmentation
   Classes: Flat, Slope
"""
    print(roboflow_info)
    
    # =========================================================================
    # SUMMARY & CONFIGURATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("CONFIGURATION FOR YOUR PROJECT")
    print("=" * 60)
    
    config_example = """
Add to your .env file:
----------------------
# For Roof/Building Detection (use one):
ROOF_MODEL_PATH=./models/keremberke/yolov8m-building-segmentation/best.pt

# For Damage Detection (after downloading from Roboflow):
DAMAGE_MODEL_PATH=./models/roof_damage/best.pt

# Or use base models for fine-tuning:
# ROOF_MODEL_PATH=./models/yolov8m-seg.pt
# DAMAGE_MODEL_PATH=./models/yolov8m-seg.pt
"""
    print(config_example)
    
    # Create example .env file
    env_example = models_dir.parent / ".env.example"
    with open(env_example, "w", encoding="utf-8") as f:
        f.write("""# AI Roof Damage Detection Configuration
# =========================================

# Google Maps API Key (Required)
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# Model Paths
# -----------
# Option A: Pre-trained building segmentation (recommended)
ROOF_MODEL_PATH=./models/keremberke/yolov8m-building-segmentation/best.pt

# Option B: Base YOLOv8 (for fine-tuning)
# ROOF_MODEL_PATH=./models/yolov8m-seg.pt

# Damage Detection (download from Roboflow first)
DAMAGE_MODEL_PATH=./models/yolov8m-seg.pt

# Performance Settings
# --------------------
MAX_MEMORY_GB=4.0
BATCH_SIZE=4
CONFIDENCE_THRESHOLD=0.5

# API Settings
# ------------
RATE_LIMIT_QPS=10
""")
    print(f"\n[OK] Created example config: {env_example}")
    
    # List downloaded models
    print("\n" + "=" * 60)
    print("DOWNLOADED MODELS")
    print("=" * 60)
    
    for root, dirs, files in os.walk(models_dir):
        level = root.replace(str(models_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.pt'):
                filepath = Path(root) / file
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"{subindent}[MODEL] {file} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 60)
    print("[OK] SETUP COMPLETE!")
    print("=" * 60)
    print("""
Next Steps:
1. Copy .env.example to .env and add your Google Maps API key
2. Update model paths in .env to match downloaded models
3. Run: python -m api.main
4. Test: http://localhost:8000/docs
""")


if __name__ == "__main__":
    main()
