#!/usr/bin/env python3
"""
Complete Model Downloader for AI Roof Damage Detection System

AVAILABLE PRE-TRAINED MODELS:
=============================

1. ROOF/BUILDING SEGMENTATION (Hugging Face - Ready to use):
   - keremberke/yolov8n-building-segmentation (6.4 MB - fastest)
   - keremberke/yolov8s-building-segmentation (22 MB - balanced)
   - keremberke/yolov8m-building-segmentation (52 MB - most accurate)

2. ROOF DAMAGE DETECTION (Hugging Face):
   - JGuevara-12/yolo-roof-damage
   - jobejaranom/yolo-roof-damage

3. ROOF DAMAGE DETECTION (Roboflow - requires free API key):
   - SmartRoof/Roof-Damage (best classes: Blister, Chipped Shingle, etc.)
   - hail/roof-damage
   - Hayden-Claims-Group/Storm-Damage-ID

4. GITHUB REPOS (for reference/training):
   - kevinkepp/rooftop-seg
   - Dmytro-Shvetsov/rooftop-segmentation
   - vituenrique/RooftopSegmentation
"""

import os
import sys
import subprocess
from pathlib import Path


def install_package(package):
    """Install a package if not present."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])


def ensure_dependencies():
    """Ensure all required packages are installed."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[INSTALL] huggingface_hub...")
        install_package("huggingface_hub")
        from huggingface_hub import hf_hub_download
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[INSTALL] ultralytics...")
        install_package("ultralytics")
        from ultralytics import YOLO
    
    return True


def download_hf_model(model_id, models_dir, filename="best.pt"):
    """Download model from Hugging Face."""
    from huggingface_hub import hf_hub_download
    
    print(f"\n[DOWNLOAD] {model_id}")
    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print(f"   [OK] {path}")
        return path
    except Exception as e:
        print(f"   [ERROR] {e}")
        # Try with different filename
        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename="model.pt",
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            print(f"   [OK] {path}")
            return path
        except:
            pass
        return None


def download_ultralytics_model(model_name, models_dir):
    """Download base YOLOv8 model."""
    from ultralytics import YOLO
    import shutil
    
    print(f"\n[DOWNLOAD] {model_name} (Ultralytics)")
    try:
        model = YOLO(model_name)
        
        # Copy to models dir
        source = Path(model_name)
        if source.exists():
            dest = models_dir / model_name
            shutil.copy(source, dest)
            print(f"   [OK] {dest}")
            return dest
        return model_name
    except Exception as e:
        print(f"   [ERROR] {e}")
        return None


def main():
    print("=" * 70)
    print("AI ROOF DAMAGE DETECTION - MODEL DOWNLOADER")
    print("=" * 70)
    
    # Setup
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    models_dir = project_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Ensure dependencies
    print("\n[SETUP] Checking dependencies...")
    ensure_dependencies()
    
    # =========================================================================
    # SECTION 1: ROOF/BUILDING SEGMENTATION MODELS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SECTION 1: ROOF/BUILDING SEGMENTATION MODELS")
    print("=" * 70)
    
    building_models = [
        "keremberke/yolov8n-building-segmentation",
        "keremberke/yolov8s-building-segmentation",
        "keremberke/yolov8m-building-segmentation",
    ]
    
    roof_model_path = None
    for model_id in building_models:
        result = download_hf_model(model_id, models_dir)
        if result and "yolov8m" in model_id:
            roof_model_path = result  # Use medium as default
    
    # =========================================================================
    # SECTION 2: ROOF DAMAGE DETECTION MODELS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SECTION 2: ROOF DAMAGE DETECTION MODELS")
    print("=" * 70)
    
    damage_models = [
        "JGuevara-12/yolo-roof-damage",
        "jobejaranom/yolo-roof-damage",
    ]
    
    damage_model_path = None
    for model_id in damage_models:
        result = download_hf_model(model_id, models_dir)
        if result:
            damage_model_path = result
    
    # =========================================================================
    # SECTION 3: BASE YOLOV8 MODELS (Fallback)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SECTION 3: BASE YOLOV8 MODELS (Fallback)")
    print("=" * 70)
    
    base_models = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt"]
    for model_name in base_models:
        download_ultralytics_model(model_name, models_dir)
    
    # =========================================================================
    # SECTION 4: ROBOFLOW MODELS (Manual)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SECTION 4: ROBOFLOW MODELS (Manual Download)")
    print("=" * 70)
    
    roboflow_guide = """
For BEST roof damage detection, get models from Roboflow Universe:

1. Create FREE account: https://app.roboflow.com/

2. RECOMMENDED MODELS:
   
   A) SmartRoof - Roof Damage Detection (BEST)
      URL: https://universe.roboflow.com/smartroof/roof-damage
      Classes: Blister, Chipped Shingle, Cracked Shingle, 
               Degranulation, Dragon Tooth, Hail Impact,
               Mechanical Damage, Puncture
      
   B) Hail Damage Detection
      URL: https://universe.roboflow.com/hail/roof-damage
      
   C) Storm Damage ID
      URL: https://universe.roboflow.com/hayden-claims-group/storm-damage-id

3. DOWNLOAD CODE (after getting API key):
   
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")
   
   # Download SmartRoof model
   project = rf.workspace().project("roof-damage")
   version = project.version(1)
   version.download("yolov8")
"""
    print(roboflow_guide)
    
    # =========================================================================
    # GENERATE CONFIG FILE
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING CONFIGURATION")
    print("=" * 70)
    
    # Create .env file
    env_file = project_dir / ".env"
    env_content = f"""# AI Roof Damage Detection - Configuration
# ==========================================

# Google Maps API Key (REQUIRED)
GOOGLE_MAPS_API_KEY=your_api_key_here

# Model Paths
# -----------
# Roof/Building Detection (Pre-trained from Hugging Face)
ROOF_MODEL_PATH=./models/keremberke/yolov8m-building-segmentation/best.pt

# Damage Detection (Use one of the options below)
# Option 1: Pre-trained from Hugging Face
DAMAGE_MODEL_PATH=./models/JGuevara-12/yolo-roof-damage/best.pt
# Option 2: Base YOLOv8 (for fine-tuning)
# DAMAGE_MODEL_PATH=./models/yolov8m-seg.pt

# Processing Settings
# -------------------
MAX_MEMORY_GB=4.0
BATCH_SIZE=4
CONFIDENCE_THRESHOLD=0.5
MAX_IMAGE_SIZE=1280

# API Settings
# ------------
RATE_LIMIT_QPS=10
DEBUG=false
"""
    
    if not env_file.exists():
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(env_content)
        print(f"[CREATED] {env_file}")
    else:
        print(f"[EXISTS] {env_file} - not overwriting")
    
    # Create .env.example
    example_file = project_dir / ".env.example"
    with open(example_file, "w", encoding="utf-8") as f:
        f.write(env_content)
    print(f"[CREATED] {example_file}")
    
    # =========================================================================
    # LIST ALL DOWNLOADED MODELS
    # =========================================================================
    print("\n" + "=" * 70)
    print("DOWNLOADED MODELS")
    print("=" * 70)
    
    total_size = 0
    model_count = 0
    
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            if f.endswith('.pt'):
                filepath = Path(root) / f
                size_mb = filepath.stat().st_size / (1024 * 1024)
                total_size += size_mb
                model_count += 1
                rel_path = filepath.relative_to(models_dir)
                print(f"  [MODEL] {rel_path} ({size_mb:.1f} MB)")
    
    print(f"\nTotal: {model_count} models, {total_size:.1f} MB")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    
    summary = """
WHAT YOU HAVE NOW:
------------------
1. Building/Roof Segmentation models (keremberke/yolov8*-building-segmentation)
   - Trained on satellite imagery for building detection
   - Ready to use for roof detection!

2. Base YOLOv8 segmentation models (yolov8*-seg.pt)
   - For fine-tuning on custom data

3. Roof damage models from Hugging Face (if available)

NEXT STEPS:
-----------
1. Add your Google Maps API key to .env file
2. Run the API: python -m api.main
3. Test: http://localhost:8000/docs

FOR BETTER DAMAGE DETECTION:
----------------------------
- Sign up at Roboflow (free)
- Download SmartRoof model (has 8 damage classes!)
- Update DAMAGE_MODEL_PATH in .env
"""
    print(summary)


if __name__ == "__main__":
    main()

