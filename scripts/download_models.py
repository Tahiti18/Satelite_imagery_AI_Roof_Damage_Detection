#!/usr/bin/env python3
"""
Download pre-trained models for roof and damage detection.
Run: python scripts/download_models.py
"""
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_with_progress(url: str, dest: Path, name: str):
    """Download file with progress indicator."""
    print(f"\n📥 Downloading {name}...")
    print(f"   URL: {url}")
    print(f"   Destination: {dest}")
    
    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        print(f"\r   Progress: {percent}%", end="", flush=True)
    
    try:
        urlretrieve(url, dest, reporthook=progress)
        print(f"\n   ✓ Downloaded: {dest.name}")
        return True
    except URLError as e:
        print(f"\n   ✗ Failed: {e}")
        return False


def setup_yolo_models():
    """Download YOLOv8 models optimized for aerial imagery."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("YOLOv8 Base Models for Fine-tuning")
    print("="*60)
    
    # YOLOv8 segmentation models (will auto-download on first use)
    # But we can pre-download for offline use
    from ultralytics import YOLO
    
    models_to_download = [
        ("yolov8n-seg.pt", "Nano - Fast, less accurate"),
        ("yolov8s-seg.pt", "Small - Balanced"),
        ("yolov8m-seg.pt", "Medium - Recommended for production"),
    ]
    
    for model_name, description in models_to_download:
        model_path = models_dir / model_name
        if model_path.exists():
            print(f"✓ {model_name} already exists ({description})")
        else:
            print(f"📥 Downloading {model_name} ({description})...")
            try:
                model = YOLO(model_name)
                # The model downloads to ultralytics cache, copy to our models dir
                import shutil
                cache_path = Path.home() / ".cache" / "ultralytics" / model_name
                if cache_path.exists():
                    shutil.copy(cache_path, model_path)
                print(f"   ✓ Downloaded {model_name}")
            except Exception as e:
                print(f"   ✗ Failed: {e}")


def setup_roboflow_models():
    """Instructions for Roboflow models."""
    print("\n" + "="*60)
    print("Roboflow Pre-trained Models (Recommended for Accuracy)")
    print("="*60)
    
    print("""
To get accurate roof detection models from Roboflow:

1. Create free account: https://app.roboflow.com/
2. Browse models: https://universe.roboflow.com/

Recommended datasets:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏠 ROOF DETECTION:
   • "aerial-building-segmentation" - Buildings from satellite
   • "rooftop-detection" - Rooftops specifically  
   • "solar-panel-detection" - Also detects roof areas

🔨 DAMAGE DETECTION:
   • "roof-damage-detection" - Hail, missing shingles
   • "construction-defects" - Cracks, blisters
   • "infrastructure-damage" - General damage patterns

Download command (after getting API key):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")

# Example: Download a roof detection model
project = rf.workspace("YOUR_WORKSPACE").project("roof-detection")
version = project.version(1)
dataset = version.download("yolov8")

# The model weights will be in: {dataset.location}/runs/detect/train/weights/best.pt
""")


def setup_huggingface_models():
    """Instructions for HuggingFace models."""
    print("\n" + "="*60)
    print("HuggingFace Models (Advanced)")
    print("="*60)
    
    print("""
For Segment Anything (SAM) - Zero-shot segmentation:

pip install segment-anything

# Download SAM checkpoint
from urllib.request import urlretrieve
urlretrieve(
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "models/sam_vit_h.pth"
)

Usage for roof extraction:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(satellite_image)

# Click on a roof to segment it
masks, _, _ = predictor.predict(point_coords=[[x, y]], point_labels=[1])
""")


def create_training_script():
    """Create a script for fine-tuning on custom data."""
    script_path = Path("scripts/train_custom_model.py")
    
    script_content = '''#!/usr/bin/env python3
"""
Fine-tune YOLOv8 on custom roof/damage dataset.

Dataset structure:
    data/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml

Run: python scripts/train_custom_model.py
"""
from ultralytics import YOLO
from pathlib import Path


def train_roof_detector():
    """Train roof detection model."""
    # Load pre-trained model
    model = YOLO("yolov8m-seg.pt")
    
    # Train on custom dataset
    results = model.train(
        data="data/roof_dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        name="roof_detector",
        patience=20,  # Early stopping
        save=True,
        device="0" if __import__("torch").cuda.is_available() else "cpu"
    )
    
    # Best model saved to: runs/segment/roof_detector/weights/best.pt
    print(f"\\nBest model: runs/segment/roof_detector/weights/best.pt")
    return results


def train_damage_detector():
    """Train damage detection model."""
    model = YOLO("yolov8m-seg.pt")
    
    results = model.train(
        data="data/damage_dataset/data.yaml",
        epochs=150,
        imgsz=640,
        batch=8,
        name="damage_detector",
        patience=25,
        save=True,
        device="0" if __import__("torch").cuda.is_available() else "cpu"
    )
    
    print(f"\\nBest model: runs/segment/damage_detector/weights/best.pt")
    return results


def create_dataset_yaml(name: str, classes: list):
    """Create data.yaml for training."""
    yaml_content = f"""
# {name} Dataset Configuration
path: data/{name}_dataset
train: train/images
val: val/images

# Classes
names:
"""
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\\n"
    
    return yaml_content


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train_custom_model.py [roof|damage]")
        print("\\nMake sure you have prepared your dataset first!")
        sys.exit(1)
    
    if sys.argv[1] == "roof":
        train_roof_detector()
    elif sys.argv[1] == "damage":
        train_damage_detector()
    else:
        print(f"Unknown model type: {sys.argv[1]}")
'''
    
    script_path.parent.mkdir(exist_ok=True)
    script_path.write_text(script_content, encoding='utf-8')
    print(f"\n[OK] Created training script: {script_path}")


def main():
    print("="*60)
    print("  AI Roof Damage Detection - Model Setup")
    print("="*60)
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    Path("scripts").mkdir(exist_ok=True)
    
    # Download base YOLO models
    setup_yolo_models()
    
    # Show Roboflow instructions
    setup_roboflow_models()
    
    # Show HuggingFace instructions
    setup_huggingface_models()
    
    # Create training script
    create_training_script()
    
    print("\n" + "="*60)
    print("  RECOMMENDED APPROACH")
    print("="*60)
    print("""
For BEST accuracy:

1. QUICK START (Good accuracy):
   → Use YOLOv8m-seg + Roboflow pre-trained weights
   → Download from: universe.roboflow.com

2. PRODUCTION (Best accuracy):
   → Collect 500+ labeled roof images
   → Fine-tune YOLOv8m-seg using train_custom_model.py
   → Expected mAP: 85-95%

3. ZERO-SHOT (No training needed):
   → Use Segment Anything (SAM) for roof extraction
   → Use CLIP for damage classification
   → Works on any roof without training

Model files should be placed in:
   models/roof_detector.pt
   models/damage_detector.pt

Then update config:
   ROOF_MODEL_PATH=./models/roof_detector.pt
   DAMAGE_MODEL_PATH=./models/damage_detector.pt
""")


if __name__ == "__main__":
    main()

