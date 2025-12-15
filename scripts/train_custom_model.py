#!/usr/bin/env python3
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
    print(f"\nBest model: runs/segment/roof_detector/weights/best.pt")
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
    
    print(f"\nBest model: runs/segment/damage_detector/weights/best.pt")
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
        yaml_content += f"  {i}: {cls}\n"
    
    return yaml_content


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train_custom_model.py [roof|damage]")
        print("\nMake sure you have prepared your dataset first!")
        sys.exit(1)
    
    if sys.argv[1] == "roof":
        train_roof_detector()
    elif sys.argv[1] == "damage":
        train_damage_detector()
    else:
        print(f"Unknown model type: {sys.argv[1]}")
