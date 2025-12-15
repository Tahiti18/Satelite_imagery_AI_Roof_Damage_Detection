#!/usr/bin/env python3
"""Quick test to see what the model detects."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.detection.roof_detector import RoofDetector

# Load model
print("Loading model...")
detector = RoofDetector(
    model_path="./models/roof_detector.pt",
    confidence_threshold=0.1,  # Very low to see all detections
    min_area_pixels=50
)

# Check if we have a stitched image
output_dir = Path("./output")
stitched_files = list(output_dir.glob("*_annotated.png"))

if stitched_files:
    image_path = stitched_files[0]
    print(f"\nTesting on: {image_path}")
    
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    print(f"Image shape: {img_array.shape}")
    
    # Run detection
    print("\nRunning detection...")
    results = detector.detect(img_array, return_masks=True)
    
    print(f"\nDetected {len(results)} roofs:")
    for i, r in enumerate(results[:10]):  # Show first 10
        print(f"  {i+1}. Confidence: {r.confidence:.3f}, Area: {r.area_pixels} pixels, BBox: {r.bbox}")
    
    if len(results) == 0:
        print("\n⚠️  No detections! Trying with even lower threshold...")
        detector.confidence_threshold = 0.01
        results = detector.detect(img_array, return_masks=True)
        print(f"With 0.01 threshold: {len(results)} detections")
        
        if len(results) > 0:
            print("\nFirst detection:")
            r = results[0]
            print(f"  Confidence: {r.confidence:.3f}")
            print(f"  Area: {r.area_pixels} pixels")
            print(f"  BBox: {r.bbox}")
else:
    print("No stitched image found. Run the full pipeline first.")

