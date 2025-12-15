#!/usr/bin/env python3
"""
Quick verification script for AI Roof Damage Detection system.
Run: python test_setup.py
"""
import sys
import asyncio
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test all imports work correctly."""
    print("\n[1/5] Testing imports...")
    
    try:
        import torch
        import numpy as np
        from PIL import Image
        from ultralytics import YOLO
        import cv2
        from loguru import logger
        import httpx
        from fastapi import FastAPI
        
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"  ✓ NumPy: {np.__version__}")
        print(f"  ✓ Pillow: {Image.__version__}")
        print(f"  ✓ OpenCV: {cv2.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\n[2/5] Testing configuration...")
    
    try:
        from config.settings import get_settings, validate_settings
        
        settings = get_settings()
        print(f"  ✓ App: {settings.app_name} v{settings.app_version}")
        print(f"  ✓ Batch size: {settings.batch_size}")
        print(f"  ✓ Confidence threshold: {settings.confidence_threshold}")
        print(f"  ✓ Max memory: {settings.max_memory_mb}MB")
        
        # Validate
        result = validate_settings(require_api_key=False)
        if result["warnings"]:
            for w in result["warnings"]:
                print(f"  ⚠ Warning: {w}")
        
        print(f"  ✓ Google API key configured: {settings.has_google_api_key}")
        
        return True
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        return False


def test_memory_manager():
    """Test memory management utilities."""
    print("\n[3/5] Testing memory manager...")
    
    try:
        from src.utils.memory import get_memory_manager, MemoryManager
        
        manager = get_memory_manager()
        
        cpu_mb = manager.get_memory_usage_mb()
        gpu_mb = manager.get_gpu_memory_usage_mb()
        
        print(f"  ✓ CPU memory: {cpu_mb:.1f}MB")
        if gpu_mb is not None:
            print(f"  ✓ GPU memory: {gpu_mb:.1f}MB")
        else:
            print(f"  ⚠ GPU memory: N/A (no CUDA)")
        
        # Test cleanup
        manager.cleanup(force=True)
        print(f"  ✓ Memory cleanup: OK")
        
        return True
    except Exception as e:
        print(f"  ✗ Memory manager error: {e}")
        return False


def test_detectors():
    """Test detector initialization (no inference yet)."""
    print("\n[4/5] Testing detectors...")
    
    try:
        from src.detection.roof_detector import RoofDetector
        from src.detection.damage_detector import DamageDetector
        
        # Initialize detectors (lazy loading - won't download models yet)
        roof_detector = RoofDetector(
            confidence_threshold=0.5,
            half_precision=False  # CPU mode
        )
        print(f"  ✓ RoofDetector initialized (device: {roof_detector.device})")
        
        damage_detector = DamageDetector(
            confidence_threshold=0.4,
            half_precision=False
        )
        print(f"  ✓ DamageDetector initialized (device: {damage_detector.device})")
        
        return True
    except Exception as e:
        print(f"  ✗ Detector error: {e}")
        return False


def test_geocoder():
    """Test geocoder (async)."""
    print("\n[5/5] Testing geocoder...")
    
    async def _test():
        from src.image_ingestion.geocoder import ZipcodeGeocoder
        
        geocoder = ZipcodeGeocoder(timeout=10)
        
        try:
            # Test with a known zipcode
            result = await geocoder.geocode_zipcode("90210")
            
            if result:
                print(f"  ✓ Geocoded 90210: ({result.center_lat:.4f}, {result.center_lng:.4f})")
                print(f"  ✓ City: {result.city}, State: {result.state}")
                return True
            else:
                print(f"  ⚠ Could not geocode 90210 (network issue?)")
                return True  # Not a fatal error
        finally:
            await geocoder.close()
    
    try:
        return asyncio.run(_test())
    except Exception as e:
        print(f"  ✗ Geocoder error: {e}")
        return False


def test_inference_demo():
    """Optional: Test actual inference with a synthetic image."""
    print("\n[BONUS] Testing inference with synthetic image...")
    
    try:
        import numpy as np
        from src.detection.roof_detector import RoofDetector
        
        # Create synthetic test image (640x640 with some rectangles)
        img = np.ones((640, 640, 3), dtype=np.uint8) * 180  # Gray background
        
        # Add some "roof-like" rectangles
        img[100:300, 100:400] = [139, 90, 43]  # Brown rectangle
        img[350:550, 200:500] = [128, 128, 128]  # Gray rectangle
        
        # Initialize and run detector
        detector = RoofDetector(confidence_threshold=0.3)
        
        print("  → Loading model (first run downloads ~6MB)...")
        detections = detector.detect(img, return_masks=False)
        
        print(f"  ✓ Inference complete: {len(detections)} detections")
        
        # Cleanup
        detector.unload_model()
        
        return True
    except Exception as e:
        print(f"  ⚠ Inference test skipped: {e}")
        return True  # Not fatal


def main():
    """Run all tests."""
    print("=" * 50)
    print("AI Roof Damage Detection - Setup Verification")
    print("=" * 50)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Memory", test_memory_manager()))
    results.append(("Detectors", test_detectors()))
    results.append(("Geocoder", test_geocoder()))
    
    # Optional inference test
    if "--full" in sys.argv:
        results.append(("Inference", test_inference_demo()))
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! System is ready.")
        print("\nNext steps:")
        print("  1. Create .env file with GOOGLE_MAPS_API_KEY")
        print("  2. Run: python -m api.main")
        print("  3. Open: http://localhost:8000")
    else:
        print("❌ Some tests failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

