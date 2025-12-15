#!/usr/bin/env python3
"""
Simple test script to analyze a zipcode.
Usage: python test_zipcode.py 75201
"""
import sys
import asyncio
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RoofDamagePipeline, PipelineConfig
from config.settings import get_settings


async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_zipcode.py <zipcode>")
        print("Example: python test_zipcode.py 75201")
        sys.exit(1)
    
    zipcode = sys.argv[1]
    
    if not zipcode.isdigit() or len(zipcode) != 5:
        print("Error: Zipcode must be 5 digits")
        sys.exit(1)
    
    print("=" * 60)
    print(f"Testing Roof Damage Detection for Zipcode: {zipcode}")
    print("=" * 60)
    
    # Get settings
    settings = get_settings()
    
    if not settings.has_maptiler_api_key:
        print("ERROR: MAPTILER_API_KEY not set in .env file")
        sys.exit(1)
    
    api_key = settings.maptiler_api_key.get_secret_value()
    
    # Create pipeline config
    config = PipelineConfig(
        tile_size=256,  # MapTiler standard tile size
        zoom_level=21,  # MAXIMUM zoom for closest view (0.075m/pixel)
        roof_confidence=0.2,  # Lower threshold for better detection
        damage_confidence=0.25,  # Lower threshold for damage detection
        output_dir="./output",
        cache_dir="./cache/tiles",
        roof_model_path=settings.roof_model_path,
        damage_model_path=settings.damage_model_path
    )
    
    # Create pipeline
    print("\n[1/5] Initializing pipeline...")
    pipeline = RoofDamagePipeline(api_key=api_key, config=config)
    
    try:
        print("[2/5] Geocoding zipcode...")
        print("[3/5] Fetching satellite images...")
        print("[4/5] Detecting roofs...")
        print("[5/5] Detecting damage...")
        print("\n" + "-" * 60)
        
        # Run analysis
        result = await pipeline.analyze_zipcode(zipcode)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"\nResults for zipcode {zipcode}:")
        print(f"  - Total roofs detected: {result.total_roofs}")
        print(f"  - Roofs with damage: {result.roofs_with_damage}")
        print(f"  - Total damage area (pixels): {result.total_damage_area_pixels}")
        if result.total_roofs > 0:
            print(f"  - Damage summary: {result.damage_summary}")
        
        print(f"\nOutput files saved to: ./output/")
        print(f"  - JSON: {zipcode}_analysis.json")
        print(f"  - GeoJSON: {zipcode}_analysis.geojson")
        if result.total_roofs > 0:
            print(f"  - Visualization: {zipcode}_annotated.png")
            print(f"  - Heatmap: {zipcode}_heatmap.png")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())

