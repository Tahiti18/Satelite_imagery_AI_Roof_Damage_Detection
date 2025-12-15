"""
Main entry point for AI Roof Damage Detection system.
Supports both CLI and API modes.
"""
import sys
import asyncio
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.pipeline import RoofDamagePipeline, PipelineConfig, analyze_zipcode_sync
from config.settings import get_settings, validate_settings

logger = get_logger(__name__)


def run_cli(args: argparse.Namespace) -> int:
    """Run command-line interface."""
    setup_logger(log_level=args.log_level.upper())
    
    try:
        settings = get_settings()
        api_key = settings.google_maps_api_key.get_secret_value()
    except Exception as e:
        print(f"Error: {e}")
        print("Please set GOOGLE_MAPS_API_KEY environment variable or in .env file")
        return 1
    
    config = PipelineConfig(
        tile_size=args.tile_size,
        zoom_level=args.zoom,
        roof_confidence=args.confidence,
        damage_confidence=args.confidence,
        output_dir=args.output,
        save_visualization=not args.no_vis,
        save_heatmap=not args.no_heatmap,
        save_json=True,
        save_geojson=args.geojson
    )
    
    print(f"Analyzing zipcode: {args.zipcode}")
    print(f"Output directory: {args.output}")
    print("-" * 40)
    
    try:
        result = analyze_zipcode_sync(args.zipcode, api_key, config)
        
        print("\n=== Analysis Results ===")
        print(f"Zipcode: {result.zipcode}")
        print(f"Processing time: {result.processing_time_sec:.1f}s")
        print(f"Tiles processed: {result.tiles_processed}")
        print(f"Total roofs: {result.total_roofs}")
        print(f"Roofs with damage: {result.roofs_with_damage}")
        print(f"\nDamage by severity:")
        for severity, count in result.damage_summary.items():
            print(f"  {severity}: {count}")
        print(f"\nDamage by type:")
        for dtype, count in result.damage_types_summary.items():
            print(f"  {dtype}: {count}")
        print(f"\nOutput files saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Analysis failed")
        return 1


def run_api(args: argparse.Namespace) -> int:
    """Run FastAPI server."""
    import uvicorn
    
    setup_logger(log_level=args.log_level.upper())
    
    print(f"Starting API server on {args.host}:{args.port}")
    print(f"Documentation: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1  # Single worker for GPU models
    )
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Roof Damage Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # CLI subcommand
    cli_parser = subparsers.add_parser("analyze", help="Analyze a zipcode")
    cli_parser.add_argument("zipcode", help="US zipcode (5 digits)")
    cli_parser.add_argument("-o", "--output", default="./output", help="Output directory")
    cli_parser.add_argument("-z", "--zoom", type=int, default=20, help="Zoom level (15-21)")
    cli_parser.add_argument("-t", "--tile-size", type=int, default=640, help="Tile size in pixels")
    cli_parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Confidence threshold")
    cli_parser.add_argument("--no-vis", action="store_true", help="Skip visualization")
    cli_parser.add_argument("--no-heatmap", action="store_true", help="Skip heatmap")
    cli_parser.add_argument("--geojson", action="store_true", help="Save GeoJSON output")
    cli_parser.add_argument("--log-level", default="info", help="Log level")
    
    # API subcommand
    api_parser = subparsers.add_parser("serve", help="Start API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    api_parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        sys.exit(run_cli(args))
    elif args.command == "serve":
        sys.exit(run_api(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()

