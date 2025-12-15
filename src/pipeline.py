"""
Main pipeline orchestration.
Coordinates all components for zipcode-based roof damage detection.
"""
import asyncio
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from loguru import logger

from .utils.memory import memory_efficient, get_memory_manager
from .utils.logger import setup_logger
from .utils.perf import PerformanceProfiler, ProcessingMetrics
from .image_ingestion import ZipcodeGeocoder, SatelliteImageFetcher, ImageStitcher, ZipcodeInfo
from .detection import RoofDetector, DamageDetector, RoofDetection, DamageDetection
from .output import ResultGenerator, AnalysisResult, Visualizer
from config.settings import Settings, get_settings


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Image fetching - OPTIMIZED FOR MAXIMUM ACCURACY
    tile_size: int = 256  # MapTiler standard tile size
    zoom_level: int = 21  # MAXIMUM zoom for closest view (0.075m/pixel - can see individual roof tiles)
    max_concurrent_downloads: int = 10  # Conservative to avoid API throttling
    tile_overlap: float = 0.0  # NO overlap - fetch exact area only
    use_cache: bool = True
    cache_dir: str = "./cache/tiles"
    
    # Detection - OPTIMIZED FOR ACCURACY
    roof_confidence: float = 0.2  # Lower threshold for better detection
    damage_confidence: float = 0.25  # Lower threshold for damage detection
    min_roof_area: int = 100
    min_damage_area: int = 25
    
    # Output
    output_dir: str = "./output"
    save_visualization: bool = True
    save_heatmap: bool = True
    save_json: bool = True
    save_geojson: bool = True
    
    # Models
    roof_model_path: Optional[str] = None
    damage_model_path: Optional[str] = None


class RoofDamagePipeline:
    """
    Main pipeline for zipcode-based roof damage detection.
    
    Usage:
        pipeline = RoofDamagePipeline(api_key="YOUR_KEY")
        result = await pipeline.analyze_zipcode("75201")
    """
    
    def __init__(
        self,
        api_key: str,
        config: Optional[PipelineConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            api_key: MapTiler API key
            config: Pipeline configuration
            device: Device for inference ('cuda', 'cpu', or None for auto)
        """
        self.config = config or PipelineConfig()
        self.device = device
        
        # Initialize components (lazy loaded)
        self._geocoder: Optional[ZipcodeGeocoder] = None
        self._fetcher: Optional[SatelliteImageFetcher] = None
        self._stitcher: Optional[ImageStitcher] = None
        self._roof_detector: Optional[RoofDetector] = None
        self._damage_detector: Optional[DamageDetector] = None
        self._result_generator: Optional[ResultGenerator] = None
        self._visualizer: Optional[Visualizer] = None
        
        self._api_key = api_key
        self._initialized = False
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("RoofDamagePipeline created")
    
    async def _initialize(self) -> None:
        """Lazy initialization of components."""
        if self._initialized:
            return
        
        logger.info("Initializing pipeline components...")
        
        # Initialize geocoder
        self._geocoder = ZipcodeGeocoder()
        
        # Initialize image fetcher - OPTIMIZED FOR ACCURACY
        self._fetcher = SatelliteImageFetcher(
            api_key=self._api_key,
            tile_size=self.config.tile_size,
            zoom=self.config.zoom_level,
            max_concurrent=self.config.max_concurrent_downloads,
            requests_per_second=10,  # Rate limit: 10 req/sec (safe for API)
            cache_dir=self.config.cache_dir if self.config.use_cache else None
        )
        
        # Initialize stitcher
        self._stitcher = ImageStitcher()
        
        # Initialize detectors
        self._roof_detector = RoofDetector(
            model_path=self.config.roof_model_path,
            confidence_threshold=self.config.roof_confidence,
            min_area_pixels=self.config.min_roof_area,
            device=self.device
        )
        
        self._damage_detector = DamageDetector(
            model_path=self.config.damage_model_path,
            confidence_threshold=self.config.damage_confidence,
            min_area_pixels=self.config.min_damage_area,
            device=self.device
        )
        
        # Initialize output generators
        self._result_generator = ResultGenerator()
        self._visualizer = Visualizer()
        
        self._initialized = True
        logger.info("Pipeline initialized")
    
    async def close(self) -> None:
        """Cleanup resources."""
        if self._geocoder:
            await self._geocoder.close()
        if self._fetcher:
            await self._fetcher.close()
        if self._roof_detector:
            self._roof_detector.unload_model()
        if self._damage_detector:
            self._damage_detector.unload_model()
        
        get_memory_manager().cleanup(force=True)
        logger.info("Pipeline closed")
    
    @memory_efficient(cleanup_after=True)
    async def analyze_zipcode(
        self,
        zipcode: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AnalysisResult:
        """
        Analyze all roofs in a zipcode for damage.
        
        This method coordinates the complete pipeline:
        geocoding, image fetching, stitching, detection
        and result generation. Performance metrics are
        collected for each major stage to help identify
        bottlenecks.
        
        Args:
            zipcode: US zipcode (5 digits)
            progress_callback: Optional callback(stage, progress_percent)
            
        Returns:
            AnalysisResult with all detections and performance data.
        """
        start_time = time.time()
        profiler = PerformanceProfiler()
        metrics = ProcessingMetrics(zipcode=zipcode)
        
        # Initialize if needed
        with profiler.profile("initialize"):
            await self._initialize()
        
        def report_progress(stage: str, progress: float):
            if progress_callback:
                progress_callback(stage, progress)
            logger.debug(f"{stage}: {progress:.0%}")
        
        # Stage 1: Geocode zipcode
        report_progress("Geocoding", 0.0)
        geocode_start = time.perf_counter()
        zipcode_info = await self._geocoder.geocode_zipcode(zipcode)
        metrics.mark_stage("geocoding", time.perf_counter() - geocode_start)
        if zipcode_info is None:
            raise ValueError(f"Invalid zipcode: {zipcode}")
        report_progress("Geocoding", 1.0)
        
        # Stage 2: Fetch satellite images - EXACT AREA ONLY (no overlap)
        report_progress("Fetching images", 0.0)
        with profiler.profile("fetch_tiles"):
            fetch_start = time.perf_counter()
            tiles = await self._fetcher.fetch_area(
                zipcode_info.bounding_box,
                use_cache=self.config.use_cache,
                overlap_percent=self.config.tile_overlap,  # 0.0 = exact area, no overlap
                progress_callback=lambda done, total: report_progress("Fetching images", done/total)
            )
            metrics.mark_stage("fetch_tiles", time.perf_counter() - fetch_start)
        
        if not tiles:
            raise RuntimeError("Failed to fetch satellite imagery")
        
        # Memory cleanup after fetching
        get_memory_manager().cleanup()
        
        # Stage 3: Stitch images
        report_progress("Stitching images", 0.0)
        with profiler.profile("stitch"):
            stitch_start = time.perf_counter()
            stitched = self._stitcher.stitch(
                tiles,
                min_lat=zipcode_info.bounding_box.min_lat,
                max_lat=zipcode_info.bounding_box.max_lat,
                min_lng=zipcode_info.bounding_box.min_lng,
                max_lng=zipcode_info.bounding_box.max_lng
            )
            metrics.mark_stage("stitch", time.perf_counter() - stitch_start)
        report_progress("Stitching images", 1.0)
        
        # Cleanup tile data after stitching (keep only stitched image)
        get_memory_manager().cleanup_intermediate_data(tiles)
        
        # Stage 4: Detect roofs
        report_progress("Detecting roofs", 0.0)
        with profiler.profile("detect_roofs"):
            detect_roofs_start = time.perf_counter()
            roofs = self._roof_detector.detect(stitched.image, return_masks=True)
            metrics.mark_stage("detect_roofs", time.perf_counter() - detect_roofs_start)
        report_progress("Detecting roofs", 1.0)
        
        # Memory cleanup after roof detection
        get_memory_manager().cleanup()
        
        # Stage 5: Detect damage on each roof
        report_progress("Detecting damage", 0.0)
        all_damages = []
        with profiler.profile("detect_damage"):
            damage_start = time.perf_counter()
            for i, roof in enumerate(roofs):
                damages = self._damage_detector.detect_on_roof(
                    stitched.image,
                    roof,
                    return_masks=True
                )
                # Cleanup after each roof to prevent memory buildup
                if i % 10 == 0:  # Cleanup every 10 roofs
                    get_memory_manager().cleanup()
                all_damages.extend(damages)
                report_progress("Detecting damage", (i + 1) / len(roofs) if roofs else 1.0)
            metrics.mark_stage("detect_damage", time.perf_counter() - damage_start)
        
        # Stage 6: Generate results
        report_progress("Generating results", 0.0)
        processing_time = time.time() - start_time
        metrics.end_time = time.time()
        metrics.profiler_stats = profiler.get_stats()
        
        result = self._result_generator.create_result(
            zipcode_info=zipcode_info,
            roofs=roofs,
            damages=all_damages,
            image_width=stitched.width,
            image_height=stitched.height,
            tiles_processed=stitched.n_tiles,
            processing_time=processing_time,
            performance_metrics=metrics.to_dict(),
        )
        
        # Save outputs
        output_base = Path(self.config.output_dir) / f"{zipcode}_{int(time.time())}"
        
        if self.config.save_json:
            self._result_generator.save_json(result, f"{output_base}.json")
        
        if self.config.save_geojson:
            self._result_generator.save_geojson(result, f"{output_base}.geojson")
        
        if self.config.save_visualization:
            vis_image = self._visualizer.draw_all(
                stitched.image, roofs, all_damages,
                draw_masks=True, draw_boxes=True, draw_labels=False
            )
            vis_image = self._visualizer.add_summary_overlay(
                vis_image,
                result.total_roofs,
                result.roofs_with_damage,
                result.damage_summary
            )
            self._visualizer.save(vis_image, f"{output_base}_annotated.png")
        
        if self.config.save_heatmap and all_damages:
            heatmap = self._visualizer.generate_heatmap(stitched.image, all_damages)
            self._visualizer.save(heatmap, f"{output_base}_heatmap.png")
        
        report_progress("Complete", 1.0)
        
        logger.info(
            f"Analysis complete for {zipcode}: "
            f"{result.total_roofs} roofs, {len(all_damages)} damages, "
            f"{processing_time:.1f}s"
        )
        
        return result


# Synchronous wrapper for simple usage
def analyze_zipcode_sync(
    zipcode: str,
    api_key: str,
    config: Optional[PipelineConfig] = None
) -> AnalysisResult:
    """
    Synchronous wrapper for analyze_zipcode.
    
    Args:
        zipcode: US zipcode
        api_key: Google Maps API key
        config: Optional pipeline configuration
        
    Returns:
        AnalysisResult
    """
    async def _run():
        pipeline = RoofDamagePipeline(api_key, config)
        try:
            return await pipeline.analyze_zipcode(zipcode)
        finally:
            await pipeline.close()
    
    return asyncio.run(_run())

