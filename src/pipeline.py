"""
Main pipeline orchestration.
Coordinates zipcode-based and address-based roof damage detection.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

from loguru import logger

from .utils.memory import memory_efficient, get_memory_manager
from .utils.perf import PerformanceProfiler, ProcessingMetrics
from .image_ingestion import ZipcodeGeocoder, SatelliteImageFetcher, ImageStitcher
from .detection import RoofDetector, DamageDetector
from .output import ResultGenerator, AnalysisResult, Visualizer


RAILWAY_SAFE_ZOOM_LEVEL = 17


@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    # Image fetching - Railway-safe defaults
    tile_size: int = 256
    zoom_level: int = RAILWAY_SAFE_ZOOM_LEVEL
    max_concurrent_downloads: int = 5
    tile_overlap: float = 0.0
    use_cache: bool = True
    cache_dir: str = "/tmp/tiles"

    # Detection
    roof_confidence: float = 0.2
    damage_confidence: float = 0.25
    min_roof_area: int = 100
    min_damage_area: int = 25

    # Output
    output_dir: str = "/tmp/output"
    save_visualization: bool = True
    save_heatmap: bool = True
    save_json: bool = True
    save_geojson: bool = True

    # Models
    roof_model_path: Optional[str] = None
    damage_model_path: Optional[str] = None


class RoofDamagePipeline:
    """
    Main pipeline for zipcode/address-based roof damage detection.
    """

    def __init__(
        self,
        api_key: str,
        config: Optional[PipelineConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or PipelineConfig()
        self.device = device

        # Railway safety guard
        if self.config.zoom_level > RAILWAY_SAFE_ZOOM_LEVEL:
            logger.warning(
                f"Requested zoom_level={self.config.zoom_level}; "
                f"forcing zoom_level={RAILWAY_SAFE_ZOOM_LEVEL}."
            )
            self.config.zoom_level = RAILWAY_SAFE_ZOOM_LEVEL

        self.config.tile_size = 256
        self.config.max_concurrent_downloads = min(self.config.max_concurrent_downloads, 5)
        self.config.cache_dir = "/tmp/tiles"
        self.config.output_dir = "/tmp/output"

        self._geocoder: Optional[ZipcodeGeocoder] = None
        self._fetcher: Optional[SatelliteImageFetcher] = None
        self._stitcher: Optional[ImageStitcher] = None
        self._roof_detector: Optional[RoofDetector] = None
        self._damage_detector: Optional[DamageDetector] = None
        self._result_generator: Optional[ResultGenerator] = None
        self._visualizer: Optional[Visualizer] = None

        self._api_key = api_key
        self._initialized = False

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"RoofDamagePipeline created "
            f"(zoom={self.config.zoom_level}, tile_size={self.config.tile_size})"
        )

    async def _initialize(self) -> None:
        """Lazy initialization of components."""
        if self._initialized:
            return

        logger.info("Initializing pipeline components...")

        self._geocoder = ZipcodeGeocoder()

        self._fetcher = SatelliteImageFetcher(
            api_key=self._api_key,
            tile_size=256,
            zoom=RAILWAY_SAFE_ZOOM_LEVEL,
            max_concurrent=5,
            requests_per_second=5,
            cache_dir=self.config.cache_dir if self.config.use_cache else None,
        )

        logger.info(
            f"SatelliteImageFetcher initialized with forced zoom={RAILWAY_SAFE_ZOOM_LEVEL}"
        )

        self._stitcher = ImageStitcher()

        self._roof_detector = RoofDetector(
            model_path=self.config.roof_model_path,
            confidence_threshold=self.config.roof_confidence,
            min_area_pixels=self.config.min_roof_area,
            device=self.device,
        )

        self._damage_detector = DamageDetector(
            model_path=self.config.damage_model_path,
            confidence_threshold=self.config.damage_confidence,
            min_area_pixels=self.config.min_damage_area,
            device=self.device,
        )

        self._result_generator = ResultGenerator(output_dir=self.config.output_dir)
        self._visualizer = Visualizer(output_dir=self.config.output_dir)

        self._initialized = True
        logger.info("Pipeline initialized")

    async def close(self) -> None:
        """Cleanup resources."""
        if self._geocoder:
            await self._geocoder.close()

        if self._fetcher:
            await self._fetcher.close()

        if self._roof_detector and hasattr(self._roof_detector, "unload_model"):
            self._roof_detector.unload_model()

        if self._damage_detector and hasattr(self._damage_detector, "unload_model"):
            self._damage_detector.unload_model()

        get_memory_manager().cleanup(force=True)
        logger.info("Pipeline closed")

    def _reporter(
        self,
        progress_callback: Optional[Callable[[str, float], None]],
    ) -> Callable[[str, float], None]:
        def report_progress(stage: str, progress: float) -> None:
            if progress_callback:
                progress_callback(stage, progress)
            logger.debug(f"{stage}: {progress:.0%}")

        return report_progress

    async def _run_detection_for_area(
        self,
        area_info,
        label_for_metrics: str,
        output_name: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AnalysisResult:
        start_time = time.time()
        profiler = PerformanceProfiler()
        metrics = ProcessingMetrics(zipcode=label_for_metrics)

        with profiler.profile("initialize"):
            await self._initialize()

        assert self._fetcher is not None
        assert self._stitcher is not None
        assert self._roof_detector is not None
        assert self._damage_detector is not None
        assert self._result_generator is not None
        assert self._visualizer is not None

        report_progress = self._reporter(progress_callback)

        report_progress("Fetching images", 0.0)

        with profiler.profile("fetch_tiles"):
            fetch_start = time.perf_counter()

            tiles = await self._fetcher.fetch_area(
                area_info.bounding_box,
                use_cache=self.config.use_cache,
                overlap_percent=self.config.tile_overlap,
                progress_callback=lambda done, total: report_progress(
                    "Fetching images",
                    done / total if total else 1.0,
                ),
            )

            metrics.mark_stage("fetch_tiles", time.perf_counter() - fetch_start)

        if not tiles:
            raise RuntimeError("Failed to fetch satellite imagery")

        get_memory_manager().cleanup()

        report_progress("Stitching images", 0.0)

        with profiler.profile("stitch"):
            stitch_start = time.perf_counter()

            stitched = self._stitcher.stitch(
                tiles,
                min_lat=area_info.bounding_box.min_lat,
                max_lat=area_info.bounding_box.max_lat,
                min_lng=area_info.bounding_box.min_lng,
                max_lng=area_info.bounding_box.max_lng,
            )

            metrics.mark_stage("stitch", time.perf_counter() - stitch_start)

        report_progress("Stitching images", 1.0)

        get_memory_manager().cleanup_intermediate_data(tiles)

        report_progress("Detecting roofs", 0.0)

        with profiler.profile("detect_roofs"):
            detect_roofs_start = time.perf_counter()
            roofs = self._roof_detector.detect(stitched.image, return_masks=True)
            metrics.mark_stage("detect_roofs", time.perf_counter() - detect_roofs_start)

        report_progress("Detecting roofs", 1.0)

        get_memory_manager().cleanup()

        report_progress("Detecting damage", 0.0)

        all_damages = []

        with profiler.profile("detect_damage"):
            damage_start = time.perf_counter()

            if roofs:
                for i, roof in enumerate(roofs):
                    damages = self._damage_detector.detect_on_roof(
                        stitched.image,
                        roof,
                        return_masks=True,
                    )

                    if i % 10 == 0:
                        get_memory_manager().cleanup()

                    all_damages.extend(damages)

                    report_progress("Detecting damage", (i + 1) / len(roofs))
            else:
                report_progress("Detecting damage", 1.0)

            metrics.mark_stage("detect_damage", time.perf_counter() - damage_start)

        report_progress("Generating results", 0.0)

        processing_time = time.time() - start_time
        metrics.end_time = time.time()
        metrics.profiler_stats = profiler.get_stats()

        result = self._result_generator.create_result(
            zipcode_info=area_info,
            roofs=roofs,
            damages=all_damages,
            image_width=stitched.width,
            image_height=stitched.height,
            tiles_processed=stitched.n_tiles,
            processing_time=processing_time,
            performance_metrics=metrics.to_dict(),
        )

        output_base = Path(self.config.output_dir) / f"{output_name}_{int(time.time())}"

        if self.config.save_json:
            self._result_generator.save_json(result, f"{output_base}.json")

        if self.config.save_geojson:
            self._result_generator.save_geojson(result, f"{output_base}.geojson")

        if self.config.save_visualization:
            vis_image = self._visualizer.draw_all(
                stitched.image,
                roofs,
                all_damages,
                draw_masks=True,
                draw_boxes=True,
                draw_labels=False,
            )

            vis_image = self._visualizer.add_summary_overlay(
                vis_image,
                result.total_roofs,
                result.roofs_with_damage,
                result.damage_summary,
            )

            self._visualizer.save(vis_image, f"{output_base}_annotated.png")

        if self.config.save_heatmap and all_damages:
            heatmap = self._visualizer.generate_heatmap(stitched.image, all_damages)
            self._visualizer.save(heatmap, f"{output_base}_heatmap.png")

        report_progress("Complete", 1.0)

        logger.info(
            f"Analysis complete for {output_name}: "
            f"{result.total_roofs} roofs, {len(all_damages)} damages, "
            f"{processing_time:.1f}s"
        )

        return result

    @memory_efficient(cleanup_after=True)
    async def analyze_zipcode(
        self,
        zipcode: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AnalysisResult:
        await self._initialize()

        assert self._geocoder is not None

        report_progress = self._reporter(progress_callback)

        report_progress("Geocoding", 0.0)

        geocode_start = time.perf_counter()
        zipcode_info = await self._geocoder.geocode_zipcode(zipcode)

        if zipcode_info is None:
            raise ValueError(f"Invalid zipcode: {zipcode}")

        report_progress("Geocoding", 1.0)

        result = await self._run_detection_for_area(
            area_info=zipcode_info,
            label_for_metrics=zipcode,
            output_name=zipcode,
            progress_callback=progress_callback,
        )

        result.performance_metrics.setdefault("stages", {})
        result.performance_metrics["stages"]["geocoding"] = (
            time.perf_counter() - geocode_start
        )

        return result

    @memory_efficient(cleanup_after=True)
    async def analyze_address(
        self,
        address: str,
        radius_meters: float = 120.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AnalysisResult:
        await self._initialize()

        assert self._geocoder is not None

        report_progress = self._reporter(progress_callback)

        report_progress("Geocoding address", 0.0)

        geocode_start = time.perf_counter()

        address_info = await self._geocoder.geocode_address(
            address=address,
            maptiler_api_key=self._api_key,
            radius_meters=radius_meters,
        )

        if address_info is None:
            raise ValueError(f"Could not geocode address: {address}")

        report_progress("Geocoding address", 1.0)

        safe_name = "".join(c if c.isalnum() else "_" for c in address.lower())[:80]

        result = await self._run_detection_for_area(
            area_info=address_info,
            label_for_metrics="address",
            output_name=safe_name,
            progress_callback=progress_callback,
        )

        result.zipcode = address
        result.city = address_info.city
        result.state = address_info.state
        result.center_lat = address_info.center_lat
        result.center_lng = address_info.center_lng

        result.performance_metrics.setdefault("stages", {})
        result.performance_metrics["stages"]["geocoding_address"] = (
            time.perf_counter() - geocode_start
        )

        return result


def analyze_zipcode_sync(
    zipcode: str,
    api_key: str,
    config: Optional[PipelineConfig] = None,
) -> AnalysisResult:
    async def _run() -> AnalysisResult:
        pipeline = RoofDamagePipeline(api_key=api_key, config=config)
        try:
            return await pipeline.analyze_zipcode(zipcode)
        finally:
            await pipeline.close()

    return asyncio.run(_run())
