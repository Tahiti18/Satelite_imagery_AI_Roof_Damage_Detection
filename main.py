"""
FastAPI application for AI Roof Damage Detection.
Supports zipcode analysis and exact-address analysis.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from src.pipeline import RoofDamagePipeline, PipelineConfig
from config.settings import get_settings


class AnalyzeZipcodeRequest(BaseModel):
    zipcode: str = Field(..., min_length=5, max_length=5)
    roof_confidence: float = 0.2
    damage_confidence: float = 0.25
    zoom_level: int = 17
    save_visualization: bool = False
    save_heatmap: bool = False
    save_json: bool = True
    save_geojson: bool = True


class AnalyzeAddressRequest(BaseModel):
    address: str = Field(..., min_length=5)
    radius_meters: float = 120.0
    roof_confidence: float = 0.2
    damage_confidence: float = 0.25
    zoom_level: int = 17
    save_visualization: bool = False
    save_heatmap: bool = False
    save_json: bool = True
    save_geojson: bool = True


app = FastAPI(
    title="AI Roof Damage Detection API",
    description="Zipcode and address-based roof damage detection from satellite imagery.",
    version="1.0.0",
)


def build_config(
    roof_confidence: float = 0.2,
    damage_confidence: float = 0.25,
    save_visualization: bool = False,
    save_heatmap: bool = False,
    save_json: bool = True,
    save_geojson: bool = True,
) -> PipelineConfig:
    return PipelineConfig(
        tile_size=256,
        zoom_level=17,
        max_concurrent_downloads=5,
        roof_confidence=roof_confidence,
        damage_confidence=damage_confidence,
        output_dir="/tmp/output",
        cache_dir="/tmp/tiles",
        save_visualization=save_visualization,
        save_heatmap=save_heatmap,
        save_json=save_json,
        save_geojson=save_geojson,
    )


def get_api_key() -> str:
    settings = get_settings()

    if not settings.has_maptiler_api_key:
        raise HTTPException(
            status_code=503,
            detail="MAPTILER_API_KEY is not configured",
        )

    return settings.maptiler_api_key.get_secret_value()


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "AI Roof Damage Detection API",
        "docs": "/docs",
        "health": "/health",
        "analyze_zipcode": "/analyze",
        "analyze_address": "/analyze-address",
    }


@app.get("/health")
async def health():
    settings = get_settings()

    return {
        "status": "healthy",
        "maptiler_configured": settings.has_maptiler_api_key,
    }


@app.post("/analyze")
async def analyze_zipcode(request: AnalyzeZipcodeRequest):
    api_key = get_api_key()

    config = build_config(
        roof_confidence=request.roof_confidence,
        damage_confidence=request.damage_confidence,
        save_visualization=request.save_visualization,
        save_heatmap=request.save_heatmap,
        save_json=request.save_json,
        save_geojson=request.save_geojson,
    )

    pipeline = RoofDamagePipeline(api_key=api_key, config=config)

    try:
        result = await pipeline.analyze_zipcode(request.zipcode)
        return result.to_dict()
    except Exception as e:
        logger.exception(f"Zipcode analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await pipeline.close()


@app.post("/analyze-address")
async def analyze_address(request: AnalyzeAddressRequest):
    api_key = get_api_key()

    config = build_config(
        roof_confidence=request.roof_confidence,
        damage_confidence=request.damage_confidence,
        save_visualization=request.save_visualization,
        save_heatmap=request.save_heatmap,
        save_json=request.save_json,
        save_geojson=request.save_geojson,
    )

    pipeline = RoofDamagePipeline(api_key=api_key, config=config)

    try:
        result = await pipeline.analyze_address(
            address=request.address,
            radius_meters=request.radius_meters,
        )
        return result.to_dict()
    except Exception as e:
        logger.exception(f"Address analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await pipeline.close()
