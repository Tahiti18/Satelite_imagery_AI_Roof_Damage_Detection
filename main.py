"""
Railway-compatible FastAPI entrypoint for the roof damage detection system.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline import RoofDamagePipeline, PipelineConfig


app = FastAPI(
    title="AI Roof Damage Detection API",
    description="Zipcode-based roof damage detection from satellite imagery.",
    version="1.0.0",
)


class AnalyzeZipcodeRequest(BaseModel):
    zipcode: str = Field(..., description="US ZIP code to analyze")
    roof_confidence: Optional[float] = Field(default=None)
    damage_confidence: Optional[float] = Field(default=None)
    zoom_level: Optional[int] = Field(default=None)
    save_visualization: bool = True
    save_heatmap: bool = True
    save_json: bool = True
    save_geojson: bool = True


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "AI Roof Damage Detection API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "maptiler_configured": bool(os.getenv("MAPTILER_API_KEY")),
    }


@app.post("/analyze")
async def analyze_zipcode(request: AnalyzeZipcodeRequest):
    api_key = os.getenv("MAPTILER_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="MAPTILER_API_KEY is not configured in Railway variables.",
        )

    config = PipelineConfig(
        roof_confidence=request.roof_confidence
        if request.roof_confidence is not None
        else 0.2,
        damage_confidence=request.damage_confidence
        if request.damage_confidence is not None
        else 0.25,
        zoom_level=request.zoom_level if request.zoom_level is not None else 21,
        save_visualization=request.save_visualization,
        save_heatmap=request.save_heatmap,
        save_json=request.save_json,
        save_geojson=request.save_geojson,
    )

    pipeline = RoofDamagePipeline(api_key=api_key, config=config)

    try:
        result = await pipeline.analyze_zipcode(request.zipcode)
        return result.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await pipeline.close()
