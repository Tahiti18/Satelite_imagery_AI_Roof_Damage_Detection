"""
FastAPI application for Roof Damage Detection API.
Production-ready with security, rate limiting, and error handling.
"""
import os
import sys
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Dict
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.utils.memory import get_memory_manager
from src.pipeline import RoofDamagePipeline, PipelineConfig
from config.settings import get_settings, Settings


# ============== Security Middleware ==============

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Check rate limit
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[client_ip] = [t for t in self.requests[client_ip] if t > minute_ago]
        
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."}
            )
        
        # Record request
        self.requests[client_ip].append(now)
        
        return await call_next(request)


# ============== Pydantic Models ==============

class AnalyzeRequest(BaseModel):
    """Request model for zipcode analysis."""
    zipcode: str = Field(..., min_length=5, max_length=5, pattern=r"^\d{5}$", description="US zipcode (5 digits)")
    save_visualization: bool = Field(default=True, description="Save annotated image")
    save_heatmap: bool = Field(default=True, description="Save damage heatmap")


class AnalyzeResponse(BaseModel):
    """Response model for analysis results."""
    success: bool
    zipcode: str
    message: str
    data: Optional[dict] = None
    output_files: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    gpu_available: bool
    memory_usage_mb: float


# ============== Application Lifecycle ==============

# Global pipeline instance
_pipeline: Optional[RoofDamagePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _pipeline
    
    # Startup
    setup_logger(log_dir="./logs")
    logger.info("Starting Roof Damage Detection API...")
    
    try:
        settings = get_settings()
        
        # Check if API key is available
        if not settings.has_maptiler_api_key:
            logger.warning("MAPTILER_API_KEY not configured - image fetching disabled")
            logger.warning("Set MAPTILER_API_KEY in .env file to enable full functionality")
        else:
            api_key = settings.maptiler_api_key.get_secret_value()
            
            config = PipelineConfig(
                tile_size=settings.tile_size,
                zoom_level=settings.zoom_level,
                roof_confidence=settings.confidence_threshold,
                damage_confidence=settings.confidence_threshold,
                output_dir=settings.output_dir,
                cache_dir=settings.temp_dir + "/tiles"
            )
            
            _pipeline = RoofDamagePipeline(api_key=api_key, config=config)
            logger.info("Pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        # Don't raise - allow API to start for health checks
    
    yield
    
    # Shutdown
    if _pipeline:
        await _pipeline.close()
    get_memory_manager().cleanup(force=True)
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="AI Roof Damage Detection API",
        description="Production-ready API for detecting roof damage from satellite imagery using computer vision.",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,  # Disable docs in production
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_requests_per_minute
    )
    
    # CORS middleware - restrict in production
    allowed_origins = settings.allowed_hosts_list if settings.allowed_hosts != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    return app


# Create app instance
app = create_app()


# ============== Dependencies ==============

async def get_pipeline() -> RoofDamagePipeline:
    """Dependency to get pipeline instance."""
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Check API key configuration."
        )
    return _pipeline


# ============== Endpoints ==============

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Roof Damage Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import torch
    
    memory_manager = get_memory_manager()
    
    return HealthResponse(
        status="healthy" if _pipeline else "degraded",
        version="1.0.0",
        gpu_available=torch.cuda.is_available(),
        memory_usage_mb=round(memory_manager.get_memory_usage_mb(), 1)
    )


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_zipcode(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    pipeline: RoofDamagePipeline = Depends(get_pipeline)
):
    """
    Analyze all roofs in a zipcode for damage.
    
    - **zipcode**: US zipcode (5 digits)
    - **save_visualization**: Whether to save annotated image
    - **save_heatmap**: Whether to save damage heatmap
    
    Returns analysis results with detected roofs and damages.
    """
    try:
        logger.info(f"Analyzing zipcode: {request.zipcode}")
        
        # Update pipeline config
        pipeline.config.save_visualization = request.save_visualization
        pipeline.config.save_heatmap = request.save_heatmap
        
        # Run analysis
        result = await pipeline.analyze_zipcode(request.zipcode)
        
        # Get output files
        output_base = Path(pipeline.config.output_dir)
        output_files = {}
        
        for f in output_base.glob(f"{request.zipcode}_*"):
            file_type = f.suffix.replace(".", "")
            if "annotated" in f.name:
                output_files["visualization"] = str(f)
            elif "heatmap" in f.name:
                output_files["heatmap"] = str(f)
            elif file_type == "json":
                output_files["json"] = str(f)
            elif file_type == "geojson":
                output_files["geojson"] = str(f)
        
        return AnalyzeResponse(
            success=True,
            zipcode=request.zipcode,
            message=f"Analysis complete: {result.total_roofs} roofs, {result.roofs_with_damage} with damage",
            data=result.to_dict(),
            output_files=output_files
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/v1/analyze/{zipcode}", response_model=AnalyzeResponse)
async def analyze_zipcode_get(
    zipcode: str = Query(..., min_length=5, max_length=5, pattern=r"^\d{5}$"),
    save_visualization: bool = Query(default=True),
    save_heatmap: bool = Query(default=True),
    pipeline: RoofDamagePipeline = Depends(get_pipeline)
):
    """
    GET endpoint for zipcode analysis (for browser testing).
    """
    request = AnalyzeRequest(
        zipcode=zipcode,
        save_visualization=save_visualization,
        save_heatmap=save_heatmap
    )
    return await analyze_zipcode(request, BackgroundTasks(), pipeline)


@app.get("/api/v1/files/{filename}")
async def get_output_file(filename: str):
    """
    Download generated output file.
    """
    settings = get_settings()
    file_path = Path(settings.output_dir) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security check - ensure file is in output directory
    try:
        file_path.resolve().relative_to(Path(settings.output_dir).resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )


# ============== Error Handlers ==============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # Single worker for GPU models
    )

