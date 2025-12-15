"""
Secure configuration management.
All sensitive data loaded from environment variables.
No hardcoded secrets.
"""
import os
import secrets
from typing import Optional
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr, field_validator


class Settings(BaseSettings):
    """Application settings with secure defaults."""
    
    # API Keys (loaded from environment) - Optional for testing without API
    maptiler_api_key: Optional[SecretStr] = Field(
        default=None, 
        description="MapTiler API Key"
    )
    
    # Security settings
    secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Application secret key for signing"
    )
    allowed_hosts: str = Field(default="*", description="Comma-separated allowed hosts")
    
    # Application settings
    app_name: str = "AI Roof Damage Detection"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Image fetching settings
    max_tiles_per_request: int = Field(default=100, ge=1, le=500)
    tile_size: int = Field(default=640, ge=256, le=2048)
    zoom_level: int = Field(default=20, ge=15, le=21)
    request_timeout: int = Field(default=30, ge=5, le=120)
    max_concurrent_requests: int = Field(default=10, ge=1, le=50)
    
    # Processing settings
    batch_size: int = Field(default=4, ge=1, le=16)
    confidence_threshold: float = Field(default=0.5, ge=0.1, le=0.99)
    iou_threshold: float = Field(default=0.45, ge=0.1, le=0.9)
    max_image_dimension: int = Field(default=4096, ge=1024, le=16384)
    
    # Memory management
    max_memory_mb: int = Field(default=4096, ge=512, le=32768)
    cleanup_interval_sec: int = Field(default=60, ge=10, le=600)
    
    # Storage paths
    data_dir: str = "./data"
    models_dir: str = "./models"
    output_dir: str = "./output"
    temp_dir: str = "./temp"
    
    # Model paths
    roof_model_path: str = "./models/roof_detector.pt"
    damage_model_path: str = "./models/damage_detector.pt"
    
    # Rate limiting
    rate_limit_requests_per_minute: int = Field(default=60, ge=10, le=1000)
    
    @field_validator('allowed_hosts')
    @classmethod
    def parse_allowed_hosts(cls, v: str) -> str:
        """Validate allowed hosts format."""
        return v.strip()
    
    @property
    def has_maptiler_api_key(self) -> bool:
        """Check if MapTiler API key is configured."""
        return (
            self.maptiler_api_key is not None 
            and len(self.maptiler_api_key.get_secret_value()) > 0
        )
    
    @property
    def allowed_hosts_list(self) -> list:
        """Get allowed hosts as list."""
        return [h.strip() for h in self.allowed_hosts.split(",") if h.strip()]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses LRU cache for performance - settings loaded once.
    """
    return Settings()


def clear_settings_cache() -> None:
    """Clear settings cache (useful for testing)."""
    get_settings.cache_clear()


def validate_settings(require_api_key: bool = False) -> dict:
    """
    Validate settings and return status.
    
    Args:
        require_api_key: If True, fail if Google API key is missing
        
    Returns:
        dict with validation status and warnings
    """
    result = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    try:
        settings = get_settings()
        
        # Check API key
        if not settings.has_maptiler_api_key:
            msg = "MAPTILER_API_KEY not set - image fetching will not work"
            if require_api_key:
                result["errors"].append(msg)
                result["valid"] = False
            else:
                result["warnings"].append(msg)
        
        # Check directories exist or can be created
        for dir_path in [settings.data_dir, settings.models_dir, settings.output_dir, settings.temp_dir]:
            path = Path(dir_path)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    result["warnings"].append(f"Cannot create directory {dir_path}: {e}")
        
        # Memory settings validation
        if settings.max_memory_mb < 1024:
            result["warnings"].append(f"Low memory limit ({settings.max_memory_mb}MB) may cause issues")
        
    except Exception as e:
        result["errors"].append(f"Configuration error: {e}")
        result["valid"] = False
    
    return result

