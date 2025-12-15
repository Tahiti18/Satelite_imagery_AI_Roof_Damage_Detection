"""
Zipcode Geocoder using US Census Bureau API.
Free, no API key required.
Memory efficient with connection pooling.
"""
import asyncio
from dataclasses import dataclass
from typing import Optional, Tuple
from functools import lru_cache

import httpx
from loguru import logger

from ..utils.memory import memory_efficient


@dataclass(frozen=True)
class BoundingBox:
    """Immutable bounding box for geographic area."""
    min_lat: float
    max_lat: float
    min_lng: float
    max_lng: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return (
            (self.min_lat + self.max_lat) / 2,
            (self.min_lng + self.max_lng) / 2
        )
    
    @property
    def width_degrees(self) -> float:
        """Width in degrees longitude."""
        return abs(self.max_lng - self.min_lng)
    
    @property
    def height_degrees(self) -> float:
        """Height in degrees latitude."""
        return abs(self.max_lat - self.min_lat)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "min_lng": self.min_lng,
            "max_lng": self.max_lng,
            "center": {"lat": self.center[0], "lng": self.center[1]}
        }


@dataclass(frozen=True)
class ZipcodeInfo:
    """Immutable zipcode information."""
    zipcode: str
    center_lat: float
    center_lng: float
    bounding_box: BoundingBox
    state: Optional[str] = None
    city: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "zipcode": self.zipcode,
            "center": {"lat": self.center_lat, "lng": self.center_lng},
            "bounding_box": self.bounding_box.to_dict(),
            "state": self.state,
            "city": self.city
        }


class ZipcodeGeocoder:
    """
    Geocoder for US zipcodes using Census Bureau API.
    
    Features:
    - Free API, no key required
    - Returns bounding box for coverage calculation
    - Connection pooling for efficiency
    - Caching for repeated lookups
    """
    
    # US Census Bureau Geocoder API
    CENSUS_API_URL = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
    
    # Fallback: Zippopotam.us API (simpler, but no bounding box)
    ZIPPOPOTAM_API_URL = "https://api.zippopotam.us/us"
    
    # Approximate zipcode size in degrees (for bounding box estimation)
    # Average US zipcode is about 0.1-0.2 degrees in each direction
    DEFAULT_ZIPCODE_RADIUS_DEG = 0.05  # ~5.5km radius
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialize geocoder.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._cache: dict = {}
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                follow_redirects=True
            )
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        self._cache.clear()
    
    @memory_efficient(cleanup_after=False)
    async def geocode_zipcode(self, zipcode: str) -> Optional[ZipcodeInfo]:
        """
        Geocode a US zipcode to coordinates and bounding box.
        
        Args:
            zipcode: US zipcode (5 digits)
            
        Returns:
            ZipcodeInfo with center and bounding box, or None if not found
        """
        # Validate zipcode format
        zipcode = zipcode.strip()
        if not zipcode.isdigit() or len(zipcode) != 5:
            logger.error(f"Invalid zipcode format: {zipcode}")
            return None
        
        # Check cache
        if zipcode in self._cache:
            logger.debug(f"Cache hit for zipcode: {zipcode}")
            return self._cache[zipcode]
        
        # Try primary method first
        result = await self._geocode_via_zippopotam(zipcode)
        
        if result is None:
            logger.warning(f"Failed to geocode zipcode: {zipcode}")
            return None
        
        # Cache result
        self._cache[zipcode] = result
        logger.info(f"Geocoded {zipcode}: center=({result.center_lat:.4f}, {result.center_lng:.4f})")
        
        return result
    
    async def _geocode_via_zippopotam(self, zipcode: str) -> Optional[ZipcodeInfo]:
        """
        Geocode using Zippopotam.us API (simple and reliable).
        
        Returns center point and estimates bounding box.
        """
        client = await self._get_client()
        
        for attempt in range(self.max_retries):
            try:
                url = f"{self.ZIPPOPOTAM_API_URL}/{zipcode}"
                response = await client.get(url)
                
                if response.status_code == 404:
                    logger.warning(f"Zipcode not found: {zipcode}")
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                # Extract coordinates
                if "places" not in data or len(data["places"]) == 0:
                    return None
                
                place = data["places"][0]
                lat = float(place["latitude"])
                lng = float(place["longitude"])
                state = place.get("state abbreviation")
                city = place.get("place name")
                
                # Estimate bounding box based on typical zipcode size
                # Urban zipcodes are smaller, rural are larger
                # We use a conservative estimate
                bbox = self._estimate_bounding_box(lat, lng, zipcode)
                
                return ZipcodeInfo(
                    zipcode=zipcode,
                    center_lat=lat,
                    center_lng=lng,
                    bounding_box=bbox,
                    state=state,
                    city=city
                )
                
            except httpx.TimeoutException:
                logger.warning(f"Timeout geocoding {zipcode}, attempt {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error geocoding {zipcode}: {e}")
                return None
            except Exception as e:
                logger.error(f"Error geocoding {zipcode}: {e}")
                return None
        
        return None
    
    def _estimate_bounding_box(
        self, 
        center_lat: float, 
        center_lng: float, 
        zipcode: str
    ) -> BoundingBox:
        """
        Estimate bounding box for a zipcode.
        
        Uses heuristics based on zipcode patterns:
        - Urban zipcodes (starts with 1,2,9) tend to be smaller
        - Rural zipcodes tend to be larger
        """
        # Determine radius based on zipcode characteristics
        first_digit = int(zipcode[0])
        
        # Rough heuristic: Northeast (0,1,2) and West Coast (9) tend to be denser
        if first_digit in [0, 1, 2, 9]:
            radius_deg = 0.03  # ~3.3km - smaller for urban areas
        elif first_digit in [3, 4, 5, 6]:
            radius_deg = 0.05  # ~5.5km - medium for suburban
        else:  # 7, 8
            radius_deg = 0.08  # ~8.8km - larger for rural/western areas
        
        # Adjust for latitude (longitude degrees are smaller near poles)
        lng_adjustment = 1 / max(0.1, abs(center_lat) / 90)
        
        return BoundingBox(
            min_lat=center_lat - radius_deg,
            max_lat=center_lat + radius_deg,
            min_lng=center_lng - (radius_deg * lng_adjustment),
            max_lng=center_lng + (radius_deg * lng_adjustment)
        )
    
    def clear_cache(self) -> None:
        """Clear the geocoding cache."""
        self._cache.clear()
        logger.debug("Geocoder cache cleared")


# Synchronous wrapper for convenience
def geocode_zipcode_sync(zipcode: str, timeout: int = 30) -> Optional[ZipcodeInfo]:
    """
    Synchronous wrapper for geocoding.
    
    Args:
        zipcode: US zipcode (5 digits)
        timeout: Request timeout in seconds
        
    Returns:
        ZipcodeInfo or None
    """
    async def _run():
        geocoder = ZipcodeGeocoder(timeout=timeout)
        try:
            return await geocoder.geocode_zipcode(zipcode)
        finally:
            await geocoder.close()
    
    return asyncio.run(_run())

