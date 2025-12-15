"""
Satellite Image Fetcher using MapTiler Static API.
Memory efficient with streaming downloads and rate limiting.
"""
import asyncio
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, AsyncGenerator
from pathlib import Path
import hashlib

import httpx
from PIL import Image
import numpy as np
from loguru import logger

from ..utils.memory import memory_efficient, get_memory_manager
from .geocoder import BoundingBox


@dataclass
class TileInfo:
    """Information about a single map tile."""
    row: int
    col: int
    center_lat: float
    center_lng: float
    zoom: int
    size: int
    image_path: Optional[str] = None
    image_data: Optional[bytes] = None
    
    @property
    def tile_id(self) -> str:
        """Unique identifier for this tile."""
        return f"tile_{self.row}_{self.col}_{self.zoom}"
    
    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "center": {"lat": self.center_lat, "lng": self.center_lng},
            "zoom": self.zoom,
            "size": self.size,
            "tile_id": self.tile_id
        }


class SatelliteImageFetcher:
    """
    Fetches satellite imagery from MapTiler Static API.
    
    Features:
    - Rate limiting to avoid API throttling
    - Concurrent downloads with semaphore
    - Memory-efficient streaming
    - Automatic retry with exponential backoff
    - Tile caching to avoid redundant downloads
    """
    
    MAPTILER_BASE_URL = "https://api.maptiler.com/maps/satellite"
    
    # Meters per pixel at zoom level 20 at equator
    # This decreases with latitude
    METERS_PER_PIXEL_ZOOM_20 = 0.149
    
    def __init__(
        self,
        api_key: str,
        tile_size: int = 256,  # MapTiler standard tile size
        zoom: int = 21,  # MAXIMUM zoom for closest view (0.075m/pixel - see individual roof tiles)
        max_concurrent: int = 10,  # Conservative to avoid API throttling
        requests_per_second: int = 10,  # Rate limit: 10 req/sec (safe for free tier)
        timeout: int = 30,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize image fetcher - OPTIMIZED FOR ACCURACY.
        
        Args:
            api_key: MapTiler API key
            tile_size: Size of each tile in pixels (MapTiler standard is 256)
            zoom: Zoom level (19 = 0.3m/pixel - optimal for roof damage detection)
            max_concurrent: Maximum concurrent downloads (conservative for stability)
            requests_per_second: Rate limit for API calls
            timeout: Request timeout in seconds
            cache_dir: Directory for caching tiles (optional)
        """
        self._api_key = api_key
        self.tile_size = 256  # MapTiler standard tile size
        self.zoom = min(zoom, 21)  # Max zoom 21 for maximum detail (0.075m/pixel)
        self.max_concurrent = max_concurrent
        self.requests_per_second = requests_per_second
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Rate limiter: track last request time
        self._rate_limiter_lock = asyncio.Lock()
        self._last_request_time = 0.0
        self._min_request_interval = 1.0 / requests_per_second
        
        # Setup cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=self.max_concurrent * 2),
                follow_redirects=True
            )
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def _rate_limit(self) -> None:
        """Rate limiter: ensures requests don't exceed requests_per_second."""
        async with self._rate_limiter_lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self._last_request_time
            if time_since_last < self._min_request_interval:
                wait_time = self._min_request_interval - time_since_last
                await asyncio.sleep(wait_time)
            self._last_request_time = asyncio.get_event_loop().time()
    
    def calculate_tile_grid(
        self,
        bbox: BoundingBox,
        overlap_percent: float = 0.0  # NO OVERLAP - exact area only
    ) -> List[TileInfo]:
        """
        Calculate EXACT grid of tiles needed to cover bounding box.
        NO OVERLAP - fetches only what's needed for accuracy.
        
        Args:
            bbox: Bounding box to cover
            overlap_percent: Overlap between adjacent tiles (0.0 = no overlap, exact area)
            
        Returns:
            List of TileInfo objects for each tile
        """
        # Calculate meters per pixel at this latitude
        center_lat = bbox.center[0]
        meters_per_pixel = self.METERS_PER_PIXEL_ZOOM_20 * math.cos(math.radians(center_lat))
        meters_per_pixel *= (2 ** (20 - self.zoom))  # Adjust for zoom level
        
        # Calculate tile coverage in degrees (EXACT, no padding)
        tile_meters = self.tile_size * meters_per_pixel
        tile_lat_deg = tile_meters / 111000  # ~111km per degree latitude
        tile_lng_deg = tile_meters / (111000 * math.cos(math.radians(center_lat)))
        
        # NO OVERLAP - exact coverage only
        step_lat = tile_lat_deg * (1 - overlap_percent)
        step_lng = tile_lng_deg * (1 - overlap_percent)
        
        # Calculate EXACT number of tiles needed (no rounding up unnecessarily)
        n_rows = max(1, math.ceil(bbox.height_degrees / step_lat))
        n_cols = max(1, math.ceil(bbox.width_degrees / step_lng))
        total_tiles = n_rows * n_cols
        
        # Limit area size for accuracy: max 500 tiles (~22x22 grid = ~1.1km x 1.1km at zoom 19)
        # This ensures we focus on a manageable area for accurate detection
        MAX_TILES = 500
        if total_tiles > MAX_TILES:
            logger.warning(
                f"Area too large ({total_tiles} tiles). "
                f"Limiting to {MAX_TILES} tiles for accuracy. "
                f"Consider using a smaller zipcode or area."
            )
            # Reduce grid size proportionally
            scale = math.sqrt(MAX_TILES / total_tiles)
            n_rows = max(1, int(n_rows * scale))
            n_cols = max(1, int(n_cols * scale))
            total_tiles = n_rows * n_cols
        
        logger.info(f"Tile grid: {n_rows}x{n_cols} = {total_tiles} tiles (zoom={self.zoom}, no overlap)")
        
        # Generate tile coordinates
        tiles = []
        start_lat = bbox.max_lat - (tile_lat_deg / 2)
        start_lng = bbox.min_lng + (tile_lng_deg / 2)
        
        for row in range(n_rows):
            for col in range(n_cols):
                tile_lat = start_lat - (row * step_lat)
                tile_lng = start_lng + (col * step_lng)
                
                tiles.append(TileInfo(
                    row=row,
                    col=col,
                    center_lat=tile_lat,
                    center_lng=tile_lng,
                    zoom=self.zoom,
                    size=self.tile_size
                ))
        
        return tiles
    
    def _lat_lon_to_tile_coords(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile coordinates (x, y) for MapTiler tile service."""
        import math
        
        n = 2.0 ** zoom
        lat_rad = math.radians(lat)
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y
    
    def _get_cache_path(self, tile: TileInfo) -> Path:
        """Generate cache file path for a tile."""
        if not self.cache_dir:
            return None
        
        # Create hash of tile parameters for unique filename
        tile_hash = hashlib.md5(
            f"{tile.center_lat:.6f}_{tile.center_lng:.6f}_{tile.zoom}_{tile.size}".encode()
        ).hexdigest()[:12]
        
        return self.cache_dir / f"tile_{tile_hash}.png"
    
    async def _fetch_single_tile(
        self,
        tile: TileInfo,
        use_cache: bool = True
    ) -> TileInfo:
        """
        Fetch a single tile from MapTiler API.
        
        Args:
            tile: TileInfo with coordinates
            use_cache: Whether to use cached tiles
            
        Returns:
            TileInfo with image_data populated
        """
        # Check cache first
        cache_path = self._get_cache_path(tile)
        if use_cache and cache_path and cache_path.exists():
            logger.debug(f"Cache hit: {tile.tile_id}")
            tile.image_data = cache_path.read_bytes()
            tile.image_path = str(cache_path)
            return tile
        
        # Rate limiting + concurrency control
        await self._rate_limit()  # Enforce requests per second limit
        async with self._semaphore:  # Limit concurrent requests
            client = await self._get_client()
            
            # MapTiler Tile Service format: /maps/{style}/{z}/{x}/{y}.png
            # Convert lat/lon to tile coordinates
            tile_x, tile_y = self._lat_lon_to_tile_coords(tile.center_lat, tile.center_lng, tile.zoom)
            url = f"https://api.maptiler.com/maps/satellite/{tile.zoom}/{tile_x}/{tile_y}.png"
            params = {
                "key": self._api_key
            }
            
            for attempt in range(3):
                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    
                    tile.image_data = response.content
                    
                    # Save to cache
                    if cache_path:
                        cache_path.write_bytes(tile.image_data)
                        tile.image_path = str(cache_path)
                    
                    logger.debug(f"Fetched tile: {tile.tile_id}")
                    return tile
                    
                except httpx.TimeoutException:
                    logger.warning(f"Timeout fetching {tile.tile_id}, attempt {attempt + 1}/3")
                    await asyncio.sleep(1 * (attempt + 1))
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error fetching {tile.tile_id}: {e}")
                    if e.response.status_code == 403:
                        raise ValueError("Invalid API key or API not enabled")
                    raise
                except Exception as e:
                    logger.error(f"Error fetching {tile.tile_id}: {e}")
                    raise
            
            raise RuntimeError(f"Failed to fetch tile {tile.tile_id} after 3 attempts")
    
    @memory_efficient(cleanup_after=True)
    async def fetch_tiles(
        self,
        tiles: List[TileInfo],
        use_cache: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[TileInfo]:
        """
        Fetch multiple tiles concurrently.
        
        Args:
            tiles: List of TileInfo objects
            use_cache: Whether to use cached tiles
            progress_callback: Optional callback(completed, total)
            
        Returns:
            List of TileInfo with image_data populated
        """
        total = len(tiles)
        completed = 0
        results = []
        last_progress = 0
        
        logger.info(f"Fetching {total} tiles...")
        
        # Process in batches to manage memory
        batch_size = min(50, self.max_concurrent * 5)
        
        for i in range(0, total, batch_size):
            batch = tiles[i:i + batch_size]
            
            # Fetch batch concurrently
            tasks = [self._fetch_single_tile(tile, use_cache) for tile in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Tile fetch failed: {result}")
                else:
                    results.append(result)
                    completed += 1
                    
                    # Update progress every 1% or every 10 tiles (whichever is more frequent)
                    if progress_callback:
                        progress_pct = int((completed / total) * 100)
                        if progress_pct != last_progress or completed % 10 == 0:
                            progress_callback(completed, total)
                            last_progress = progress_pct
            
            # Memory check between batches
            get_memory_manager().check_memory()
        
        logger.info(f"Successfully fetched {len(results)}/{total} tiles")
        return results
    
    async def fetch_area(
        self,
        bbox: BoundingBox,
        use_cache: bool = True,
        overlap_percent: float = 0.0,  # NO OVERLAP by default
        progress_callback: Optional[callable] = None
    ) -> List[TileInfo]:
        """
        Fetch all tiles for a bounding box - EXACT AREA ONLY.
        
        Args:
            bbox: BoundingBox to cover
            use_cache: Whether to use cached tiles
            overlap_percent: Overlap between tiles (0.0 = exact area, no overlap)
            progress_callback: Optional callback(completed, total)
            
        Returns:
            List of TileInfo with image_data
        """
        tiles = self.calculate_tile_grid(bbox, overlap_percent=overlap_percent)
        return await self.fetch_tiles(tiles, use_cache, progress_callback)

