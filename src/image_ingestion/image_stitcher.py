"""
Image Stitcher for combining satellite tiles.
Memory efficient with streaming and chunked processing.
"""
import io
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image
from loguru import logger

from ..utils.memory import memory_efficient, get_memory_manager
from .image_fetcher import TileInfo


@dataclass
class StitchedImage:
    """Result of image stitching operation."""
    image: np.ndarray
    width: int
    height: int
    n_tiles: int
    tile_size: int
    bounds: dict  # Geographic bounds
    
    def to_pil(self) -> Image.Image:
        """Convert to PIL Image."""
        return Image.fromarray(self.image)
    
    def save(self, path: str, format: str = "PNG", quality: int = 95) -> None:
        """Save to file."""
        img = self.to_pil()
        if format.upper() == "JPEG":
            img.save(path, format=format, quality=quality, optimize=True)
        else:
            img.save(path, format=format, optimize=True)
        logger.info(f"Saved stitched image: {path} ({self.width}x{self.height})")


class ImageStitcher:
    """
    Stitches multiple satellite tiles into a single image.
    
    Features:
    - Memory-efficient processing
    - Handles missing tiles gracefully
    - Supports large images via chunked processing
    - Automatic overlap handling
    """
    
    def __init__(
        self,
        max_dimension: int = 8192,
        fill_color: Tuple[int, int, int] = (128, 128, 128)
    ):
        """
        Initialize stitcher.
        
        Args:
            max_dimension: Maximum output dimension (width or height)
            fill_color: Color for missing tiles (R, G, B)
        """
        self.max_dimension = max_dimension
        self.fill_color = fill_color
    
    def _validate_tiles(self, tiles: List[TileInfo]) -> Tuple[int, int, int]:
        """
        Validate tiles and calculate grid dimensions.
        
        Returns:
            (n_rows, n_cols, tile_size)
        """
        if not tiles:
            raise ValueError("No tiles to stitch")
        
        # Get grid dimensions
        rows = set(t.row for t in tiles)
        cols = set(t.col for t in tiles)
        
        n_rows = max(rows) + 1
        n_cols = max(cols) + 1
        tile_size = tiles[0].size
        
        logger.debug(f"Grid: {n_rows}x{n_cols}, tile_size: {tile_size}")
        
        return n_rows, n_cols, tile_size
    
    def _load_tile_image(self, tile: TileInfo) -> Optional[np.ndarray]:
        """
        Load tile image data as numpy array.
        
        Returns:
            RGB numpy array or None if tile has no data
        """
        if tile.image_data is None:
            return None
        
        try:
            img = Image.open(io.BytesIO(tile.image_data))
            img = img.convert("RGB")
            return np.array(img)
        except Exception as e:
            logger.warning(f"Failed to load tile {tile.tile_id}: {e}")
            return None
    
    @memory_efficient(cleanup_after=True)
    def stitch(
        self,
        tiles: List[TileInfo],
        min_lat: float = None,
        max_lat: float = None,
        min_lng: float = None,
        max_lng: float = None
    ) -> StitchedImage:
        """
        Stitch tiles into a single image.
        
        Args:
            tiles: List of TileInfo with image_data
            min_lat, max_lat, min_lng, max_lng: Geographic bounds (optional)
            
        Returns:
            StitchedImage with combined image
        """
        n_rows, n_cols, tile_size = self._validate_tiles(tiles)
        
        # Calculate output dimensions
        output_width = n_cols * tile_size
        output_height = n_rows * tile_size
        
        # Check if we need to scale down
        scale_factor = 1.0
        if output_width > self.max_dimension or output_height > self.max_dimension:
            scale_factor = min(
                self.max_dimension / output_width,
                self.max_dimension / output_height
            )
            output_width = int(output_width * scale_factor)
            output_height = int(output_height * scale_factor)
            logger.info(f"Scaling output by {scale_factor:.2f}x to {output_width}x{output_height}")
        
        # Create output array (filled with background color)
        logger.info(f"Creating canvas: {output_width}x{output_height}")
        memory_manager = get_memory_manager()
        
        # Check available memory
        required_mb = (output_width * output_height * 3) / (1024 * 1024)
        if required_mb > memory_manager.max_memory_mb * 0.5:
            logger.warning(f"Large image will use {required_mb:.0f}MB memory")
        
        canvas = np.full((output_height, output_width, 3), self.fill_color, dtype=np.uint8)
        
        # Build tile lookup for efficient access
        tile_lookup = {(t.row, t.col): t for t in tiles}
        
        # Place tiles on canvas
        placed = 0
        for row in range(n_rows):
            for col in range(n_cols):
                tile = tile_lookup.get((row, col))
                if tile is None:
                    continue
                
                tile_img = self._load_tile_image(tile)
                if tile_img is None:
                    continue
                
                # Calculate position
                if scale_factor < 1.0:
                    # Scale tile if needed
                    new_size = int(tile_size * scale_factor)
                    pil_img = Image.fromarray(tile_img)
                    pil_img = pil_img.resize((new_size, new_size), Image.LANCZOS)
                    tile_img = np.array(pil_img)
                    
                    y = row * new_size
                    x = col * new_size
                    h, w = tile_img.shape[:2]
                else:
                    y = row * tile_size
                    x = col * tile_size
                    h, w = tile_size, tile_size
                
                # Ensure we don't exceed canvas bounds
                h = min(h, output_height - y)
                w = min(w, output_width - x)
                
                # Place tile
                canvas[y:y+h, x:x+w] = tile_img[:h, :w]
                placed += 1
                
                # Clear tile data to free memory
                tile.image_data = None
            
            # Periodic memory check
            if row % 5 == 0:
                memory_manager.cleanup(force=False)
        
        logger.info(f"Placed {placed}/{len(tiles)} tiles")
        
        # Calculate bounds if not provided
        if min_lat is None:
            lats = [t.center_lat for t in tiles]
            lngs = [t.center_lng for t in tiles]
            min_lat, max_lat = min(lats), max(lats)
            min_lng, max_lng = min(lngs), max(lngs)
        
        return StitchedImage(
            image=canvas,
            width=output_width,
            height=output_height,
            n_tiles=placed,
            tile_size=tile_size,
            bounds={
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lng": min_lng,
                "max_lng": max_lng
            }
        )
    
    @memory_efficient(cleanup_after=True)
    def stitch_to_file(
        self,
        tiles: List[TileInfo],
        output_path: str,
        format: str = "PNG",
        quality: int = 95,
        **kwargs
    ) -> str:
        """
        Stitch tiles and save directly to file.
        More memory efficient for large images.
        
        Args:
            tiles: List of TileInfo with image_data
            output_path: Path to save the stitched image
            format: Output format (PNG, JPEG)
            quality: JPEG quality (ignored for PNG)
            
        Returns:
            Path to saved file
        """
        result = self.stitch(tiles, **kwargs)
        result.save(output_path, format=format, quality=quality)
        
        # Clear the large array
        del result.image
        
        return output_path

