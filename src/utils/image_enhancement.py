"""
Image enhancement for satellite imagery.
Improves visibility of buildings/roofs for better detection accuracy.
"""
import cv2
import numpy as np
from typing import Tuple
from loguru import logger


def enhance_satellite_image(
    image: np.ndarray,
    contrast_factor: float = 1.3,
    brightness_factor: float = 1.1,
    sharpen: bool = True,
    denoise: bool = True
) -> np.ndarray:
    """
    Enhance satellite image for better object visibility.
    
    Args:
        image: Input RGB image (H, W, 3) uint8
        contrast_factor: Contrast enhancement (1.0 = no change, >1.0 = more contrast)
        brightness_factor: Brightness enhancement (1.0 = no change, >1.0 = brighter)
        sharpen: Apply sharpening filter
        denoise: Apply denoising
        
    Returns:
        Enhanced RGB image (H, W, 3) uint8
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    enhanced = image.copy()
    
    # 1. Denoise (reduce noise while preserving edges)
    if denoise:
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)
        logger.debug("Applied denoising")
    
    # 2. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This is better than global contrast for satellite images
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Adjust contrast further if needed
    if contrast_factor != 1.0:
        l = np.clip(l.astype(np.float32) * contrast_factor, 0, 255).astype(np.uint8)
    
    # Merge channels back
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    logger.debug(f"Applied contrast enhancement (factor: {contrast_factor})")
    
    # 3. Adjust brightness
    if brightness_factor != 1.0:
        enhanced = np.clip(enhanced.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        logger.debug(f"Applied brightness enhancement (factor: {brightness_factor})")
    
    # 4. Sharpen (enhance edges for better detection)
    if sharpen:
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * 0.5  # Reduced strength
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        logger.debug("Applied sharpening")
    
    return enhanced


def normalize_satellite_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize satellite image to standard range.
    
    Args:
        image: Input image (any dtype/range)
        
    Returns:
        Normalized RGB image (H, W, 3) uint8 [0-255]
    """
    if image.dtype == np.uint8:
        return image
    
    # Normalize to 0-255 range
    if image.max() > 1.0:
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
    else:
        image = image * 255
    
    return np.clip(image, 0, 255).astype(np.uint8)

