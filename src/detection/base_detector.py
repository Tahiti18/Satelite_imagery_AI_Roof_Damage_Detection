"""
Base detector class with common functionality.
Memory efficient with lazy loading and batch processing.
"""
import gc
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from loguru import logger

from ..utils.memory import memory_efficient, get_memory_manager


class BaseDetector(ABC):
    """
    Abstract base class for object detection models.
    
    Features:
    - Lazy model loading
    - Automatic device selection (GPU/CPU)
    - Memory-efficient batch processing
    - Input validation and preprocessing
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        half_precision: bool = True
    ):
        """
        Initialize detector.
        
        Args:
            model_path: Path to model weights (None for default)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('cuda', 'cpu', or None for auto)
            half_precision: Use FP16 for faster inference (GPU only)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.half_precision = half_precision
        
        # Auto-select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Lazy loaded model
        self._model = None
        self._model_loaded = False
        
        logger.info(f"{self.__class__.__name__} initialized (device: {self.device})")
    
    @property
    def model(self):
        """Lazy load model on first access."""
        if not self._model_loaded:
            self._load_model()
        return self._model
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        pass
    
    @abstractmethod
    def _postprocess(self, outputs, original_shape: Tuple[int, int]) -> List:
        """Postprocess model outputs."""
        pass
    
    def validate_image(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """
        Validate and convert image to numpy array.
        
        Args:
            image: Input image (numpy array, PIL Image, or path)
            
        Returns:
            RGB numpy array
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, PIL Image, or path, got {type(image)}")
        
        if image.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, C), got {image.ndim}D")
        
        if image.shape[2] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {image.shape[2]}")
        
        return image
    
    @memory_efficient(cleanup_after=True)
    def detect(
        self,
        image: Union[np.ndarray, Image.Image, str]
    ) -> List:
        """
        Run detection on a single image.
        
        Args:
            image: Input image
            
        Returns:
            List of detections
        """
        image = self.validate_image(image)
        original_shape = image.shape[:2]
        
        # Preprocess
        input_tensor = self._preprocess(image)
        
        # Run inference
        with torch.no_grad():
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            outputs = self.model(input_tensor)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
        
        # Postprocess
        detections = self._postprocess(outputs, original_shape)
        
        return detections
    
    @memory_efficient(cleanup_after=True)
    def detect_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        batch_size: int = 4
    ) -> List[List]:
        """
        Run detection on multiple images with batch processing.
        
        Args:
            images: List of input images
            batch_size: Number of images per batch
            
        Returns:
            List of detection lists (one per image)
        """
        results = []
        memory_manager = get_memory_manager()
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Process batch
            batch_results = [self.detect(img) for img in batch]
            results.extend(batch_results)
            
            # Memory cleanup between batches
            memory_manager.cleanup(force=False)
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size}")
        
        return results
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"{self.__class__.__name__} model unloaded")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.unload_model()

