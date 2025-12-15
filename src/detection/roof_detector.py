"""
Roof Detection using YOLOv8-seg.
Detects and segments roof boundaries in satellite imagery.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
from PIL import Image
from loguru import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. Install with: pip install ultralytics")

from .base_detector import BaseDetector
from ..utils.memory import memory_efficient


@dataclass
class RoofDetection:
    """Single roof detection result."""
    id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: Optional[np.ndarray] = None  # Binary mask
    polygon: Optional[List[Tuple[float, float]]] = None  # Polygon coordinates
    area_pixels: int = 0
    center: Tuple[float, float] = (0, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (JSON-serializable)."""
        result = {
            "id": self.id,
            "confidence": float(round(self.confidence, 4)),
            "bbox": {
                "x1": int(self.bbox[0]),
                "y1": int(self.bbox[1]),
                "x2": int(self.bbox[2]),
                "y2": int(self.bbox[3])
            },
            "area_pixels": int(self.area_pixels),
            "center": {"x": float(round(self.center[0], 2)), "y": float(round(self.center[1], 2))}
        }
        
        if self.polygon is not None:
            result["polygon"] = [{"x": float(round(p[0], 2)), "y": float(round(p[1], 2))} for p in self.polygon]
        
        return result
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class RoofDetector(BaseDetector):
    """
    Detects and segments roofs in satellite imagery using YOLOv8-seg.
    
    Can use:
    - Pre-trained model for general building detection
    - Custom fine-tuned model for roof-specific detection
    """
    
    # Default to YOLOv8 segmentation model
    DEFAULT_MODEL = "yolov8n-seg.pt"  # Nano for speed, use 'yolov8m-seg.pt' for accuracy
    
    # Class names for detection (building/roof)
    ROOF_CLASSES = ["building", "roof", "house"]  # Common class names
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        half_precision: bool = True,
        min_area_pixels: int = 100
    ):
        """
        Initialize roof detector.
        
        Args:
            model_path: Path to model weights (None for default)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on
            half_precision: Use FP16 for faster inference
            min_area_pixels: Minimum roof area in pixels
        """
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device,
            half_precision=half_precision
        )
        self.min_area_pixels = min_area_pixels
    
    def _load_model(self) -> None:
        """Load YOLOv8 segmentation model."""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is required. Install with: pip install ultralytics")
        
        model_path = self.model_path or self.DEFAULT_MODEL
        
        logger.info(f"Loading roof detection model: {model_path}")
        
        self._model = YOLO(model_path)
        
        # Move to device
        self._model.to(self.device)
        
        # Enable half precision on GPU
        if self.half_precision and self.device == "cuda":
            self._model.model.half()
        
        self._model_loaded = True
        logger.info(f"Roof detection model loaded on {self.device}")
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess and enhance image for better detection.
        Applies satellite image enhancement for improved visibility.
        """
        from ..utils.image_enhancement import enhance_satellite_image, normalize_satellite_image
        
        # Normalize image to uint8 [0-255]
        image = normalize_satellite_image(image)
        
        # Enhance image for better object visibility
        # These settings optimize for satellite imagery building detection
        enhanced = enhance_satellite_image(
            image,
            contrast_factor=1.4,  # Higher contrast for better building edges
            brightness_factor=1.15,  # Slightly brighter
            sharpen=True,  # Sharpen edges
            denoise=True  # Reduce noise
        )
        
        return enhanced
    
    def _postprocess(
        self,
        results,
        original_shape: Tuple[int, int]
    ) -> List[RoofDetection]:
        """
        Process YOLO results into RoofDetection objects.
        
        Args:
            results: YOLO results object
            original_shape: Original image (height, width)
            
        Returns:
            List of RoofDetection objects
        """
        detections = []
        detection_id = 0
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
            
            for i, box in enumerate(boxes):
                # Get confidence
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                
                # Get class (for filtering if using pre-trained model)
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id] if cls_id in result.names else "unknown"
                
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                # Get mask if available
                mask = None
                polygon = None
                area_pixels = (x2 - x1) * (y2 - y1)  # Default to bbox area
                
                if masks is not None and i < len(masks.data):
                    mask_tensor = masks.data[i].cpu().numpy()
                    
                    # Resize mask to original shape
                    mask_pil = Image.fromarray((mask_tensor * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize((original_shape[1], original_shape[0]), Image.NEAREST)
                    mask = np.array(mask_pil) > 127
                    
                    # Calculate actual area
                    area_pixels = int(np.sum(mask))
                    
                    # Extract polygon from mask
                    polygon = self._mask_to_polygon(mask)
                
                # Filter by minimum area
                if area_pixels < self.min_area_pixels:
                    continue
                
                # Calculate center
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                detection = RoofDetection(
                    id=detection_id,
                    confidence=conf,
                    bbox=bbox,
                    mask=mask,
                    polygon=polygon,
                    area_pixels=area_pixels,
                    center=center
                )
                
                detections.append(detection)
                detection_id += 1
        
        logger.debug(f"Detected {len(detections)} roofs")
        return detections
    
    def _mask_to_polygon(self, mask: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """
        Convert binary mask to polygon coordinates.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            List of (x, y) polygon vertices, or None
        """
        import cv2
        
        try:
            # Find contours
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            
            # Simplify polygon
            epsilon = 0.01 * cv2.arcLength(largest, True)
            simplified = cv2.approxPolyDP(largest, epsilon, True)
            
            # Convert to list of tuples
            polygon = [(float(p[0][0]), float(p[0][1])) for p in simplified]
            
            return polygon if len(polygon) >= 3 else None
            
        except Exception as e:
            logger.warning(f"Failed to extract polygon: {e}")
            return None
    
    @memory_efficient(cleanup_after=True)
    def detect(
        self,
        image: np.ndarray,
        return_masks: bool = True,
        chunk_size: int = 1280,  # Process large images in chunks
        overlap: int = 200  # Overlap between chunks to avoid missing detections at edges
    ) -> List[RoofDetection]:
        """
        Detect roofs in an image.
        For large images, splits into overlapping chunks for better detection.
        
        Args:
            image: Input image (numpy array, PIL Image, or path)
            return_masks: Whether to include segmentation masks
            chunk_size: Size of chunks for large images (YOLO works best with 640-1280px)
            overlap: Overlap between chunks in pixels
            
        Returns:
            List of RoofDetection objects
        """
        image = self.validate_image(image)
        original_shape = image.shape[:2]
        height, width = original_shape
        
        # If image is small enough, process directly
        if width <= chunk_size and height <= chunk_size:
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            detections = self._postprocess(results, original_shape)
            if not return_masks:
                for det in detections:
                    det.mask = None
            return detections
        
        # For large images, process in overlapping chunks
        logger.info(f"Large image ({width}x{height}), processing in {chunk_size}x{chunk_size} chunks")
        all_detections = []
        step = chunk_size - overlap
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                # Extract chunk with overlap
                x_end = min(x + chunk_size, width)
                y_end = min(y + chunk_size, height)
                
                # Adjust if we're at the edge
                if x_end == width:
                    x = max(0, x_end - chunk_size)
                if y_end == height:
                    y = max(0, y_end - chunk_size)
                
                chunk = image[y:y_end, x:x_end]
                
                if chunk.size == 0:
                    continue
                
                # Run detection on chunk
                results = self.model(
                    chunk,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                chunk_detections = self._postprocess(results, (y_end - y, x_end - x))
                
                # Adjust coordinates to original image space
                for det in chunk_detections:
                    # Adjust bbox coordinates
                    x1, y1, x2, y2 = det.bbox
                    det.bbox = (x1 + x, y1 + y, x2 + x, y2 + y)
                    
                    # Adjust center
                    cx, cy = det.center
                    det.center = (cx + x, cy + y)
                    
                    # Adjust polygon if present
                    if det.polygon:
                        det.polygon = [(px + x, py + y) for px, py in det.polygon]
                    
                    # For large images, skip full-size masks to save memory (use polygon instead)
                    if det.mask is not None and return_masks:
                        # Only keep mask if image is small enough (< 5M pixels)
                        # For larger images, polygon is sufficient and much more memory efficient
                        if width * height < 5_000_000:  # < 5M pixels (~2236x2236)
                            try:
                                # Create full-size mask only for smaller images
                                full_mask = np.zeros(original_shape, dtype=bool)
                                mask_h, mask_w = det.mask.shape
                                full_mask[y:y+mask_h, x:x+mask_w] = det.mask
                                det.mask = full_mask
                            except (MemoryError, ValueError):
                                det.mask = None
                                logger.debug("Skipping mask due to memory constraints")
                        else:
                            # For large images, skip mask (polygon is enough)
                            det.mask = None
                            logger.debug("Skipping full mask for large image, using polygon")
                    elif not return_masks:
                        det.mask = None
                
                all_detections.extend(chunk_detections)
        
        # Remove duplicates (detections that overlap significantly)
        if len(all_detections) > 1:
            all_detections = self._remove_duplicate_detections(all_detections)
        
        logger.info(f"Detected {len(all_detections)} roofs after chunk processing")
        
        # Optionally remove masks to save memory
        if not return_masks:
            for det in all_detections:
                det.mask = None
        
        return all_detections
    
    def _remove_duplicate_detections(self, detections: List[RoofDetection], iou_threshold: float = 0.5) -> List[RoofDetection]:
        """Remove duplicate detections from overlapping chunks."""
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept = []
        
        for det in sorted_dets:
            is_duplicate = False
            x1, y1, x2, y2 = det.bbox
            det_area = (x2 - x1) * (y2 - y1)
            
            for kept_det in kept:
                kx1, ky1, kx2, ky2 = kept_det.bbox
                kept_area = (kx2 - kx1) * (ky2 - ky1)
                
                # Calculate IoU
                inter_x1 = max(x1, kx1)
                inter_y1 = max(y1, ky1)
                inter_x2 = min(x2, kx2)
                inter_y2 = min(y2, ky2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    union_area = det_area + kept_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > iou_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                kept.append(det)
        
        return kept
    
    def detect_and_crop(
        self,
        image: np.ndarray,
        padding: int = 10
    ) -> List[Tuple[RoofDetection, np.ndarray]]:
        """
        Detect roofs and return cropped images.
        
        Args:
            image: Input image
            padding: Padding around crop in pixels
            
        Returns:
            List of (RoofDetection, cropped_image) tuples
        """
        image = self.validate_image(image)
        detections = self.detect(image, return_masks=True)
        
        results = []
        h, w = image.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Crop image
            crop = image[y1:y2, x1:x2].copy()
            
            results.append((det, crop))
        
        return results

