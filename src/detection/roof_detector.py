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

YOLO = None
YOLO_AVAILABLE = False
YOLO_IMPORT_ERROR = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as exc:
    YOLO_IMPORT_ERROR = repr(exc)
    logger.warning(f"ultralytics import failed: {YOLO_IMPORT_ERROR}")

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
    
    DEFAULT_MODEL = "yolov8n-seg.pt"
    ROOF_CLASSES = ["building", "roof", "house"]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        half_precision: bool = True,
        min_area_pixels: int = 100
    ):
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
            raise ImportError(
                f"ultralytics could not be imported. Actual error: {YOLO_IMPORT_ERROR}"
            )
        
        model_path = self.model_path or self.DEFAULT_MODEL
        
        logger.info(f"Loading roof detection model: {model_path}")
        
        self._model = YOLO(model_path)
        
        self._model.to(self.device)
        
        if self.half_precision and self.device == "cuda":
            self._model.model.half()
        
        self._model_loaded = True
        logger.info(f"Roof detection model loaded on {self.device}")
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        from ..utils.image_enhancement import enhance_satellite_image, normalize_satellite_image
        
        image = normalize_satellite_image(image)
        
        enhanced = enhance_satellite_image(
            image,
            contrast_factor=1.4,
            brightness_factor=1.15,
            sharpen=True,
            denoise=True
        )
        
        return enhanced
    
    def _postprocess(
        self,
        results,
        original_shape: Tuple[int, int]
    ) -> List[RoofDetection]:
        detections = []
        detection_id = 0
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
            
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id] if cls_id in result.names else "unknown"
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                mask = None
                polygon = None
                area_pixels = (x2 - x1) * (y2 - y1)
                
                if masks is not None and i < len(masks.data):
                    mask_tensor = masks.data[i].cpu().numpy()
                    
                    mask_pil = Image.fromarray((mask_tensor * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize((original_shape[1], original_shape[0]), Image.NEAREST)
                    mask = np.array(mask_pil) > 127
                    
                    area_pixels = int(np.sum(mask))
                    polygon = self._mask_to_polygon(mask)
                
                if area_pixels < self.min_area_pixels:
                    continue
                
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
        import cv2
        
        try:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(largest, True)
            simplified = cv2.approxPolyDP(largest, epsilon, True)
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
        chunk_size: int = 1280,
        overlap: int = 200
    ) -> List[RoofDetection]:
        image = self.validate_image(image)
        original_shape = image.shape[:2]
        height, width = original_shape
        
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
        
        logger.info(f"Large image ({width}x{height}), processing in {chunk_size}x{chunk_size} chunks")
        all_detections = []
        step = chunk_size - overlap
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                x_end = min(x + chunk_size, width)
                y_end = min(y + chunk_size, height)
                
                if x_end == width:
                    x = max(0, x_end - chunk_size)
                if y_end == height:
                    y = max(0, y_end - chunk_size)
                
                chunk = image[y:y_end, x:x_end]
                
                if chunk.size == 0:
                    continue
                
                results = self.model(
                    chunk,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                chunk_detections = self._postprocess(results, (y_end - y, x_end - x))
                
                for det in chunk_detections:
                    x1, y1, x2, y2 = det.bbox
                    det.bbox = (x1 + x, y1 + y, x2 + x, y2 + y)
                    
                    cx, cy = det.center
                    det.center = (cx + x, cy + y)
                    
                    if det.polygon:
                        det.polygon = [(px + x, py + y) for px, py in det.polygon]
                    
                    if det.mask is not None and return_masks:
                        if width * height < 5_000_000:
                            try:
                                full_mask = np.zeros(original_shape, dtype=bool)
                                mask_h, mask_w = det.mask.shape
                                full_mask[y:y+mask_h, x:x+mask_w] = det.mask
                                det.mask = full_mask
                            except (MemoryError, ValueError):
                                det.mask = None
                                logger.debug("Skipping mask due to memory constraints")
                        else:
                            det.mask = None
                            logger.debug("Skipping full mask for large image, using polygon")
                    elif not return_masks:
                        det.mask = None
                
                all_detections.extend(chunk_detections)
        
        if len(all_detections) > 1:
            all_detections = self._remove_duplicate_detections(all_detections)
        
        logger.info(f"Detected {len(all_detections)} roofs after chunk processing")
        
        if not return_masks:
            for det in all_detections:
                det.mask = None
        
        return all_detections
    
    def _remove_duplicate_detections(self, detections: List[RoofDetection], iou_threshold: float = 0.5) -> List[RoofDetection]:
        if not detections:
            return []
        
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept = []
        
        for det in sorted_dets:
            is_duplicate = False
            x1, y1, x2, y2 = det.bbox
            det_area = (x2 - x1) * (y2 - y1)
            
            for kept_det in kept:
                kx1, ky1, kx2, ky2 = kept_det.bbox
                kept_area = (kx2 - kx1) * (ky2 - ky1)
                
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
        image = self.validate_image(image)
        detections = self.detect(image, return_masks=True)
        
        results = []
        h, w = image.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            crop = image[y1:y2, x1:x2].copy()
            results.append((det, crop))
        
        return results
