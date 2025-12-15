"""
Damage Detection using YOLOv8-seg.
Detects and segments damage within roof regions.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np
import torch
from PIL import Image
from loguru import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from .base_detector import BaseDetector
from .roof_detector import RoofDetection
from ..utils.memory import memory_efficient


class DamageSeverity(Enum):
    """Damage severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DamageType(Enum):
    """Types of roof damage."""
    HAIL = "hail_damage"
    MISSING_SHINGLES = "missing_shingles"
    CRACKS = "cracks"
    BLISTERS = "blisters"
    PONDING = "ponding"
    WARPING = "warping"
    FLASHING = "flashing_damage"
    SOFT_SPOTS = "soft_spots"
    MEMBRANE = "membrane_damage"
    UNKNOWN = "unknown"


@dataclass
class DamageDetection:
    """Single damage detection result."""
    id: int
    damage_type: DamageType
    confidence: float
    severity: DamageSeverity
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: Optional[np.ndarray] = None
    polygon: Optional[List[Tuple[float, float]]] = None
    area_pixels: int = 0
    center: Tuple[float, float] = (0, 0)
    roof_id: Optional[int] = None  # Associated roof
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (JSON-serializable)."""
        result = {
            "id": self.id,
            "damage_type": self.damage_type.value,
            "confidence": float(round(self.confidence, 4)),
            "severity": self.severity.value,
            "bbox": {
                "x1": int(self.bbox[0]),
                "y1": int(self.bbox[1]),
                "x2": int(self.bbox[2]),
                "y2": int(self.bbox[3])
            },
            "area_pixels": int(self.area_pixels),
            "center": {"x": float(round(self.center[0], 2)), "y": float(round(self.center[1], 2))}
        }
        
        if self.roof_id is not None:
            result["roof_id"] = int(self.roof_id)
        
        if self.polygon is not None:
            result["polygon"] = [{"x": float(round(p[0], 2)), "y": float(round(p[1], 2))} for p in self.polygon]
        
        return result


class DamageDetector(BaseDetector):
    """
    Detects and classifies damage in roof images using YOLOv8-seg.
    
    Damage types detected:
    - Hail damage
    - Missing shingles
    - Cracks and blisters
    - Ponding (water accumulation)
    - Warping/buckling
    - Flashing damage
    - Soft spots
    - Membrane damage
    """
    
    # Default model (use custom trained for better accuracy)
    DEFAULT_MODEL = "yolov8n-seg.pt"
    
    # Damage class mapping (adjust based on your trained model)
    DAMAGE_CLASS_MAP = {
        0: DamageType.HAIL,
        1: DamageType.MISSING_SHINGLES,
        2: DamageType.CRACKS,
        3: DamageType.BLISTERS,
        4: DamageType.PONDING,
        5: DamageType.WARPING,
        6: DamageType.FLASHING,
        7: DamageType.SOFT_SPOTS,
        8: DamageType.MEMBRANE
    }
    
    # Severity thresholds based on damage area percentage
    SEVERITY_THRESHOLDS = {
        DamageSeverity.LOW: 0.01,      # < 1% of roof area
        DamageSeverity.MEDIUM: 0.05,   # 1-5% of roof area
        DamageSeverity.HIGH: 0.15,     # 5-15% of roof area
        DamageSeverity.CRITICAL: 1.0   # > 15% of roof area
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        half_precision: bool = True,
        min_area_pixels: int = 25
    ):
        """
        Initialize damage detector.
        
        Args:
            model_path: Path to custom trained damage model
            confidence_threshold: Minimum confidence
            iou_threshold: IoU threshold for NMS
            device: Device to run on
            half_precision: Use FP16 inference
            min_area_pixels: Minimum damage area in pixels
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
        """Load YOLOv8 segmentation model for damage detection."""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is required. Install with: pip install ultralytics")
        
        model_path = self.model_path or self.DEFAULT_MODEL
        
        logger.info(f"Loading damage detection model: {model_path}")
        
        self._model = YOLO(model_path)
        self._model.to(self.device)
        
        if self.half_precision and self.device == "cuda":
            self._model.model.half()
        
        self._model_loaded = True
        logger.info(f"Damage detection model loaded on {self.device}")
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess and enhance image for better damage detection.
        Applies enhancement for improved visibility of roof damage.
        """
        from ..utils.image_enhancement import enhance_satellite_image, normalize_satellite_image
        
        # Normalize image to uint8 [0-255]
        image = normalize_satellite_image(image)
        
        # Enhance image for better damage visibility
        # Higher contrast and sharpening help detect damage details
        enhanced = enhance_satellite_image(
            image,
            contrast_factor=1.5,  # Higher contrast for damage edges
            brightness_factor=1.2,  # Brighter for better visibility
            sharpen=True,  # Sharpen to highlight damage
            denoise=True  # Reduce noise
        )
        
        return enhanced
    
    def _classify_damage_type(self, class_id: int, class_name: str) -> DamageType:
        """Map class ID to damage type."""
        if class_id in self.DAMAGE_CLASS_MAP:
            return self.DAMAGE_CLASS_MAP[class_id]
        
        # Try to match by name
        name_lower = class_name.lower()
        for damage_type in DamageType:
            if damage_type.value.replace("_", " ") in name_lower:
                return damage_type
        
        return DamageType.UNKNOWN
    
    def _calculate_severity(
        self,
        damage_area: int,
        roof_area: Optional[int] = None,
        damage_type: DamageType = DamageType.UNKNOWN
    ) -> DamageSeverity:
        """
        Calculate damage severity based on area and type.
        
        Args:
            damage_area: Damage area in pixels
            roof_area: Total roof area in pixels (if available)
            damage_type: Type of damage
            
        Returns:
            DamageSeverity level
        """
        # If we have roof area, calculate percentage
        if roof_area and roof_area > 0:
            damage_ratio = damage_area / roof_area
            
            if damage_ratio < self.SEVERITY_THRESHOLDS[DamageSeverity.LOW]:
                return DamageSeverity.LOW
            elif damage_ratio < self.SEVERITY_THRESHOLDS[DamageSeverity.MEDIUM]:
                return DamageSeverity.MEDIUM
            elif damage_ratio < self.SEVERITY_THRESHOLDS[DamageSeverity.HIGH]:
                return DamageSeverity.HIGH
            else:
                return DamageSeverity.CRITICAL
        
        # Otherwise, use absolute thresholds
        if damage_area < 500:
            return DamageSeverity.LOW
        elif damage_area < 2000:
            return DamageSeverity.MEDIUM
        elif damage_area < 5000:
            return DamageSeverity.HIGH
        else:
            return DamageSeverity.CRITICAL
    
    def _mask_to_polygon(self, mask: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """Convert binary mask to polygon."""
        import cv2
        
        try:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest, True)
            simplified = cv2.approxPolyDP(largest, epsilon, True)
            
            return [(float(p[0][0]), float(p[0][1])) for p in simplified]
            
        except Exception as e:
            logger.warning(f"Failed to extract damage polygon: {e}")
            return None
    
    def _postprocess(
        self,
        results,
        original_shape: Tuple[int, int],
        roof_area: Optional[int] = None,
        roof_id: Optional[int] = None
    ) -> List[DamageDetection]:
        """Process YOLO results into DamageDetection objects."""
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
                cls_name = result.names.get(cls_id, "unknown")
                damage_type = self._classify_damage_type(cls_id, cls_name)
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                mask = None
                polygon = None
                area_pixels = int((x2 - x1) * (y2 - y1))
                
                if masks is not None and i < len(masks.data):
                    mask_tensor = masks.data[i].cpu().numpy()
                    mask_pil = Image.fromarray((mask_tensor * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize((original_shape[1], original_shape[0]), Image.NEAREST)
                    mask = np.array(mask_pil) > 127
                    area_pixels = int(np.sum(mask))
                    polygon = self._mask_to_polygon(mask)
                
                if area_pixels < self.min_area_pixels:
                    continue
                
                severity = self._calculate_severity(area_pixels, roof_area, damage_type)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                detection = DamageDetection(
                    id=detection_id,
                    damage_type=damage_type,
                    confidence=conf,
                    severity=severity,
                    bbox=bbox,
                    mask=mask,
                    polygon=polygon,
                    area_pixels=area_pixels,
                    center=center,
                    roof_id=roof_id
                )
                
                detections.append(detection)
                detection_id += 1
        
        logger.debug(f"Detected {len(detections)} damage areas")
        return detections
    
    @memory_efficient(cleanup_after=True)
    def detect(
        self,
        image: np.ndarray,
        roof_area: Optional[int] = None,
        roof_id: Optional[int] = None,
        return_masks: bool = True
    ) -> List[DamageDetection]:
        """
        Detect damage in an image.
        
        Args:
            image: Input image
            roof_area: Total roof area in pixels (for severity calculation)
            roof_id: Associated roof ID
            return_masks: Whether to include masks
            
        Returns:
            List of DamageDetection objects
        """
        image = self.validate_image(image)
        original_shape = image.shape[:2]
        
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = self._postprocess(results, original_shape, roof_area, roof_id)
        
        if not return_masks:
            for det in detections:
                det.mask = None
        
        return detections
    
    def detect_on_roof(
        self,
        image: np.ndarray,
        roof: RoofDetection,
        return_masks: bool = True
    ) -> List[DamageDetection]:
        """
        Detect damage within a specific roof region.
        
        Args:
            image: Full image
            roof: RoofDetection to analyze
            return_masks: Whether to include masks
            
        Returns:
            List of DamageDetection objects within the roof
        """
        image = self.validate_image(image)
        
        # Crop to roof region
        x1, y1, x2, y2 = roof.bbox
        roof_crop = image[y1:y2, x1:x2]
        
        # Detect damage in crop
        detections = self.detect(
            roof_crop,
            roof_area=roof.area_pixels,
            roof_id=roof.id,
            return_masks=return_masks
        )
        
        # Adjust coordinates back to full image
        for det in detections:
            det.bbox = (
                det.bbox[0] + x1,
                det.bbox[1] + y1,
                det.bbox[2] + x1,
                det.bbox[3] + y1
            )
            det.center = (det.center[0] + x1, det.center[1] + y1)
            
            if det.polygon:
                det.polygon = [(p[0] + x1, p[1] + y1) for p in det.polygon]
        
        return detections
    
    def generate_heatmap(
        self,
        image_shape: Tuple[int, int],
        detections: List[DamageDetection],
        blur_size: int = 31
    ) -> np.ndarray:
        """
        Generate damage heatmap from detections.
        
        Args:
            image_shape: (height, width) of output
            detections: List of damage detections
            blur_size: Gaussian blur kernel size
            
        Returns:
            Heatmap array (0-255) where higher = more damage
        """
        import cv2
        
        heatmap = np.zeros(image_shape[:2], dtype=np.float32)
        
        # Severity weights
        severity_weights = {
            DamageSeverity.LOW: 0.25,
            DamageSeverity.MEDIUM: 0.5,
            DamageSeverity.HIGH: 0.75,
            DamageSeverity.CRITICAL: 1.0
        }
        
        for det in detections:
            weight = severity_weights.get(det.severity, 0.5)
            
            if det.mask is not None:
                heatmap += det.mask.astype(np.float32) * weight
            else:
                x1, y1, x2, y2 = det.bbox
                heatmap[y1:y2, x1:x2] += weight
        
        # Normalize and blur
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        heatmap = cv2.GaussianBlur(heatmap, (blur_size, blur_size), 0)
        heatmap = (heatmap * 255).astype(np.uint8)
        
        return heatmap

