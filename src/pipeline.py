"""
Output generation and visualization utilities for the roof damage pipeline.

This module is intentionally compatible with src.pipeline.RoofDamagePipeline.
It provides:
- AnalysisResult dataclass
- ResultGenerator for JSON / GeoJSON output
- Visualizer for annotated images and damage heatmaps
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


@dataclass
class DamageSummary:
    """Summary of detected roof damage."""
    total_damage_instances: int = 0
    damage_types: Dict[str, int] = field(default_factory=dict)
    average_confidence: float = 0.0
    max_confidence: float = 0.0


@dataclass
class AnalysisResult:
    """Structured result returned by the roof damage pipeline."""
    zipcode: str
    city: Optional[str] = None
    state: Optional[str] = None
    center_lat: Optional[float] = None
    center_lng: Optional[float] = None
    bounding_box: Optional[Dict[str, float]] = None

    total_roofs: int = 0
    roofs_with_damage: int = 0
    total_damages: int = 0
    damage_summary: Dict[str, Any] = field(default_factory=dict)

    roofs: List[Dict[str, Any]] = field(default_factory=list)
    damages: List[Dict[str, Any]] = field(default_factory=list)

    image_width: Optional[int] = None
    image_height: Optional[int] = None
    tiles_processed: Optional[int] = None
    processing_time: Optional[float] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class ResultGenerator:
    """Creates and saves structured pipeline results."""

    def __init__(self, output_dir: str | Path = "./output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_result(
        self,
        zipcode_info: Any,
        roofs: Sequence[Any],
        damages: Sequence[Any],
        image_width: int,
        image_height: int,
        tiles_processed: int,
        processing_time: float,
        performance_metrics: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        roof_records = [self._object_to_record(r, index=i) for i, r in enumerate(roofs)]
        damage_records = [self._object_to_record(d, index=i) for i, d in enumerate(damages)]

        roof_ids_with_damage = set()
        for damage in damages:
            roof_id = getattr(damage, "roof_id", None)
            if roof_id is not None:
                roof_ids_with_damage.add(roof_id)

        if not roof_ids_with_damage:
            roof_ids_with_damage = self._infer_damaged_roofs(roofs, damages)

        damage_summary = self._summarize_damages(damages)

        bbox = getattr(zipcode_info, "bounding_box", None)
        bbox_dict = None
        if bbox is not None:
            bbox_dict = {
                "min_lat": getattr(bbox, "min_lat", None),
                "max_lat": getattr(bbox, "max_lat", None),
                "min_lng": getattr(bbox, "min_lng", None),
                "max_lng": getattr(bbox, "max_lng", None),
            }

        return AnalysisResult(
            zipcode=str(getattr(zipcode_info, "zipcode", "")),
            city=getattr(zipcode_info, "city", None),
            state=getattr(zipcode_info, "state", None),
            center_lat=getattr(zipcode_info, "center_lat", None)
            or getattr(zipcode_info, "lat", None)
            or getattr(zipcode_info, "latitude", None),
            center_lng=getattr(zipcode_info, "center_lng", None)
            or getattr(zipcode_info, "lng", None)
            or getattr(zipcode_info, "longitude", None),
            bounding_box=bbox_dict,
            total_roofs=len(roofs),
            roofs_with_damage=len(roof_ids_with_damage),
            total_damages=len(damages),
            damage_summary=damage_summary,
            roofs=roof_records,
            damages=damage_records,
            image_width=image_width,
            image_height=image_height,
            tiles_processed=tiles_processed,
            processing_time=processing_time,
            performance_metrics=performance_metrics or {},
        )

    def save_json(self, result: AnalysisResult, path: str | Path) -> str:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.to_json(indent=2), encoding="utf-8")
        logger.info(f"Saved JSON result to {output_path}")
        return str(output_path)

    def save_geojson(self, result: AnalysisResult, path: str | Path) -> str:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features: List[Dict[str, Any]] = []

        for roof in result.roofs:
            feature = self._record_to_feature(roof, feature_type="roof")
            if feature:
                features.append(feature)

        for damage in result.damages:
            feature = self._record_to_feature(damage, feature_type="damage")
            if feature:
                features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "properties": {
                "zipcode": result.zipcode,
                "city": result.city,
                "state": result.state,
                "total_roofs": result.total_roofs,
                "roofs_with_damage": result.roofs_with_damage,
                "total_damages": result.total_damages,
                "created_at": result.created_at,
            },
            "features": features,
        }

        output_path.write_text(json.dumps(geojson, indent=2, default=str), encoding="utf-8")
        logger.info(f"Saved GeoJSON result to {output_path}")
        return str(output_path)

    def _object_to_record(self, obj: Any, index: int = 0) -> Dict[str, Any]:
        if isinstance(obj, dict):
            record = dict(obj)
            record.setdefault("index", index)
            return self._json_safe(record)

        record: Dict[str, Any] = {"index": index}

        for attr in [
            "id",
            "roof_id",
            "damage_id",
            "class_id",
            "class_name",
            "label",
            "confidence",
            "score",
            "bbox",
            "box",
            "polygon",
            "points",
            "area",
            "area_pixels",
            "centroid",
            "mask_area",
            "severity",
            "damage_type",
        ]:
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                record[attr] = self._json_safe(value)

        if hasattr(obj, "__dict__"):
            for key, value in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                if key not in record:
                    record[key] = self._json_safe(value)

        return record

    def _json_safe(self, value: Any) -> Any:
        if np is not None:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, (np.bool_,)):
                return bool(value)

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]

        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}

        try:
            json.dumps(value)
            return value
        except Exception:
            return str(value)

    def _summarize_damages(self, damages: Sequence[Any]) -> Dict[str, Any]:
        damage_types: Dict[str, int] = {}
        confidences: List[float] = []

        for damage in damages:
            label = (
                getattr(damage, "damage_type", None)
                or getattr(damage, "class_name", None)
                or getattr(damage, "label", None)
                or "damage"
            )
            label = str(label)
            damage_types[label] = damage_types.get(label, 0) + 1

            confidence = getattr(damage, "confidence", None)
            if confidence is None:
                confidence = getattr(damage, "score", None)
            if confidence is not None:
                try:
                    confidences.append(float(confidence))
                except Exception:
                    pass

        return {
            "total_damage_instances": len(damages),
            "damage_types": damage_types,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0,
        }

    def _infer_damaged_roofs(self, roofs: Sequence[Any], damages: Sequence[Any]) -> set:
        damaged = set()

        for i, roof in enumerate(roofs):
            roof_bbox = self._get_bbox(roof)
            if roof_bbox is None:
                continue

            for damage in damages:
                damage_bbox = self._get_bbox(damage)
                if damage_bbox is None:
                    continue

                if self._bbox_intersects(roof_bbox, damage_bbox):
                    damaged.add(i)
                    break

        return damaged

    def _get_bbox(self, obj: Any) -> Optional[Tuple[float, float, float, float]]:
        bbox = None

        if isinstance(obj, dict):
            bbox = obj.get("bbox") or obj.get("box")
        else:
            bbox = getattr(obj, "bbox", None) or getattr(obj, "box", None)

        if bbox is None or len(bbox) < 4:
            return None

        try:
            return tuple(float(x) for x in bbox[:4])  # type: ignore
        except Exception:
            return None

    def _bbox_intersects(
        self,
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
    ) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    def _record_to_feature(
        self,
        record: Dict[str, Any],
        feature_type: str,
    ) -> Optional[Dict[str, Any]]:
        geometry = record.get("geometry")

        if geometry:
            geom = geometry
        else:
            polygon = record.get("polygon") or record.get("points")
            if polygon:
                coords = [[list(p) for p in polygon]]
                if coords[0] and coords[0][0] != coords[0][-1]:
                    coords[0].append(coords[0][0])
                geom = {"type": "Polygon", "coordinates": coords}
            else:
                bbox = record.get("bbox") or record.get("box")
                if bbox and len(bbox) >= 4:
                    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                    geom = {
                        "type": "Polygon",
                        "coordinates": [[
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2],
                            [x1, y1],
                        ]],
                    }
                else:
                    return None

        properties = {
            k: v
            for k, v in record.items()
            if k not in {"geometry", "polygon", "points", "bbox", "box", "mask"}
        }
        properties["feature_type"] = feature_type

        return {
            "type": "Feature",
            "geometry": geom,
            "properties": properties,
        }


class Visualizer:
    """Image annotation utilities for roofs and damage detections."""

    def __init__(self, output_dir: str | Path = "./output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def draw_all(
        self,
        image: Any,
        roofs: Sequence[Any],
        damages: Sequence[Any],
        draw_masks: bool = True,
        draw_boxes: bool = True,
        draw_labels: bool = False,
    ) -> Any:
        if cv2 is None or np is None:
            logger.warning("OpenCV/numpy unavailable; returning image without visualization")
            return image

        vis = self._ensure_bgr_image(image)

        for roof in roofs:
            if draw_masks:
                self._draw_mask(vis, roof, color=(0, 180, 255), alpha=0.25)
            if draw_boxes:
                self._draw_bbox(vis, roof, color=(0, 180, 255), label="roof" if draw_labels else None)
            self._draw_polygon(vis, roof, color=(0, 180, 255))

        for damage in damages:
            if draw_masks:
                self._draw_mask(vis, damage, color=(0, 0, 255), alpha=0.35)
            if draw_boxes:
                label = self._get_label(damage) if draw_labels else None
                self._draw_bbox(vis, damage, color=(0, 0, 255), label=label)
            self._draw_polygon(vis, damage, color=(0, 0, 255))

        return vis

    def add_summary_overlay(
        self,
        image: Any,
        total_roofs: int,
        roofs_with_damage: int,
        damage_summary: Dict[str, Any],
    ) -> Any:
        if cv2 is None or np is None:
            return image

        vis = self._ensure_bgr_image(image)
        overlay = vis.copy()

        panel_h = 110
        cv2.rectangle(overlay, (10, 10), (460, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

        lines = [
            f"Roofs detected: {total_roofs}",
            f"Roofs with possible damage: {roofs_with_damage}",
            f"Damage instances: {damage_summary.get('total_damage_instances', 0)}",
            f"Avg confidence: {damage_summary.get('average_confidence', 0.0):.2f}",
        ]

        y = 35
        for line in lines:
            cv2.putText(
                vis,
                line,
                (25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 22

        return vis

    def generate_heatmap(self, image: Any, damages: Sequence[Any]) -> Any:
        if cv2 is None or np is None:
            return image

        base = self._ensure_bgr_image(image)
        h, w = base.shape[:2]
        heat = np.zeros((h, w), dtype=np.float32)

        for damage in damages:
            mask = self._get_mask(damage)
            if mask is not None:
                mask = self._resize_mask(mask, w, h)
                heat += mask.astype(np.float32)
                continue

            bbox = self._get_bbox(damage)
            if bbox:
                x1, y1, x2, y2 = self._clip_bbox(bbox, w, h)
                if x2 > x1 and y2 > y1:
                    heat[y1:y2, x1:x2] += 1.0

        if heat.max() > 0:
            heat = heat / heat.max()
            heat_uint8 = (heat * 255).astype(np.uint8)
        else:
            heat_uint8 = heat.astype(np.uint8)

        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(base, 0.65, heat_color, 0.35, 0)
        return blended

    def save(self, image: Any, path: str | Path) -> str:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if cv2 is None:
            raise RuntimeError("OpenCV is required to save visualization images")

        img = self._ensure_bgr_image(image)
        success = cv2.imwrite(str(output_path), img)

        if not success:
            raise RuntimeError(f"Failed to save image to {output_path}")

        logger.info(f"Saved visualization to {output_path}")
        return str(output_path)

    def _ensure_bgr_image(self, image: Any) -> Any:
        if cv2 is None or np is None:
            return image

        if isinstance(image, (str, Path)):
            loaded = cv2.imread(str(image))
            if loaded is None:
                raise ValueError(f"Unable to read image: {image}")
            return loaded

        arr = image.copy() if hasattr(image, "copy") else np.array(image)

        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        if len(arr.shape) == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

        return arr

    def _get_bbox(self, obj: Any) -> Optional[Tuple[float, float, float, float]]:
        bbox = None

        if isinstance(obj, dict):
            bbox = obj.get("bbox") or obj.get("box")
        else:
            bbox = getattr(obj, "bbox", None) or getattr(obj, "box", None)

        if bbox is None or len(bbox) < 4:
            return None

        try:
            return tuple(float(v) for v in bbox[:4])  # type: ignore
        except Exception:
            return None

    def _clip_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width - 1, int(x2)))
        y2 = max(0, min(height - 1, int(y2)))
        return x1, y1, x2, y2

    def _draw_bbox(
        self,
        image: Any,
        obj: Any,
        color: Tuple[int, int, int],
        label: Optional[str] = None,
    ) -> None:
        bbox = self._get_bbox(obj)
        if not bbox:
            return

        h, w = image.shape[:2]
        x1, y1, x2, y2 = self._clip_bbox(bbox, w, h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        if label:
            cv2.putText(
                image,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_polygon(
        self,
        image: Any,
        obj: Any,
        color: Tuple[int, int, int],
    ) -> None:
        polygon = None

        if isinstance(obj, dict):
            polygon = obj.get("polygon") or obj.get("points")
        else:
            polygon = getattr(obj, "polygon", None) or getattr(obj, "points", None)

        if polygon is None:
            return

        try:
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, color, 2)
        except Exception:
            return

    def _get_mask(self, obj: Any) -> Optional[Any]:
        if isinstance(obj, dict):
            return obj.get("mask")
        return getattr(obj, "mask", None)

    def _resize_mask(self, mask: Any, width: int, height: int) -> Any:
        mask_arr = np.array(mask)

        if mask_arr.ndim > 2:
            mask_arr = mask_arr.squeeze()

        if mask_arr.shape[0] != height or mask_arr.shape[1] != width:
            mask_arr = cv2.resize(mask_arr.astype("float32"), (width, height))

        return mask_arr > 0.5

    def _draw_mask(
        self,
        image: Any,
        obj: Any,
        color: Tuple[int, int, int],
        alpha: float = 0.3,
    ) -> None:
        mask = self._get_mask(obj)
        if mask is None:
            return

        h, w = image.shape[:2]

        try:
            mask_arr = self._resize_mask(mask, w, h)
        except Exception:
            return

        colored = np.zeros_like(image)
        colored[:] = color

        image[mask_arr] = cv2.addWeighted(
            image[mask_arr],
            1.0 - alpha,
            colored[mask_arr],
            alpha,
            0,
        )

    def _get_label(self, obj: Any) -> str:
        if isinstance(obj, dict):
            label = (
                obj.get("damage_type")
                or obj.get("class_name")
                or obj.get("label")
                or "damage"
            )
            confidence = obj.get("confidence") or obj.get("score")
        else:
            label = (
                getattr(obj, "damage_type", None)
                or getattr(obj, "class_name", None)
                or getattr(obj, "label", None)
                or "damage"
            )
            confidence = getattr(obj, "confidence", None) or getattr(obj, "score", None)

        if confidence is not None:
            try:
                return f"{label} {float(confidence):.2f}"
            except Exception:
                return str(label)

        return str(label)
