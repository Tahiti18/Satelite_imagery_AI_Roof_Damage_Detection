"""
Geocoding utilities for zipcode and address-based roof damage detection.
Zipcode geocoding uses Zippopotam.us.
Address geocoding uses MapTiler Geocoding API.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Optional, Tuple

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
        return (
            (self.min_lat + self.max_lat) / 2,
            (self.min_lng + self.max_lng) / 2,
        )

    @property
    def width_degrees(self) -> float:
        return abs(self.max_lng - self.min_lng)

    @property
    def height_degrees(self) -> float:
        return abs(self.max_lat - self.min_lat)

    def to_dict(self) -> dict:
        return {
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "min_lng": self.min_lng,
            "max_lng": self.max_lng,
            "center": {"lat": self.center[0], "lng": self.center[1]},
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
        return {
            "zipcode": self.zipcode,
            "center": {"lat": self.center_lat, "lng": self.center_lng},
            "bounding_box": self.bounding_box.to_dict(),
            "state": self.state,
            "city": self.city,
        }


@dataclass(frozen=True)
class AddressInfo:
    """Immutable address information compatible with pipeline result generation."""
    address: str
    center_lat: float
    center_lng: float
    bounding_box: BoundingBox
    city: Optional[str] = None
    state: Optional[str] = None

    @property
    def zipcode(self) -> str:
        return "address"

    def to_dict(self) -> dict:
        return {
            "address": self.address,
            "center": {"lat": self.center_lat, "lng": self.center_lng},
            "bounding_box": self.bounding_box.to_dict(),
            "state": self.state,
            "city": self.city,
        }


class ZipcodeGeocoder:
    """
    Geocoder for US zipcodes and street addresses.

    Zipcodes:
        Uses Zippopotam.us API.

    Addresses:
        Uses MapTiler Geocoding API.
    """

    ZIPPOPOTAM_API_URL = "https://api.zippopotam.us/us"
    DEFAULT_ZIPCODE_RADIUS_DEG = 0.05

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._cache: dict = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        self._cache.clear()

    @memory_efficient(cleanup_after=False)
    async def geocode_zipcode(self, zipcode: str) -> Optional[ZipcodeInfo]:
        zipcode = zipcode.strip()

        if not zipcode.isdigit() or len(zipcode) != 5:
            logger.error(f"Invalid zipcode format: {zipcode}")
            return None

        cache_key = f"zipcode:{zipcode}"
        if cache_key in self._cache:
            logger.debug(f"Cache hit for zipcode: {zipcode}")
            return self._cache[cache_key]

        result = await self._geocode_via_zippopotam(zipcode)

        if result is None:
            logger.warning(f"Failed to geocode zipcode: {zipcode}")
            return None

        self._cache[cache_key] = result
        logger.info(
            f"Geocoded {zipcode}: center=({result.center_lat:.4f}, {result.center_lng:.4f})"
        )

        return result

    async def _geocode_via_zippopotam(self, zipcode: str) -> Optional[ZipcodeInfo]:
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

                if "places" not in data or len(data["places"]) == 0:
                    return None

                place = data["places"][0]
                lat = float(place["latitude"])
                lng = float(place["longitude"])
                state = place.get("state abbreviation")
                city = place.get("place name")

                bbox = self._estimate_bounding_box(lat, lng, zipcode)

                return ZipcodeInfo(
                    zipcode=zipcode,
                    center_lat=lat,
                    center_lng=lng,
                    bounding_box=bbox,
                    state=state,
                    city=city,
                )

            except httpx.TimeoutException:
                logger.warning(
                    f"Timeout geocoding {zipcode}, attempt {attempt + 1}/{self.max_retries}"
                )
                await asyncio.sleep(1 * (attempt + 1))
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error geocoding {zipcode}: {e}")
                return None
            except Exception as e:
                logger.error(f"Error geocoding {zipcode}: {e}")
                return None

        return None

    async def geocode_address(
        self,
        address: str,
        maptiler_api_key: str,
        radius_meters: float = 120.0,
    ) -> Optional[AddressInfo]:
        """
        Geocode a street address using MapTiler and create a small property-level bounding box.
        """
        address = address.strip()

        if not address:
            logger.error("Invalid empty address")
            return None

        radius_meters = max(25.0, min(float(radius_meters), 500.0))

        cache_key = f"address:{address.lower()}:{radius_meters}"
        if cache_key in self._cache:
            logger.debug(f"Cache hit for address: {address}")
            return self._cache[cache_key]

        client = await self._get_client()

        try:
            response = await client.get(
                f"https://api.maptiler.com/geocoding/{address}.json",
                params={
                    "key": maptiler_api_key,
                    "limit": 1,
                    "country": "us",
                },
            )
            response.raise_for_status()
            data = response.json()

            features = data.get("features", [])
            if not features:
                logger.warning(f"Address not found: {address}")
                return None

            feature = features[0]
            center = feature.get("center")

            if not center or len(center) < 2:
                logger.warning(f"No center returned for address: {address}")
                return None

            lng = float(center[0])
            lat = float(center[1])

            city = None
            state = None

            for item in feature.get("context", []) or []:
                item_id = item.get("id", "")
                text = item.get("text", "")
                if item_id.startswith("place"):
                    city = text
                elif item_id.startswith("region"):
                    state = text

            bbox = self._bounding_box_from_radius(
                center_lat=lat,
                center_lng=lng,
                radius_meters=radius_meters,
            )

            result = AddressInfo(
                address=address,
                center_lat=lat,
                center_lng=lng,
                bounding_box=bbox,
                city=city,
                state=state,
            )

            self._cache[cache_key] = result

            logger.info(
                f"Geocoded address '{address}': center=({lat:.6f}, {lng:.6f}), "
                f"radius={radius_meters}m"
            )

            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error geocoding address '{address}': {e}")
            return None
        except Exception as e:
            logger.error(f"Error geocoding address '{address}': {e}")
            return None

    def _estimate_bounding_box(
        self,
        center_lat: float,
        center_lng: float,
        zipcode: str,
    ) -> BoundingBox:
        first_digit = int(zipcode[0])

        if first_digit in [0, 1, 2, 9]:
            radius_deg = 0.03
        elif first_digit in [3, 4, 5, 6]:
            radius_deg = 0.05
        else:
            radius_deg = 0.08

        lng_adjustment = 1 / max(0.1, abs(center_lat) / 90)

        return BoundingBox(
            min_lat=center_lat - radius_deg,
            max_lat=center_lat + radius_deg,
            min_lng=center_lng - (radius_deg * lng_adjustment),
            max_lng=center_lng + (radius_deg * lng_adjustment),
        )

    def _bounding_box_from_radius(
        self,
        center_lat: float,
        center_lng: float,
        radius_meters: float,
    ) -> BoundingBox:
        lat_delta = radius_meters / 111_320.0
        lng_delta = radius_meters / (
            111_320.0 * max(0.1, math.cos(math.radians(center_lat)))
        )

        return BoundingBox(
            min_lat=center_lat - lat_delta,
            max_lat=center_lat + lat_delta,
            min_lng=center_lng - lng_delta,
            max_lng=center_lng + lng_delta,
        )

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.debug("Geocoder cache cleared")


def geocode_zipcode_sync(zipcode: str, timeout: int = 30) -> Optional[ZipcodeInfo]:
    async def _run():
        geocoder = ZipcodeGeocoder(timeout=timeout)
        try:
            return await geocoder.geocode_zipcode(zipcode)
        finally:
            await geocoder.close()

    return asyncio.run(_run())
