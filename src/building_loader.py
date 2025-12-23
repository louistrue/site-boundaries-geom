"""
Swiss Building Data Loader

Efficiently load Swiss building footprints and 3D data from various APIs.
Integrates with existing terrain workflow.
"""

import logging
import time
from typing import Optional, Tuple, List, Dict, Literal
from dataclasses import dataclass
from functools import wraps

import requests
from shapely.geometry import shape, box, Polygon, MultiPolygon
from shapely.ops import unary_union

try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    logging.warning("pyproj not available - coordinate conversion limited")


logger = logging.getLogger(__name__)


@dataclass
class BuildingFeature:
    """Represents a building feature"""
    id: str
    geometry: Polygon
    height: Optional[float] = None
    building_class: Optional[str] = None
    roof_type: Optional[str] = None
    year_built: Optional[int] = None
    attributes: Optional[Dict] = None


def rate_limit(max_per_second: float):
    """
    Rate limiting decorator to prevent API abuse

    Args:
        max_per_second: Maximum requests per second
    """
    min_interval = 1.0 / max_per_second
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)

            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result

        return wrapper
    return decorator


class SwissBuildingLoader:
    """
    Load Swiss building data from geo.admin.ch APIs

    Supports multiple data sources:
    - WFS (Web Feature Service) - best for area queries
    - STAC API - best for bulk tile downloads
    - REST API - best for point queries
    """

    # API Configuration
    WFS_URL = "https://wms.geo.admin.ch/"
    STAC_BASE = "https://data.geo.admin.ch/api/stac/v1"
    REST_BASE = "https://api3.geo.admin.ch/rest/services"

    # Layer names
    BUILDINGS_LAYER = "ch.swisstopo.swissbuildings3d_3_0"
    BUILDINGS_LAYER_BETA = "ch.swisstopo.swissbuildings3d_3_0-beta"

    def __init__(
        self,
        timeout: int = 60,
        retry_count: int = 3,
        use_cache: bool = True
    ):
        """
        Initialize building loader

        Args:
            timeout: Request timeout in seconds
            retry_count: Number of retries for failed requests
            use_cache: Whether to cache results (future feature)
        """
        self.timeout = timeout
        self.retry_count = retry_count
        self.use_cache = use_cache

        # Initialize coordinate transformer if available
        if PYPROJ_AVAILABLE:
            self.transformer_to_wgs84 = Transformer.from_crs(
                "EPSG:2056", "EPSG:4326", always_xy=True
            )
            self.transformer_to_2056 = Transformer.from_crs(
                "EPSG:4326", "EPSG:2056", always_xy=True
            )
        else:
            self.transformer_to_wgs84 = None
            self.transformer_to_2056 = None

        logger.info("SwissBuildingLoader initialized")

    def epsg2056_to_wgs84(self, x: float, y: float) -> Tuple[float, float]:
        """Convert EPSG:2056 to WGS84"""
        if self.transformer_to_wgs84:
            return self.transformer_to_wgs84.transform(x, y)
        else:
            # Rough approximation if pyproj not available
            lon = (x - 2600000) / 111320 + 7.44
            lat = (y - 1200000) / 111320 + 46.0
            return lon, lat

    def bbox_2056_to_wgs84(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Convert bbox from EPSG:2056 to WGS84"""
        min_lon, min_lat = self.epsg2056_to_wgs84(bbox[0], bbox[1])
        max_lon, max_lat = self.epsg2056_to_wgs84(bbox[2], bbox[3])
        return (min_lon, min_lat, max_lon, max_lat)

    @rate_limit(max_per_second=5)
    def _request_with_retry(self, url: str, params: Dict, method: str = "GET") -> requests.Response:
        """
        Make HTTP request with retry logic

        Args:
            url: Request URL
            params: Query parameters
            method: HTTP method

        Returns:
            Response object

        Raises:
            requests.RequestException: If all retries fail
        """
        last_exception = None

        for attempt in range(self.retry_count):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, params=params, timeout=self.timeout)
                else:
                    response = requests.post(url, params=params, timeout=self.timeout)

                response.raise_for_status()
                return response

            except requests.RequestException as e:
                last_exception = e
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.retry_count}): {e}")

                if attempt < self.retry_count - 1:
                    # Exponential backoff
                    time.sleep(2 ** attempt)

        raise last_exception

    def get_buildings_wfs(
        self,
        bbox_2056: Tuple[float, float, float, float],
        max_features: int = 1000,
        cql_filter: Optional[str] = None
    ) -> List[BuildingFeature]:
        """
        Get buildings using WFS service

        This is the recommended method for area-based queries.
        Fast and returns detailed building attributes.

        Args:
            bbox_2056: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
            max_features: Maximum number of buildings to return
            cql_filter: Optional CQL filter (e.g., "hoehe_max >= 50")

        Returns:
            List of BuildingFeature objects
        """
        logger.info(f"Fetching buildings via WFS in bbox: {bbox_2056}")

        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": self.BUILDINGS_LAYER,
            "srsName": "EPSG:2056",
            "bbox": f"{bbox_2056[0]},{bbox_2056[1]},{bbox_2056[2]},{bbox_2056[3]},EPSG:2056",
            "outputFormat": "application/json",
            "count": max_features
        }

        if cql_filter:
            params["CQL_FILTER"] = cql_filter

        try:
            response = self._request_with_retry(self.WFS_URL, params)
            data = response.json()

            buildings = []
            for feature in data.get("features", []):
                building = self._parse_building_feature(feature)
                if building:
                    buildings.append(building)

            logger.info(f"Retrieved {len(buildings)} buildings via WFS")
            return buildings

        except Exception as e:
            logger.error(f"WFS request failed: {e}")
            raise

    def get_buildings_stac(
        self,
        bbox_2056: Tuple[float, float, float, float],
        limit: int = 100
    ) -> List[Dict]:
        """
        Get building data tiles using STAC API

        Best for bulk downloads. Returns metadata about data tiles,
        which can then be downloaded separately.

        Args:
            bbox_2056: Bounding box in EPSG:2056
            limit: Maximum number of tiles to return

        Returns:
            List of STAC items with download links
        """
        logger.info(f"Fetching building tiles via STAC in bbox: {bbox_2056}")

        # Convert bbox to WGS84
        bbox_wgs84 = self.bbox_2056_to_wgs84(bbox_2056)

        url = f"{self.STAC_BASE}/collections/{self.BUILDINGS_LAYER}/items"
        params = {
            "bbox": ",".join(map(str, bbox_wgs84)),
            "limit": limit
        }

        try:
            response = self._request_with_retry(url, params)
            data = response.json()

            items = data.get("features", [])
            logger.info(f"Retrieved {len(items)} STAC tiles")

            return items

        except Exception as e:
            logger.error(f"STAC request failed: {e}")
            raise

    def get_buildings_around_point(
        self,
        x: float,
        y: float,
        radius: float = 500,
        method: Literal["wfs", "stac"] = "wfs"
    ) -> List[BuildingFeature]:
        """
        Get buildings within radius of a point

        Args:
            x: X coordinate in EPSG:2056
            y: Y coordinate in EPSG:2056
            radius: Radius in meters
            method: API method to use

        Returns:
            List of BuildingFeature objects
        """
        # Create bbox
        bbox = (x - radius, y - radius, x + radius, y + radius)

        if method == "wfs":
            buildings = self.get_buildings_wfs(bbox)

            # Filter to circular area
            center = shape({"type": "Point", "coordinates": [x, y]})
            filtered = [
                b for b in buildings
                if b.geometry.distance(center) <= radius
            ]

            logger.info(f"Filtered {len(filtered)}/{len(buildings)} buildings within {radius}m radius")
            return filtered

        elif method == "stac":
            tiles = self.get_buildings_stac(bbox)
            # TODO: Download and parse tiles
            logger.warning("STAC tile parsing not yet implemented")
            return []

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_buildings_on_parcel(
        self,
        egrid: str,
        buffer_m: float = 0
    ) -> List[BuildingFeature]:
        """
        Get buildings on a cadastral parcel

        Integrates with existing cadastral API from site_solid.py

        Args:
            egrid: Swiss EGRID identifier
            buffer_m: Buffer around parcel boundary in meters

        Returns:
            List of BuildingFeature objects
        """
        logger.info(f"Fetching buildings for EGRID: {egrid}")

        # Import here to avoid circular dependency
        from src.site_solid import get_site_boundary_from_api

        # Get parcel boundary using existing function
        site_boundary, metadata = get_site_boundary_from_api(egrid)

        # Create bbox with buffer
        bounds = site_boundary.bounds  # (minx, miny, maxx, maxy)
        bbox = (
            bounds[0] - buffer_m,
            bounds[1] - buffer_m,
            bounds[2] + buffer_m,
            bounds[3] + buffer_m
        )

        # Get buildings in bbox
        buildings = self.get_buildings_wfs(bbox)

        # Filter to buildings that intersect the parcel (with buffer)
        if buffer_m > 0:
            search_area = site_boundary.buffer(buffer_m)
        else:
            search_area = site_boundary

        filtered_buildings = [
            b for b in buildings
            if search_area.intersects(b.geometry)
        ]

        logger.info(
            f"Found {len(filtered_buildings)} buildings on parcel "
            f"(buffer: {buffer_m}m)"
        )

        return filtered_buildings

    def _parse_building_feature(self, feature: Dict) -> Optional[BuildingFeature]:
        """
        Parse a GeoJSON feature into BuildingFeature

        Args:
            feature: GeoJSON feature dict

        Returns:
            BuildingFeature or None if invalid
        """
        try:
            # Parse geometry
            geom = shape(feature["geometry"])

            # Extract to 2D if 3D
            if geom.has_z:
                geom = Polygon([(x, y) for x, y, *_ in geom.exterior.coords])

            # Extract properties
            props = feature.get("properties", {})

            return BuildingFeature(
                id=feature.get("id", props.get("id", "unknown")),
                geometry=geom,
                height=props.get("hoehe_max") or props.get("height"),
                building_class=props.get("gebaeudeklasse") or props.get("building_class"),
                roof_type=props.get("dachform") or props.get("roof_type"),
                year_built=props.get("baujahr") or props.get("year_built"),
                attributes=props
            )

        except Exception as e:
            logger.warning(f"Failed to parse building feature: {e}")
            return None

    def get_buildings_by_height(
        self,
        bbox_2056: Tuple[float, float, float, float],
        min_height: Optional[float] = None,
        max_height: Optional[float] = None
    ) -> List[BuildingFeature]:
        """
        Get buildings filtered by height range

        Args:
            bbox_2056: Bounding box in EPSG:2056
            min_height: Minimum height in meters (inclusive)
            max_height: Maximum height in meters (inclusive)

        Returns:
            List of BuildingFeature objects
        """
        # Build CQL filter
        filters = []
        if min_height is not None:
            filters.append(f"hoehe_max >= {min_height}")
        if max_height is not None:
            filters.append(f"hoehe_max <= {max_height}")

        cql_filter = " AND ".join(filters) if filters else None

        return self.get_buildings_wfs(bbox_2056, cql_filter=cql_filter)

    def get_building_statistics(
        self,
        buildings: List[BuildingFeature]
    ) -> Dict:
        """
        Calculate statistics for a list of buildings

        Args:
            buildings: List of BuildingFeature objects

        Returns:
            Dictionary with statistics
        """
        if not buildings:
            return {
                "count": 0,
                "total_footprint_area_m2": 0,
                "avg_height_m": 0,
                "max_height_m": 0,
                "min_height_m": 0
            }

        heights = [b.height for b in buildings if b.height is not None]
        footprint_areas = [b.geometry.area for b in buildings]

        return {
            "count": len(buildings),
            "total_footprint_area_m2": sum(footprint_areas),
            "avg_footprint_area_m2": sum(footprint_areas) / len(footprint_areas),
            "avg_height_m": sum(heights) / len(heights) if heights else 0,
            "max_height_m": max(heights) if heights else 0,
            "min_height_m": min(heights) if heights else 0,
            "buildings_with_height": len(heights)
        }


# Convenience functions matching existing API patterns

def get_buildings_around_egrid(
    egrid: str,
    radius_m: float = 500,
    buffer_m: float = 0
) -> Tuple[List[BuildingFeature], Dict]:
    """
    Get buildings around a cadastral parcel identified by EGRID

    Args:
        egrid: Swiss EGRID identifier
        radius_m: Radius around parcel center (for area query)
        buffer_m: Buffer around parcel boundary (for intersection filter)

    Returns:
        Tuple of (buildings list, statistics dict)
    """
    loader = SwissBuildingLoader()
    buildings = loader.get_buildings_on_parcel(egrid, buffer_m)
    stats = loader.get_building_statistics(buildings)

    return buildings, stats


def get_buildings_in_bbox(
    bbox_2056: Tuple[float, float, float, float],
    method: Literal["wfs", "stac"] = "wfs"
) -> Tuple[List[BuildingFeature], Dict]:
    """
    Get buildings in bounding box

    Args:
        bbox_2056: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
        method: API method ('wfs' or 'stac')

    Returns:
        Tuple of (buildings list, statistics dict)
    """
    loader = SwissBuildingLoader()

    if method == "wfs":
        buildings = loader.get_buildings_wfs(bbox_2056)
    elif method == "stac":
        tiles = loader.get_buildings_stac(bbox_2056)
        # TODO: Parse tiles
        buildings = []
    else:
        raise ValueError(f"Unknown method: {method}")

    stats = loader.get_building_statistics(buildings)

    return buildings, stats


if __name__ == "__main__":
    """Example usage"""
    import sys

    logging.basicConfig(level=logging.INFO)

    # Example 1: Get buildings in bbox
    print("\n" + "="*80)
    print("Example 1: Get buildings in bounding box (Zurich HB)")
    print("="*80)

    bbox = (2682500, 1247500, 2683000, 1248000)  # 500m x 500m
    try:
        buildings, stats = get_buildings_in_bbox(bbox, method="wfs")
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.1f}" if isinstance(value, float) else f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Get buildings on parcel
    if len(sys.argv) > 1:
        print("\n" + "="*80)
        print(f"Example 2: Get buildings for EGRID: {sys.argv[1]}")
        print("="*80)

        try:
            buildings, stats = get_buildings_around_egrid(sys.argv[1], buffer_m=10)
            print(f"\nStatistics:")
            for key, value in stats.items():
                print(f"  {key}: {value:.1f}" if isinstance(value, float) else f"  {key}: {value}")
        except Exception as e:
            print(f"Error: {e}")
