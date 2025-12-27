"""
Swiss Road and Transportation Network Loader

Efficiently load Swiss road and transportation network data from geo.admin.ch APIs.
Integrates with existing terrain workflow for complete site context.
"""

import logging
import time
from typing import Optional, Tuple, List, Dict, Literal, ClassVar
from dataclasses import dataclass
from functools import wraps
from threading import Lock

import requests
from shapely.geometry import shape, box, LineString, MultiLineString, Polygon
from shapely.ops import unary_union, linemerge

try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    logging.warning("pyproj not available - coordinate conversion limited")


logger = logging.getLogger(__name__)


@dataclass
class RoadFeature:
    """Represents a road or path feature"""
    id: str
    geometry: LineString  # Road centerline or polygon
    road_class: Optional[str] = None  # e.g., "Autobahn", "Hauptstrasse", "Nebenstrasse"
    surface_type: Optional[str] = None  # e.g., "Asphalt", "Gravel", "Unpaved"
    width: Optional[float] = None  # Road width in meters
    name: Optional[str] = None  # Street name
    road_number: Optional[str] = None  # e.g., "A1", "N2"
    attributes: Optional[Dict] = None


def rate_limit(max_per_second: float):
    """
    Thread-safe rate limiting decorator to prevent API abuse

    Args:
        max_per_second: Maximum requests per second
    """
    min_interval = 1.0 / max_per_second
    last_called = [0.0]
    lock = Lock()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                elapsed = time.time() - last_called[0]
                left_to_wait = min_interval - elapsed
                if left_to_wait > 0:
                    time.sleep(left_to_wait)
                last_called[0] = time.time()

            result = func(*args, **kwargs)
            return result

        return wrapper
    return decorator


class SwissRoadLoader:
    """
    Load Swiss road and transportation network data from geo.admin.ch APIs

    Supports multiple data sources:
    - swissTLM3D Roads - complete road network with classification
    - Vector 25 Roads - topographic road representation
    - Main roads network - highway and main road network
    """

    # API Configuration
    REST_BASE = "https://api3.geo.admin.ch/rest/services"

    # Road layer names
    ROADS_TLM3D = "ch.swisstopo.swisstlm3d-strassen"  # Complete road network
    ROADS_VEC25 = "ch.swisstopo.vec25-strassennetz"  # Vector 25k roads
    ROADS_MAIN = "ch.astra.hauptstrassennetz"  # Main roads network (ASTRA)

    # Road classification types
    ROAD_CLASSES: ClassVar[Dict[str, str]] = {
        "Autobahn": "Highway/Motorway",
        "Autostrasse": "Expressway",
        "Hauptstrasse": "Main road",
        "Nebenstrasse": "Secondary road",
        "Verbindungsstrasse": "Connecting road",
        "Gemeindestrasse": "Local road",
        "Privatstrasse": "Private road",
        "Weg": "Path/Track",
        "Fussweg": "Footpath"
    }

    def __init__(
        self,
        timeout: int = 60,
        retry_count: int = 3
    ):
        """
        Initialize road loader

        Args:
            timeout: Request timeout in seconds
            retry_count: Number of retries for failed requests
        """
        self.timeout = timeout
        self.retry_count = retry_count

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

        logger.info("SwissRoadLoader initialized")

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
        if self.retry_count < 1:
            raise ValueError("retry_count must be at least 1")

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

        if last_exception is None:
            raise RuntimeError("Request failed but no exception was captured")
        raise last_exception

    def get_roads_rest(
        self,
        bbox_2056: Tuple[float, float, float, float],
        layer: Optional[str] = None,
        max_features: int = 5000
    ) -> List[RoadFeature]:
        """
        Get roads using REST API MapServer Identify endpoint.
        
        Uses grid-based queries to overcome the ~200 feature API limit.

        Args:
            bbox_2056: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
            layer: Layer to query (default: ROADS_TLM3D)
            max_features: Maximum number of roads to return

        Returns:
            List of RoadFeature objects with geometry
        """
        if layer is None:
            layer = self.ROADS_TLM3D

        logger.info(f"Fetching roads via REST API in bbox: {bbox_2056}, layer: {layer}")

        # Calculate bbox dimensions
        width = bbox_2056[2] - bbox_2056[0]
        height = bbox_2056[3] - bbox_2056[1]
        
        # API returns max ~200 features per request
        # Use grid cells of ~250m to stay under limit in dense areas
        cell_size = 250.0
        n_cols = max(1, int(width / cell_size) + 1)
        n_rows = max(1, int(height / cell_size) + 1)
        
        url = f"{self.REST_BASE}/api/MapServer/identify"
        
        all_roads = {}  # Use dict to dedupe by ID
        
        for row in range(n_rows):
            for col in range(n_cols):
                # Calculate cell bbox
                cell_minx = bbox_2056[0] + col * cell_size
                cell_miny = bbox_2056[1] + row * cell_size
                cell_maxx = min(bbox_2056[2], cell_minx + cell_size)
                cell_maxy = min(bbox_2056[3], cell_miny + cell_size)
                
                params = {
                    "geometryType": "esriGeometryEnvelope",
                    "geometry": f"{cell_minx},{cell_miny},{cell_maxx},{cell_maxy}",
                    "layers": f"all:{layer}",
                    "mapExtent": f"{cell_minx},{cell_miny},{cell_maxx},{cell_maxy}",
                    "imageDisplay": "1000,1000,96",
                    "tolerance": 0,
                    "returnGeometry": "true",
                    "geometryFormat": "geojson",
                    "sr": "2056"
                }

                try:
                    response = self._request_with_retry(url, params)
                    data = response.json()

                    for result in data.get("results", []):
                        road = self._parse_rest_result(result)
                        if road and road.id not in all_roads:
                            all_roads[road.id] = road
                            if len(all_roads) >= max_features:
                                break
                                
                except Exception as e:
                    logger.warning(f"Grid cell query failed: {e}")
                    continue
                    
                if len(all_roads) >= max_features:
                    break
            if len(all_roads) >= max_features:
                break

        roads = list(all_roads.values())
        logger.info(f"Retrieved {len(roads)} roads via REST API ({n_cols}x{n_rows} grid)")
        return roads

    def _parse_rest_result(self, result: Dict) -> Optional[RoadFeature]:
        """
        Parse a REST API identify result into RoadFeature

        Args:
            result: REST API result dict

        Returns:
            RoadFeature or None if invalid
        """
        try:
            # Parse geometry
            geom_data = result.get("geometry", {})
            if not geom_data:
                return None

            geom = shape(geom_data)

            # Convert to LineString if needed
            if geom.geom_type == "MultiLineString":
                if not geom.geoms:
                    return None
                # Try to merge contiguous segments first
                merged = linemerge(geom)
                if merged.geom_type == "LineString":
                    geom = merged
                else:
                    # Fall back to longest if merge doesn't produce single line
                    geom = max(geom.geoms, key=lambda ls: ls.length)
            elif geom.geom_type == "Polygon":
                # Convert polygon to centerline (simplified approach)
                # exterior returns LinearRing, convert to LineString
                coords = list(geom.exterior.coords)
                geom = LineString(coords)

            # Extract to 2D if 3D
            if hasattr(geom, 'has_z') and geom.has_z:
                coords = [(x, y) for x, y, *_ in geom.coords]
                geom = LineString(coords)

            # Extract properties
            attrs = result.get("attributes", {})
            feature_id = str(result.get("id", attrs.get("id", "unknown")))

            # Parse road classification and attributes
            road_class = attrs.get("objektart") or attrs.get("road_class") or attrs.get("klasse")
            surface_type = attrs.get("belagsart") or attrs.get("surface")
            name = attrs.get("name") or attrs.get("strassenname")
            road_number = attrs.get("nummer") or attrs.get("road_number")
            width = attrs.get("breite") or attrs.get("width")

            return RoadFeature(
                id=feature_id,
                geometry=geom,
                road_class=road_class,
                surface_type=surface_type,
                width=float(width) if width else None,
                name=name,
                road_number=road_number,
                attributes=attrs
            )

        except Exception as e:
            logger.warning(f"Failed to parse REST result: {e}")
            return None

    def get_roads_around_point(
        self,
        x: float,
        y: float,
        radius: float = 500,
        layer: Optional[str] = None
    ) -> List[RoadFeature]:
        """
        Get roads within radius of a point

        Args:
            x: X coordinate in EPSG:2056
            y: Y coordinate in EPSG:2056
            radius: Radius in meters
            layer: Layer to query (default: ROADS_TLM3D)

        Returns:
            List of RoadFeature objects
        """
        # Create bbox
        bbox = (x - radius, y - radius, x + radius, y + radius)

        roads = self.get_roads_rest(bbox, layer=layer)

        # Filter to circular area
        from shapely.geometry import Point
        center = Point(x, y)
        filtered = [
            r for r in roads
            if r.geometry.distance(center) <= radius
        ]

        logger.info(f"Filtered {len(filtered)}/{len(roads)} roads within {radius}m radius")
        return filtered

    def get_roads_on_parcel(
        self,
        egrid: str,
        buffer_m: float = 10
    ) -> List[RoadFeature]:
        """
        Get roads on or near a cadastral parcel

        Args:
            egrid: Swiss EGRID identifier
            buffer_m: Buffer around parcel boundary in meters

        Returns:
            List of RoadFeature objects
        """
        logger.info(f"Fetching roads for EGRID: {egrid}")

        # Import here to avoid circular dependency
        try:
            from src.loaders.cadastre import fetch_boundary_by_egrid
        except (ImportError, ModuleNotFoundError):
            try:
                from loaders.cadastre import fetch_boundary_by_egrid
            except (ImportError, ModuleNotFoundError):
                from terrain_with_site import fetch_boundary_by_egrid

        # Get parcel boundary (metadata not used here)
        site_boundary, _ = fetch_boundary_by_egrid(egrid)
        if site_boundary is None:
            logger.warning(f"No boundary found for EGRID {egrid}")
            return []

        # Create bbox with buffer
        bounds = site_boundary.bounds
        bbox = (
            bounds[0] - buffer_m,
            bounds[1] - buffer_m,
            bounds[2] + buffer_m,
            bounds[3] + buffer_m
        )

        # Get roads in bbox
        roads = self.get_roads_rest(bbox)

        # Filter to roads that intersect the parcel (with buffer)
        if buffer_m > 0:
            search_area = site_boundary.buffer(buffer_m)
        else:
            search_area = site_boundary

        filtered_roads = [
            r for r in roads
            if search_area.intersects(r.geometry)
        ]

        logger.info(
            f"Found {len(filtered_roads)} roads on parcel "
            f"(buffer: {buffer_m}m)"
        )

        return filtered_roads

    def get_road_statistics(
        self,
        roads: List[RoadFeature]
    ) -> Dict:
        """
        Calculate statistics for a list of roads

        Args:
            roads: List of RoadFeature objects

        Returns:
            Dictionary with statistics
        """
        if not roads:
            return {
                "count": 0,
                "total_length_m": 0,
                "avg_length_m": 0,
                "road_classes": {}
            }

        total_length = sum(r.geometry.length for r in roads)

        # Count by road class
        class_counts = {}
        for road in roads:
            road_class = road.road_class or "Unknown"
            class_counts[road_class] = class_counts.get(road_class, 0) + 1

        return {
            "count": len(roads),
            "total_length_m": total_length,
            "avg_length_m": total_length / len(roads),
            "road_classes": class_counts
        }


# Convenience functions

def get_roads_around_egrid(
    egrid: str,
    buffer_m: float = 10
) -> Tuple[List[RoadFeature], Dict]:
    """
    Get roads around a cadastral parcel identified by EGRID

    Args:
        egrid: Swiss EGRID identifier
        buffer_m: Buffer around parcel boundary

    Returns:
        Tuple of (roads list, statistics dict)
    """
    loader = SwissRoadLoader()
    roads = loader.get_roads_on_parcel(egrid, buffer_m)
    stats = loader.get_road_statistics(roads)

    return roads, stats


def get_roads_in_bbox(
    bbox_2056: Tuple[float, float, float, float],
    layer: Optional[str] = None
) -> Tuple[List[RoadFeature], Dict]:
    """
    Get roads in bounding box

    Args:
        bbox_2056: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
        layer: Layer to query (default: swissTLM3D roads)

    Returns:
        Tuple of (roads list, statistics dict)
    """
    loader = SwissRoadLoader()
    roads = loader.get_roads_rest(bbox_2056, layer=layer)
    stats = loader.get_road_statistics(roads)

    return roads, stats


if __name__ == "__main__":
    """Example usage"""
    import sys

    logging.basicConfig(level=logging.INFO)

    # Example 1: Get roads in bbox
    print("\n" + "="*80)
    print("Example 1: Get roads in bounding box (Zurich HB)")
    print("="*80)

    bbox = (2682500, 1247500, 2683000, 1248000)  # 500m x 500m
    try:
        roads, stats = get_roads_in_bbox(bbox)
        print(f"\nStatistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value:.1f}" if isinstance(value, float) else f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Get roads on parcel
    if len(sys.argv) > 1:
        print("\n" + "="*80)
        print(f"Example 2: Get roads for EGRID: {sys.argv[1]}")
        print("="*80)

        try:
            roads, stats = get_roads_around_egrid(sys.argv[1], buffer_m=10)
            print(f"\nStatistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value:.1f}" if isinstance(value, float) else f"  {key}: {value}")
        except Exception as e:
            print(f"Error: {e}")
