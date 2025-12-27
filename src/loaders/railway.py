"""
OpenStreetMap Railway Loader

Load railway track data from OpenStreetMap Overpass API.
Converts WGS84 coordinates to EPSG:2056 for integration with Swiss data.
"""

import logging
import time
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from functools import wraps
from threading import Lock

import requests
from shapely.geometry import LineString, Point

try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    logging.warning("pyproj not available - coordinate conversion limited")


logger = logging.getLogger(__name__)

# Overpass API endpoint
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"

# Rate limiting for Overpass API (10 requests per minute)
_rate_limit_lock = Lock()
_last_request_time = [0.0]
_min_request_interval = 6.0  # 6 seconds = 10 requests per minute


def rate_limit_overpass(func):
    """Thread-safe rate limiting decorator for Overpass API"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _last_request_time
        with _rate_limit_lock:
            elapsed = time.time() - _last_request_time[0]
            left_to_wait = _min_request_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            _last_request_time[0] = time.time()
        
        return func(*args, **kwargs)
    return wrapper


@dataclass
class RailwayFeature:
    """Represents a railway track feature"""
    id: str
    geometry: LineString
    railway_type: str  # "rail", "tram", "light_rail", "subway", "narrow_gauge", "funicular"
    name: Optional[str] = None
    electrified: Optional[str] = None  # "contact_line", "rail", "yes", "no"
    gauge: Optional[str] = None  # Track gauge in mm (e.g., "1435", "1000")
    tracks: Optional[int] = None  # Number of tracks
    service: Optional[str] = None  # "main", "branch", "spur", "yard", "siding"
    usage: Optional[str] = None  # "main", "branch", "industrial", "tourism"
    attributes: Optional[Dict] = None


class SwissRailwayLoader:
    """
    Load railway track data from OpenStreetMap Overpass API
    
    Converts WGS84 coordinates to EPSG:2056 for integration with Swiss data.
    """
    
    def __init__(
        self,
        timeout: int = 30,
        retry_count: int = 3
    ):
        """
        Initialize railway loader
        
        Args:
            timeout: Request timeout in seconds
            retry_count: Number of retries for failed requests
        """
        self.timeout = timeout
        self.retry_count = retry_count
        
        # Initialize coordinate transformer
        if PYPROJ_AVAILABLE:
            self.transformer_to_2056 = Transformer.from_crs(
                "EPSG:4326", "EPSG:2056", always_xy=True
            )
            self.transformer_to_wgs84 = Transformer.from_crs(
                "EPSG:2056", "EPSG:4326", always_xy=True
            )
        else:
            self.transformer_to_2056 = None
            self.transformer_to_wgs84 = None
            logger.warning("pyproj not available - coordinate conversion will be approximate")
        
        logger.info("SwissRailwayLoader initialized")
    
    def wgs84_to_epsg2056(self, lon: float, lat: float) -> Tuple[float, float]:
        """Convert WGS84 to EPSG:2056"""
        if self.transformer_to_2056:
            return self.transformer_to_2056.transform(lon, lat)
        else:
            # Rough approximation if pyproj not available
            logger.debug("Using approximate coordinate conversion (pyproj unavailable)")
            x = (lon - 7.44) * 111320 + 2600000
            y = (lat - 46.0) * 111320 + 1200000
            return x, y
    
    def bbox_wgs84_to_epsg2056(self, bbox_wgs84: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Convert bbox from WGS84 to EPSG:2056"""
        min_lon, min_lat, max_lon, max_lat = bbox_wgs84
        min_x, min_y = self.wgs84_to_epsg2056(min_lon, min_lat)
        max_x, max_y = self.wgs84_to_epsg2056(max_lon, max_lat)
        return (min_x, min_y, max_x, max_y)
    
    @rate_limit_overpass
    def _request_overpass(self, query: str) -> Optional[Dict]:
        """
        Make request to Overpass API with retry logic
        
        Args:
            query: Overpass QL query string
            
        Returns:
            JSON response dict or None if failed
        """
        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    OVERPASS_API_URL,
                    data={'data': query},
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.warning(f"Overpass API request failed (attempt {attempt + 1}/{self.retry_count}): {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All Overpass API retries failed")
                    return None
        return None
    
    def get_railways_in_bbox(
        self,
        bbox_2056: Tuple[float, float, float, float],
        railway_types: Optional[List[str]] = None
    ) -> List[RailwayFeature]:
        """
        Get railways in bounding box
        
        Args:
            bbox_2056: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
            railway_types: List of railway types to include (default: all)
                          Options: "rail", "tram", "light_rail", "subway", "narrow_gauge", "funicular"
        
        Returns:
            List of RailwayFeature objects
        """
        # Convert bbox corners from EPSG:2056 to WGS84
        # bbox_2056 is (min_x, min_y, max_x, max_y)
        min_x_2056, min_y_2056, max_x_2056, max_y_2056 = bbox_2056
        
        # Convert corners to WGS84 (reverse conversion)
        if self.transformer_to_wgs84:
            min_lon, min_lat = self.transformer_to_wgs84.transform(min_x_2056, min_y_2056)
            max_lon, max_lat = self.transformer_to_wgs84.transform(max_x_2056, max_y_2056)
        else:
            # Approximate conversion
            min_lon = (min_x_2056 - 2600000) / 111320 + 7.44
            min_lat = (min_y_2056 - 1200000) / 111320 + 46.0
            max_lon = (max_x_2056 - 2600000) / 111320 + 7.44
            max_lat = (max_y_2056 - 1200000) / 111320 + 46.0
        
        # Overpass expects (south, west, north, east) - ensure correct order
        south = min(min_lat, max_lat)
        north = max(min_lat, max_lat)
        west = min(min_lon, max_lon)
        east = max(min_lon, max_lon)
        
        # Build query
        if railway_types:
            railway_filter = "|".join(railway_types)
            railway_condition = f'["railway"~"^{railway_filter}$"]'
        else:
            railway_condition = '["railway"]'
        
        query = f'''
[out:json][timeout:{self.timeout}];
(
  way{railway_condition}({south},{west},{north},{east});
);
out geom;
'''
        
        logger.info(f"Querying Overpass API for railways in bbox: {bbox_2056}")
        
        data = self._request_overpass(query)
        if not data:
            return []
        
        railways = []
        elements = data.get('elements', [])
        
        for element in elements:
            if element.get('type') != 'way':
                continue
            
            tags = element.get('tags', {})
            railway_type = tags.get('railway', 'rail')
            
            # Skip if not in requested types
            if railway_types and railway_type not in railway_types:
                continue
            
            # Get geometry
            geometry_data = element.get('geometry', [])
            if not geometry_data:
                continue
            
            # Convert coordinates from WGS84 to EPSG:2056
            coords_2056 = []
            for point in geometry_data:
                lon = point.get('lon')
                lat = point.get('lat')
                if lon is not None and lat is not None:
                    x, y = self.wgs84_to_epsg2056(lon, lat)
                    coords_2056.append((x, y))
            
            if len(coords_2056) < 2:
                continue
            
            # Create LineString
            try:
                geometry = LineString(coords_2056)
            except Exception as e:
                logger.warning(f"Failed to create LineString for railway {element.get('id')}: {e}")
                continue
            
            # Extract attributes
            feature = RailwayFeature(
                id=str(element.get('id', 'unknown')),
                geometry=geometry,
                railway_type=railway_type,
                name=tags.get('name'),
                electrified=tags.get('electrified'),
                gauge=tags.get('gauge'),
                tracks=self._parse_int(tags.get('tracks')),
                service=tags.get('service'),
                usage=tags.get('usage'),
                attributes=tags
            )
            
            railways.append(feature)
        
        logger.info(f"Retrieved {len(railways)} railway features from Overpass API")
        return railways
    
    def _parse_int(self, value: Optional[str]) -> Optional[int]:
        """Parse integer from string, return None if invalid"""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def get_railways_around_point(
        self,
        x: float,
        y: float,
        radius: float = 500,
        railway_types: Optional[List[str]] = None
    ) -> List[RailwayFeature]:
        """
        Get railways within radius of a point
        
        Args:
            x: X coordinate in EPSG:2056
            y: Y coordinate in EPSG:2056
            radius: Radius in meters
            railway_types: List of railway types to include
            
        Returns:
            List of RailwayFeature objects
        """
        # Create bbox
        bbox = (
            x - radius,
            y - radius,
            x + radius,
            y + radius
        )
        
        railways = self.get_railways_in_bbox(bbox, railway_types)
        
        # Filter to circular area
        center = Point(x, y)
        filtered = [
            r for r in railways
            if r.geometry.distance(center) <= radius
        ]
        
        logger.info(f"Filtered {len(filtered)}/{len(railways)} railways within {radius}m radius")
        return filtered
    
    def get_railways_on_parcel(
        self,
        egrid: str,
        buffer_m: float = 10,
        railway_types: Optional[List[str]] = None
    ) -> List[RailwayFeature]:
        """
        Get railways on or near a cadastral parcel
        
        Args:
            egrid: Swiss EGRID identifier
            buffer_m: Buffer around parcel boundary in meters
            railway_types: List of railway types to include
            
        Returns:
            List of RailwayFeature objects
        """
        logger.info(f"Fetching railways for EGRID: {egrid}")
        
        # Import here to avoid circular dependency
        try:
            from src.loaders.cadastre import fetch_boundary_by_egrid
        except ImportError:
            from loaders.cadastre import fetch_boundary_by_egrid
        
        # Get parcel boundary
        site_boundary, _metadata = fetch_boundary_by_egrid(egrid)
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
        
        # Get railways in bbox
        railways = self.get_railways_in_bbox(bbox, railway_types)
        
        # Filter to railways that intersect the parcel (with buffer)
        if buffer_m > 0:
            search_area = site_boundary.buffer(buffer_m)
        else:
            search_area = site_boundary
        
        filtered_railways = [
            r for r in railways
            if search_area.intersects(r.geometry)
        ]
        
        logger.info(
            f"Found {len(filtered_railways)} railways on parcel "
            f"(buffer: {buffer_m}m)"
        )
        
        return filtered_railways


# Convenience functions

def get_railways_around_egrid(
    egrid: str,
    buffer_m: float = 10,
    railway_types: Optional[List[str]] = None
) -> List[RailwayFeature]:
    """
    Get railways around a cadastral parcel identified by EGRID
    
    Args:
        egrid: Swiss EGRID identifier
        buffer_m: Buffer around parcel boundary
        railway_types: List of railway types to include
        
    Returns:
        List of RailwayFeature objects
    """
    loader = SwissRailwayLoader()
    return loader.get_railways_on_parcel(egrid, buffer_m, railway_types)


def get_railways_in_bbox(
    bbox_2056: Tuple[float, float, float, float],
    railway_types: Optional[List[str]] = None
) -> List[RailwayFeature]:
    """
    Get railways in bounding box
    
    Args:
        bbox_2056: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
        railway_types: List of railway types to include
        
    Returns:
        List of RailwayFeature objects
    """
    loader = SwissRailwayLoader()
    return loader.get_railways_in_bbox(bbox_2056, railway_types)


if __name__ == "__main__":
    """Example usage"""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Example: Get railways in bbox (Zurich HB area)
    print("\n" + "="*80)
    print("Example: Get railways in bounding box (Zurich HB)")
    print("="*80)
    
    bbox = (2682500, 1247500, 2683500, 1248500)  # 500m x 500m
    try:
        railways = get_railways_in_bbox(bbox)
        print(f"\nFound {len(railways)} railways")
        for r in railways[:5]:
            print(f"  - {r.name or 'unnamed'} ({r.railway_type})")
    except Exception as e:
        print(f"Error: {e}")

