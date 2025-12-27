"""
OpenStreetMap Bridge Loader

Load bridge data from OpenStreetMap Overpass API.
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
class BridgeFeature:
    """Represents a bridge feature"""
    id: str
    geometry: LineString  # Bridge centerline/deck
    bridge_type: str  # "yes", "viaduct", "movable", "cantilever", "covered"
    name: Optional[str] = None
    structure: Optional[str] = None  # "arch", "beam", "suspension", "truss", "cable-stayed"
    material: Optional[str] = None  # "concrete", "steel", "stone", "wood", "composite"
    layer: Optional[int] = None  # Vertical ordering
    width: Optional[float] = None  # Bridge width in meters
    carries: Optional[str] = None  # "highway", "railway", "footway", "cycleway"
    maxweight: Optional[str] = None  # Maximum weight capacity
    attributes: Optional[Dict] = None


class SwissBridgeLoader:
    """
    Load bridge data from OpenStreetMap Overpass API
    
    Converts WGS84 coordinates to EPSG:2056 for integration with Swiss data.
    """
    
    def __init__(
        self,
        timeout: int = 30,
        retry_count: int = 3
    ):
        """
        Initialize bridge loader
        
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
        
        logger.info("SwissBridgeLoader initialized")
    
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
    
    def _determine_carries(self, tags: Dict) -> Optional[str]:
        """Determine what the bridge carries based on tags"""
        if 'highway' in tags:
            return 'highway'
        elif 'railway' in tags:
            return 'railway'
        elif 'footway' in tags or 'pedestrian' in tags:
            return 'footway'
        elif 'cycleway' in tags or 'bicycle' in tags:
            return 'cycleway'
        return None
    
    def _estimate_width(self, tags: Dict, geometry: LineString) -> Optional[float]:
        """Estimate bridge width from tags or geometry"""
        # Try explicit width tag
        width_tag = tags.get('width') or tags.get('bridge:width')
        if width_tag:
            try:
                # Handle formats like "5 m", "5m", "5"
                width_str = str(width_tag).lower().replace('m', '').strip()
                return float(width_str)
            except (ValueError, TypeError):
                pass
        
        # Estimate from highway type
        highway = tags.get('highway', '')
        highway_widths = {
            'motorway': 12.0,
            'trunk': 11.0,
            'primary': 10.0,
            'secondary': 9.0,
            'tertiary': 7.0,
            'residential': 6.0,
            'service': 5.0,
            'footway': 2.0,
            'path': 1.5,
            'cycleway': 2.5,
        }
        if highway in highway_widths:
            return highway_widths[highway]
        
        # Estimate from railway type
        railway = tags.get('railway', '')
        railway_widths = {
            'rail': 5.0,  # Standard gauge track width
            'tram': 2.5,
            'light_rail': 2.5,
            'narrow_gauge': 2.0,
        }
        if railway in railway_widths:
            return railway_widths[railway]
        
        # Default based on length (longer bridges tend to be wider)
        length = geometry.length
        if length > 100:
            return 10.0
        elif length > 50:
            return 7.0
        else:
            return 5.0
    
    def get_bridges_in_bbox(
        self,
        bbox_2056: Tuple[float, float, float, float],
        bridge_types: Optional[List[str]] = None
    ) -> List[BridgeFeature]:
        """
        Get bridges in bounding box
        
        Args:
            bbox_2056: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
            bridge_types: List of bridge types to include (default: all)
                         Options: "yes", "viaduct", "movable", "cantilever", "covered"
        
        Returns:
            List of BridgeFeature objects
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
        query = f'''
[out:json][timeout:{self.timeout}];
(
  way["bridge"]({south},{west},{north},{east});
);
out geom;
'''
        
        logger.info(f"Querying Overpass API for bridges in bbox: {bbox_2056}")
        
        data = self._request_overpass(query)
        if not data:
            return []
        
        bridges = []
        elements = data.get('elements', [])
        
        for element in elements:
            if element.get('type') != 'way':
                continue
            
            tags = element.get('tags', {})
            bridge_type = tags.get('bridge', 'yes')
            
            # Skip if not in requested types
            if bridge_types and bridge_type not in bridge_types:
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
                logger.warning(f"Failed to create LineString for bridge {element.get('id')}: {e}")
                continue
            
            # Estimate width
            width = self._estimate_width(tags, geometry)
            
            # Determine what it carries
            carries = self._determine_carries(tags)
            
            # Parse layer
            layer = self._parse_int(tags.get('layer'))
            
            # Extract attributes
            feature = BridgeFeature(
                id=str(element.get('id', 'unknown')),
                geometry=geometry,
                bridge_type=bridge_type,
                name=tags.get('name'),
                structure=tags.get('bridge:structure'),
                material=tags.get('bridge:material') or tags.get('material'),
                layer=layer,
                width=width,
                carries=carries,
                maxweight=tags.get('maxweight'),
                attributes=tags
            )
            
            bridges.append(feature)
        
        logger.info(f"Retrieved {len(bridges)} bridge features from Overpass API")
        return bridges
    
    def _parse_int(self, value: Optional[str]) -> Optional[int]:
        """Parse integer from string, return None if invalid"""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def get_bridges_around_point(
        self,
        x: float,
        y: float,
        radius: float = 500,
        bridge_types: Optional[List[str]] = None
    ) -> List[BridgeFeature]:
        """
        Get bridges within radius of a point
        
        Args:
            x: X coordinate in EPSG:2056
            y: Y coordinate in EPSG:2056
            radius: Radius in meters
            bridge_types: List of bridge types to include
            
        Returns:
            List of BridgeFeature objects
        """
        # Create bbox
        bbox = (
            x - radius,
            y - radius,
            x + radius,
            y + radius
        )
        
        bridges = self.get_bridges_in_bbox(bbox, bridge_types)
        
        # Filter to circular area
        center = Point(x, y)
        filtered = [
            b for b in bridges
            if b.geometry.distance(center) <= radius
        ]
        
        logger.info(f"Filtered {len(filtered)}/{len(bridges)} bridges within {radius}m radius")
        return filtered
    
    def get_bridges_on_parcel(
        self,
        egrid: str,
        buffer_m: float = 10,
        bridge_types: Optional[List[str]] = None
    ) -> List[BridgeFeature]:
        """
        Get bridges on or near a cadastral parcel
        
        Args:
            egrid: Swiss EGRID identifier
            buffer_m: Buffer around parcel boundary in meters
            bridge_types: List of bridge types to include
            
        Returns:
            List of BridgeFeature objects
        """
        logger.info(f"Fetching bridges for EGRID: {egrid}")
        
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
        
        # Get bridges in bbox
        bridges = self.get_bridges_in_bbox(bbox, bridge_types)
        
        # Filter to bridges that intersect the parcel (with buffer)
        if buffer_m > 0:
            search_area = site_boundary.buffer(buffer_m)
        else:
            search_area = site_boundary
        
        filtered_bridges = [
            b for b in bridges
            if search_area.intersects(b.geometry)
        ]
        
        logger.info(
            f"Found {len(filtered_bridges)} bridges on parcel "
            f"(buffer: {buffer_m}m)"
        )
        
        return filtered_bridges


# Convenience functions

def get_bridges_around_egrid(
    egrid: str,
    buffer_m: float = 10,
    bridge_types: Optional[List[str]] = None
) -> List[BridgeFeature]:
    """
    Get bridges around a cadastral parcel identified by EGRID
    
    Args:
        egrid: Swiss EGRID identifier
        buffer_m: Buffer around parcel boundary
        bridge_types: List of bridge types to include
        
    Returns:
        List of BridgeFeature objects
    """
    loader = SwissBridgeLoader()
    return loader.get_bridges_on_parcel(egrid, buffer_m, bridge_types)


def get_bridges_in_bbox(
    bbox_2056: Tuple[float, float, float, float],
    bridge_types: Optional[List[str]] = None
) -> List[BridgeFeature]:
    """
    Get bridges in bounding box
    
    Args:
        bbox_2056: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
        bridge_types: List of bridge types to include
        
    Returns:
        List of BridgeFeature objects
    """
    loader = SwissBridgeLoader()
    return loader.get_bridges_in_bbox(bbox_2056, bridge_types)


if __name__ == "__main__":
    """Example usage"""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Example: Get bridges in bbox (Zurich HB area)
    print("\n" + "="*80)
    print("Example: Get bridges in bounding box (Zurich HB)")
    print("="*80)
    
    bbox = (2682500, 1247500, 2683500, 1248500)  # 500m x 500m
    try:
        bridges = get_bridges_in_bbox(bbox)
        print(f"\nFound {len(bridges)} bridges")
        for b in bridges[:5]:
            print(f"  - {b.name or 'unnamed'} ({b.bridge_type}, carries: {b.carries})")
    except Exception as e:
        print(f"Error: {e}")

