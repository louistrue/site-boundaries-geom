"""
Swiss Water Network Loader

Efficiently load Swiss water network data (creeks, rivers, lakes) from geo.admin.ch APIs.
Uses swissTLM3D layer which provides actual polygon geometries for water surfaces (real widths).
"""

import logging
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass

import requests
from shapely.geometry import LineString, Polygon, shape

logger = logging.getLogger(__name__)

# REST API endpoint
REST_API_URL = "https://api3.geo.admin.ch/rest/services/all/MapServer/identify"

# swissTLM3D water layer - provides ACTUAL polygon geometries for water surfaces
# objektart values:
#   4 = Fliessgewässer (river/stream line)
#   6 = Fliessgewässer unterirdisch (underground stream)
#   101 = Wasserfläche (water surface polygon - REAL GEOMETRY!)
WATER_TLM3D_LAYER = "ch.swisstopo.swisstlm3d-gewaessernetz"

# Fallback layer (lines only, need buffering)
WATER_NETWORK_LAYER = "ch.swisstopo.vec25-gewaessernetz_referenz"

# Water type mappings for fallback layer
WATER_TYPE_MAP = {
    "Bach": "creek",
    "Bach_U": "underground_creek",
    "Bachachs": "creek_axis",
    "Fluss": "river",
    "Seeachse": "lake_axis",
    "See": "lake"
}

# Default widths by type (meters) - only used for line geometries as fallback
WATER_WIDTHS = {
    "creek": 3.0,
    "underground_creek": 2.0,
    "creek_axis": 3.0,
    "river": 25.0,  # Increased default for rivers
    "lake_axis": 15.0,
    "lake": None
}

# Known major Swiss rivers with approximate widths (meters)
# Based on actual measurements at typical urban locations
MAJOR_RIVER_WIDTHS = {
    # Major rivers
    "rhein": 150.0,
    "rhine": 150.0,
    "aare": 80.0,
    "aar": 80.0,
    "limmat": 50.0,
    "reuss": 60.0,
    "sihl": 25.0,
    "rhone": 100.0,
    "rhône": 100.0,
    "saane": 40.0,
    "sarine": 40.0,
    "thur": 50.0,
    "birs": 30.0,
    "emme": 40.0,
    "linth": 35.0,
    "glatt": 20.0,
    "töss": 25.0,
    "broye": 25.0,
    "orbe": 20.0,
    "venoge": 15.0,
    "arve": 50.0,
    "doubs": 40.0,
    "inn": 50.0,
    "ticino": 60.0,
    "maggia": 40.0,
    # Medium rivers
    "lorze": 15.0,
    "kleine emme": 25.0,
    "wigger": 12.0,
    "suhre": 12.0,
    "wynige": 10.0,
    "sense": 25.0,
}


@dataclass
class WaterFeature:
    """Represents a water feature (creek, river, lake)."""
    id: str
    geometry: Union[LineString, Polygon]  # LineString for streams/rivers, Polygon for lakes
    water_type: str  # "creek", "river", "lake", etc.
    name: Optional[str] = None
    gewiss_number: Optional[int] = None  # GEWISS identifier
    width: Optional[float] = None  # Width in meters (for buffering)
    is_underground: bool = False
    attributes: Optional[Dict] = None


class SwissWaterLoader:
    """Load water network data from Swiss geo.admin.ch REST API."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SiteBoundariesGeom/1.0 (Water Data Loader)"
        })
    
    def get_water_in_bounds(
        self,
        bounds: Tuple[float, float, float, float],
        fetch_elevations_func=None,
        max_features: int = 500
    ) -> List[WaterFeature]:
        """
        Get water features within bounds using REST API identify.
        
        Uses swissTLM3D layer which provides ACTUAL polygon geometries for water surfaces,
        giving real widths instead of estimated ones.
        
        Args:
            bounds: (minx, miny, maxx, maxy) in EPSG:2056
            fetch_elevations_func: Optional function to fetch elevations
            max_features: Maximum features to return
        
        Returns:
            List of WaterFeature objects
        """
        minx, miny, maxx, maxy = bounds
        
        # Expand map extent for better coverage
        extent_buffer = 1000
        map_extent = f"{minx-extent_buffer},{miny-extent_buffer},{maxx+extent_buffer},{maxy+extent_buffer}"
        
        # First try swissTLM3D layer (has real polygon geometries)
        features = self._fetch_from_tlm3d(minx, miny, maxx, maxy, map_extent, max_features)
        
        # Fall back to vec25 layer if no results
        if not features:
            features = self._fetch_from_vec25(minx, miny, maxx, maxy, map_extent, max_features)
        
        # Fetch elevations if function provided
        if features and fetch_elevations_func:
            print(f"  Fetching elevations for {len(features)} water features...")
            coords = []
            for f in features:
                if isinstance(f.geometry, LineString):
                    midpoint = f.geometry.interpolate(0.5, normalized=True)
                    coords.append((midpoint.x, midpoint.y))
                elif isinstance(f.geometry, Polygon):
                    coords.append((f.geometry.centroid.x, f.geometry.centroid.y))
            
            if coords:
                elevations = fetch_elevations_func(coords)
                for feature, elev in zip(features, elevations):
                    if feature.attributes is None:
                        feature.attributes = {}
                    feature.attributes['elevation'] = elev
        
        return features
    
    def _fetch_from_tlm3d(self, minx, miny, maxx, maxy, map_extent, max_features) -> List[WaterFeature]:
        """Fetch water from swissTLM3D layer (real polygon geometries)."""
        params = {
            "geometry": f"{minx},{miny},{maxx},{maxy}",
            "geometryType": "esriGeometryEnvelope",
            "layers": f"all:{WATER_TLM3D_LAYER}",
            "mapExtent": map_extent,
            "imageDisplay": "1000,1000,96",
            "tolerance": 0,
            "returnGeometry": "true",
            "sr": "2056",
        }
        
        try:
            response = self.session.get(REST_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            
            # Separate polygon (real surface) and line features
            polygon_results = [r for r in results if 'rings' in r.get('geometry', {})]
            line_results = [r for r in results if 'paths' in r.get('geometry', {})]
            
            print(f"  swissTLM3D: {len(polygon_results)} polygon surfaces, {len(line_results)} line features")
            
            features = []
            # Prioritize polygon features (real water surface geometry)
            for result in polygon_results[:max_features]:
                feature = self._parse_tlm3d_result(result, is_polygon=True)
                if feature:
                    features.append(feature)
            
            # Add line features if we have room (for small streams)
            remaining = max_features - len(features)
            if remaining > 0:
                for result in line_results[:remaining]:
                    feature = self._parse_tlm3d_result(result, is_polygon=False)
                    if feature:
                        features.append(feature)
            
            return features
            
        except Exception as e:
            logger.error(f"swissTLM3D query failed: {e}")
            return []
    
    def _fetch_from_vec25(self, minx, miny, maxx, maxy, map_extent, max_features) -> List[WaterFeature]:
        """Fallback: fetch from vec25 layer (line geometries only)."""
        params = {
            "geometry": f"{minx},{miny},{maxx},{maxy}",
            "geometryType": "esriGeometryEnvelope",
            "layers": f"all:{WATER_NETWORK_LAYER}",
            "mapExtent": map_extent,
            "imageDisplay": "1000,1000,96",
            "tolerance": 0,
            "returnGeometry": "true",
            "sr": "2056",
        }
        
        try:
            response = self.session.get(REST_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            print(f"  vec25 fallback: {len(results)} features")
            
            features = []
            for result in results[:max_features]:
                feature = self._parse_result(result)
                if feature:
                    features.append(feature)
            
            return features
            
        except Exception as e:
            logger.error(f"vec25 query failed: {e}")
            return []
    
    def _parse_tlm3d_result(self, result: Dict, is_polygon: bool) -> Optional[WaterFeature]:
        """Parse a swissTLM3D result into a WaterFeature."""
        try:
            geom_data = result.get("geometry", {})
            attrs = result.get("attributes", {})
            feature_id = str(result.get("id", "unknown"))
            
            geom = None
            if is_polygon and 'rings' in geom_data:
                rings = geom_data["rings"]
                if rings and len(rings[0]) >= 3:
                    coords = [(p[0], p[1]) for p in rings[0]]
                    geom = Polygon(coords)
            elif 'paths' in geom_data:
                paths = geom_data["paths"]
                if paths and len(paths[0]) >= 2:
                    coords = [(p[0], p[1]) for p in paths[0]]
                    geom = LineString(coords)
            
            if geom is None:
                return None
            
            # Determine water type from objektart
            objektart = attrs.get("objektart")
            if objektart == 101:
                water_type = "water_surface"  # Actual polygon surface
            elif objektart == 6:
                water_type = "underground_creek"
            elif objektart == 4:
                water_type = "river"
            else:
                water_type = "creek"
            
            name = attrs.get("name", "").strip() or None
            gwl_nr = attrs.get("gwl_nr")
            
            # For polygon surfaces, no width needed (we have real geometry)
            # For lines, look up width by river name first, then use default
            width = None
            if not is_polygon:
                # Check if this is a known major river
                if name:
                    name_lower = name.lower().strip()
                    for river_name, river_width in MAJOR_RIVER_WIDTHS.items():
                        if river_name in name_lower or name_lower in river_name:
                            width = river_width
                            break
                # Fall back to type-based width
                if width is None:
                    width = WATER_WIDTHS.get(water_type, 5.0)
            
            return WaterFeature(
                id=feature_id,
                geometry=geom,
                water_type=water_type,
                name=name,
                gewiss_number=gwl_nr,
                width=width,
                is_underground=(water_type == "underground_creek"),
                attributes=attrs
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse TLM3D result: {e}")
            return None
    
    def _parse_result(self, result: Dict) -> Optional[WaterFeature]:
        """Parse a REST API result into a WaterFeature."""
        try:
            geom_data = result.get("geometry", {})
            attrs = result.get("attributes", {})
            feature_id = str(result.get("id", "unknown"))
            
            # Parse geometry - handle ESRI format (paths for lines, rings for polygons)
            geom = None
            if "paths" in geom_data:
                # LineString geometry (streams/rivers)
                paths = geom_data["paths"]
                if paths and len(paths) > 0:
                    # Use first path (most common case)
                    coords = [(p[0], p[1]) for p in paths[0]]
                    if len(coords) >= 2:
                        geom = LineString(coords)
            elif "rings" in geom_data:
                # Polygon geometry (lakes)
                rings = geom_data["rings"]
                if rings and len(rings) > 0:
                    # Use first ring (exterior)
                    coords = [(p[0], p[1]) for p in rings[0]]
                    if len(coords) >= 3:
                        geom = Polygon(coords)
            else:
                # Try shapely shape() as fallback
                try:
                    geom = shape(geom_data)
                except Exception:
                    pass
            
            if geom is None:
                logger.debug(f"Could not parse geometry for water feature {feature_id}")
                return None
            
            # Get water type
            obj_val = attrs.get("objectval", "")
            water_type = WATER_TYPE_MAP.get(obj_val, "creek")  # Default to creek
            
            # Determine if underground
            is_underground = water_type == "underground_creek"
            
            # Get name
            name = attrs.get("name", "").strip() or None
            
            # Get GEWISS number
            gewiss_nr = attrs.get("gewissnr")
            if gewiss_nr is not None:
                try:
                    gewiss_nr = int(gewiss_nr)
                except (ValueError, TypeError):
                    gewiss_nr = None
            
            # Get width - check known rivers first, then use type default
            width = None
            if name:
                name_lower = name.lower().strip()
                for river_name, river_width in MAJOR_RIVER_WIDTHS.items():
                    if river_name in name_lower or name_lower in river_name:
                        width = river_width
                        break
            if width is None:
                width = WATER_WIDTHS.get(water_type, 5.0)
            
            # Validate geometry type
            if not isinstance(geom, (LineString, Polygon)):
                logger.debug(f"Unsupported geometry type: {type(geom)}")
                return None
            
            # Convert lakes to polygons if needed
            if water_type == "lake" and isinstance(geom, LineString):
                # If we got a line for a lake, buffer it slightly
                geom = geom.buffer(5.0)  # 5m buffer for lake representation
            
            return WaterFeature(
                id=feature_id,
                geometry=geom,
                water_type=water_type,
                name=name,
                gewiss_number=gewiss_nr,
                width=width,
                is_underground=is_underground,
                attributes=attrs
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse result: {e}")
            return None
    
    def get_water_statistics(self, features: List[WaterFeature]) -> Dict:
        """Calculate statistics for water features."""
        if not features:
            return {
                "count": 0,
                "total_length_m": 0,
                "by_type": {}
            }
        
        total_length = 0
        by_type = {}
        
        for feature in features:
            # Count by type
            wtype = feature.water_type
            by_type[wtype] = by_type.get(wtype, 0) + 1
            
            # Calculate length
            if isinstance(feature.geometry, LineString):
                total_length += feature.geometry.length
            elif isinstance(feature.geometry, Polygon):
                total_length += feature.geometry.exterior.length
        
        return {
            "count": len(features),
            "total_length_m": total_length,
            "by_type": by_type
        }


def get_water_around_bounds(
    bounds: Tuple[float, float, float, float],
    fetch_elevations_func=None
) -> List[WaterFeature]:
    """
    Convenience function to get water features in an area.
    
    Args:
        bounds: (minx, miny, maxx, maxy) in EPSG:2056
        fetch_elevations_func: Optional elevation function
    
    Returns:
        List of WaterFeature objects
    """
    loader = SwissWaterLoader()
    return loader.get_water_in_bounds(bounds, fetch_elevations_func)

