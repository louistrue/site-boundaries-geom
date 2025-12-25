"""
Swiss Building Data Loader

Efficiently load Swiss building footprints and 3D data from various APIs.
Integrates with existing terrain workflow.
"""

import logging
import time
import tempfile
import zipfile
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Literal
from dataclasses import dataclass
from functools import wraps

import requests
from shapely.geometry import shape, box, Polygon, MultiPolygon, Point
from shapely.ops import unary_union

try:
    import fiona
    from fiona import drivers
    FIONA_AVAILABLE = True
except ImportError:
    FIONA_AVAILABLE = False
    logging.warning("fiona not available - 3D building download disabled")

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
    - REST API (MapServer Identify) - best for area queries with polygon geometry
    - STAC API - best for bulk tile downloads of 3D data
    - WFS (Web Feature Service) - NOT AVAILABLE on Swiss servers
    """

    # API Configuration
    WFS_URL = "https://wms.geo.admin.ch/"  # Note: WFS is disabled on this server
    STAC_BASE = "https://data.geo.admin.ch/api/stac/v1"
    REST_BASE = "https://api3.geo.admin.ch/rest/services"

    # Layer names
    BUILDINGS_LAYER = "ch.swisstopo.swissbuildings3d_3_0"
    BUILDINGS_LAYER_BETA = "ch.swisstopo.swissbuildings3d_3_0-beta"
    
    # Working building layers (with polygon geometry)
    BUILDINGS_VEC25 = "ch.swisstopo.vec25-gebaeude"  # Vector 25k building footprints
    BUILDINGS_REGISTER = "ch.bfs.gebaeude_wohnungs_register"  # Building register (point data)

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

    def get_buildings_rest(
        self,
        bbox_2056: Tuple[float, float, float, float],
        max_features: int = 1000
    ) -> List[BuildingFeature]:
        """
        Get buildings using REST API MapServer Identify endpoint
        
        This is the RECOMMENDED method - uses Vector 25k building footprints
        which provide accurate polygon geometry.

        Args:
            bbox_2056: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
            max_features: Maximum number of buildings to return

        Returns:
            List of BuildingFeature objects with polygon geometry
        """
        logger.info(f"Fetching buildings via REST API in bbox: {bbox_2056}")

        url = f"{self.REST_BASE}/api/MapServer/identify"
        params = {
            "geometryType": "esriGeometryEnvelope",
            "geometry": f"{bbox_2056[0]},{bbox_2056[1]},{bbox_2056[2]},{bbox_2056[3]}",
            "layers": f"all:{self.BUILDINGS_VEC25}",
            "mapExtent": f"{bbox_2056[0]},{bbox_2056[1]},{bbox_2056[2]},{bbox_2056[3]}",
            "imageDisplay": "1000,1000,96",
            "tolerance": 0,
            "returnGeometry": "true",
            "geometryFormat": "geojson",
            "sr": "2056"
        }

        try:
            response = self._request_with_retry(url, params)
            data = response.json()

            buildings = []
            for result in data.get("results", []):
                if max_features > 0 and len(buildings) >= max_features:
                    break
                building = self._parse_rest_result(result)
                if building:
                    buildings.append(building)

            logger.info(f"Retrieved {len(buildings)} buildings via REST API")
            return buildings

        except Exception as e:
            logger.error(f"REST API request failed: {e}")
            raise

    def get_buildings_3d(
        self,
        bbox_2056: Tuple[float, float, float, float],
        timeout: int = 60,
        max_tiles: int = 1
    ) -> List[BuildingFeature]:
        """
        Get buildings with real 3D geometry from Swiss GDB tiles
        
        Downloads ~14MB GDB file, parses with Fiona, extracts buildings.
        Expected time: 10-30 seconds depending on network.
        
        Args:
            bbox_2056: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
            timeout: Download timeout in seconds
            max_tiles: Maximum number of tiles to process (default: 1 to save disk space)
            
        Returns:
            List of BuildingFeature objects with 3D geometry and heights
        """
        if not FIONA_AVAILABLE:
            raise ImportError("fiona is required for 3D building download. Install with: pip install fiona")
        
        logger.info(f"Fetching 3D buildings via STAC GDB in bbox: {bbox_2056}")
        
        # Step 1: Query STAC for tiles covering bbox
        bbox_wgs84 = self.bbox_2056_to_wgs84(bbox_2056)
        tiles = self.get_buildings_stac(bbox_2056, limit=max_tiles)
        
        if not tiles:
            logger.warning("No STAC tiles found for bbox")
            return []
        
        logger.info(f"Found {len(tiles)} tiles, processing up to {max_tiles} tiles...")
        
        all_buildings = []
        
        # Step 2: Download and parse each tile
        for tile_idx, tile in enumerate(tiles):
            try:
                # Find GDB asset
                gdb_asset = None
                for asset_name, asset_info in tile.get("assets", {}).items():
                    if "gdb.zip" in asset_name.lower():
                        gdb_asset = asset_info
                        break
                
                if not gdb_asset:
                    logger.warning(f"No GDB asset found in tile {tile.get('id')}")
                    continue
                
                gdb_url = gdb_asset.get("href")
                if not gdb_url:
                    continue
                
                logger.info(f"Processing tile {tile_idx + 1}/{len(tiles)}: {tile.get('id')}")
                logger.info(f"Downloading GDB from {gdb_url[:80]}...")
                download_start = time.time()
                
                # Extract to temp directory (auto-cleans up on exit)
                with tempfile.TemporaryDirectory(prefix="gdb_tile_") as temp_dir:
                    zip_path = os.path.join(temp_dir, "tile.gdb.zip")
                    
                    # Download with streaming to avoid memory issues
                    response = requests.get(gdb_url, timeout=timeout, stream=True)
                    response.raise_for_status()
                    
                    # Get content length for progress
                    total_size = int(response.headers.get('content-length', 0))
                    
                    # Write downloaded content to file
                    downloaded_size = 0
                    with open(zip_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)
                                downloaded_size += len(chunk)
                    
                    download_time = time.time() - download_start
                    size_mb = downloaded_size / 1024 / 1024
                    logger.info(f"Downloaded {size_mb:.1f} MB in {download_time:.1f}s")
                    
                    # Extract zip
                    extract_start = time.time()
                    extract_dir = os.path.join(temp_dir, "gdb")
                    os.makedirs(extract_dir, exist_ok=True)
                    
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    extract_time = time.time() - extract_start
                    logger.info(f"Extracted in {extract_time:.1f}s")
                    
                    # Find GDB directory
                    gdb_dirs = [d for d in os.listdir(extract_dir) if d.endswith(".gdb")]
                    if not gdb_dirs:
                        logger.warning(f"No .gdb directory found in extracted files")
                        continue
                    
                    gdb_path = os.path.join(extract_dir, gdb_dirs[0])
                    
                    # Parse with Fiona
                    parse_start = time.time()
                    buildings = self._parse_gdb_tile(gdb_path, bbox_2056)
                    parse_time = time.time() - parse_start
                    
                    logger.info(f"Parsed {len(buildings)} buildings from tile in {parse_time:.1f}s")
                    all_buildings.extend(buildings)
                    
                    # Temp directory automatically cleaned up here when exiting 'with' block
                    logger.debug(f"Cleaned up temp files for tile {tile.get('id')}")
            
            except Exception as e:
                logger.error(f"Failed to process tile {tile.get('id')}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        logger.info(f"Retrieved {len(all_buildings)} 3D buildings total")
        return all_buildings

    def _parse_gdb_tile(
        self,
        gdb_path: str,
        bbox_2056: Tuple[float, float, float, float]
    ) -> List[BuildingFeature]:
        """
        Parse a GDB tile file and extract buildings within bbox
        
        Args:
            gdb_path: Path to extracted .gdb directory
            bbox_2056: Bounding box to filter buildings
            
        Returns:
            List of BuildingFeature objects with 3D geometry
        """
        buildings = []
        
        try:
            # Open GDB with Fiona
            # GDB typically has layers like "Buildings", "Roof", "Wall", etc.
            layers = fiona.listlayers(gdb_path)
            logger.debug(f"GDB layers: {layers}")
            
            # Try common layer names
            building_layer = None
            for layer_name in ["Buildings", "Building", "Gebaeude", "buildings"]:
                if layer_name in layers:
                    building_layer = layer_name
                    break
            
            if not building_layer and layers:
                # Use first layer if no match
                building_layer = layers[0]
            
            if not building_layer:
                logger.warning(f"No suitable layer found in GDB")
                return []
            
            logger.info(f"Reading layer: {building_layer}")
            
            # Create bbox polygon for filtering
            bbox_poly = box(bbox_2056[0], bbox_2056[1], bbox_2056[2], bbox_2056[3])
            
            # Read features
            with fiona.open(gdb_path, layer=building_layer) as src:
                logger.info(f"CRS: {src.crs}")
                
                for feature in src:
                    try:
                        # Get geometry
                        geom = shape(feature["geometry"])
                        
                        # Filter to bbox
                        if not bbox_poly.intersects(geom):
                            continue
                        
                        # Extract 3D geometry
                        # GDB has MultiPolygonZ or PolygonZ with Z coordinates
                        building = self._extract_building_from_3d_geom(
                            feature, geom, bbox_2056
                        )
                        
                        if building:
                            buildings.append(building)
                    
                    except Exception as e:
                        logger.warning(f"Failed to parse feature: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Failed to parse GDB: {e}")
            raise
        
        return buildings

    def _extract_building_from_3d_geom(
        self,
        feature: Dict,
        geom,
        bbox_2056: Tuple[float, float, float, float]
    ) -> Optional[BuildingFeature]:
        """
        Extract building information from 3D geometry
        
        Args:
            feature: Fiona feature dict
            geom: Shapely geometry (may have Z coordinates)
            bbox_2056: Bounding box
            
        Returns:
            BuildingFeature or None
        """
        try:
            # Get feature ID
            feature_id = str(feature.get("id", feature.get("properties", {}).get("id", "unknown")))
            
            # Extract Z coordinates to calculate height
            z_values = []
            
            if hasattr(geom, "geoms"):
                # MultiPolygon - get all Z values
                for poly in geom.geoms:
                    if hasattr(poly, "exterior"):
                        coords = list(poly.exterior.coords)
                        # Check each coordinate individually for Z value
                        for coord in coords:
                            if len(coord) > 2:
                                z_values.append(coord[2])
            elif hasattr(geom, "exterior"):
                # Polygon - get Z values from exterior
                coords = list(geom.exterior.coords)
                # Check each coordinate individually for Z value
                for coord in coords:
                    if len(coord) > 2:
                        z_values.append(coord[2])
            
            # Calculate height
            height = None
            if z_values:
                height = max(z_values) - min(z_values)
            
            # Extract 2D footprint (project to XY plane)
            if hasattr(geom, "geoms"):
                # MultiPolygon - take largest polygon
                footprint_2d = max(geom.geoms, key=lambda p: p.area if hasattr(p, "area") else 0)
            else:
                footprint_2d = geom
            
            # Convert to 2D polygon
            if hasattr(footprint_2d, "exterior"):
                coords_2d = [(x, y) for x, y, *_ in footprint_2d.exterior.coords]
                footprint = Polygon(coords_2d)
            else:
                # Already 2D
                footprint = footprint_2d
            
            # Validate footprint
            if footprint.is_empty or not footprint.is_valid:
                footprint = footprint.buffer(0)
            if footprint.is_empty:
                return None
            
            # Get properties
            props = feature.get("properties", {})
            
            # Store 3D geometry in attributes for later use in IFC conversion
            # Convert geometry to GeoJSON-like dict for storage
            geom_dict = None
            try:
                # Try to get __geo_interface__ which Shapely geometries provide
                if hasattr(geom, "__geo_interface__"):
                    geom_dict = geom.__geo_interface__
                elif hasattr(geom, "geoms"):
                    # MultiPolygon - manually construct GeoJSON
                    coords_list = []
                    for poly in geom.geoms:
                        if hasattr(poly, "exterior"):
                            poly_coords = list(poly.exterior.coords)
                            if poly_coords:
                                # Check if 3D
                                if len(poly_coords[0]) > 2:
                                    ring_coords = [[float(x), float(y), float(z)] for x, y, z in poly_coords]
                                else:
                                    ring_coords = [[float(x), float(y)] for x, y in poly_coords]
                                coords_list.append([ring_coords])
                    if coords_list:
                        geom_dict = {"type": "MultiPolygon", "coordinates": coords_list}
            except Exception as e:
                logger.debug(f"Could not serialize 3D geometry: {e}")
            
            if geom_dict:
                props["geometry_3d"] = geom_dict
            
            return BuildingFeature(
                id=feature_id,
                geometry=footprint,
                height=height,
                building_class=props.get("building_class") or props.get("type"),
                roof_type=props.get("roof_type") or props.get("dachform"),
                year_built=props.get("year_built") or props.get("baujahr"),
                attributes=props
            )
        
        except Exception as e:
            logger.warning(f"Failed to extract building from 3D geometry: {e}")
            return None

    def _parse_rest_result(self, result: Dict) -> Optional[BuildingFeature]:
        """
        Parse a REST API identify result into BuildingFeature

        Args:
            result: REST API result dict

        Returns:
            BuildingFeature or None if invalid
        """
        try:
            # Parse geometry
            geom_data = result.get("geometry", {})
            if not geom_data:
                return None
                
            geom = shape(geom_data)

            # Handle MultiPolygon by taking the largest polygon
            if geom.geom_type == "MultiPolygon":
                largest = max(geom.geoms, key=lambda p: p.area)
                geom = largest

            # Extract to 2D if 3D
            if geom.has_z:
                geom = Polygon([(x, y) for x, y, *_ in geom.exterior.coords])

            # Extract properties
            attrs = result.get("attributes", {})
            feature_id = str(result.get("id", attrs.get("id", "unknown")))

            return BuildingFeature(
                id=feature_id,
                geometry=geom,
                height=None,  # Vec25 doesn't have height data
                building_class=result.get("layerName"),
                roof_type=None,
                year_built=None,
                attributes=attrs
            )

        except Exception as e:
            logger.warning(f"Failed to parse REST result: {e}")
            return None

    def get_buildings_around_point(
        self,
        x: float,
        y: float,
        radius: float = 500,
        method: Literal["rest", "wfs", "stac"] = "rest"
    ) -> List[BuildingFeature]:
        """
        Get buildings within radius of a point

        Args:
            x: X coordinate in EPSG:2056
            y: Y coordinate in EPSG:2056
            radius: Radius in meters
            method: API method to use ("rest" recommended, "wfs" disabled, "stac" returns tiles)

        Returns:
            List of BuildingFeature objects
        """
        # Create bbox
        bbox = (x - radius, y - radius, x + radius, y + radius)

        if method == "rest":
            buildings = self.get_buildings_rest(bbox)

            # Filter to circular area
            center = shape({"type": "Point", "coordinates": [x, y]})
            filtered = [
                b for b in buildings
                if b.geometry.distance(center) <= radius
            ]

            logger.info(f"Filtered {len(filtered)}/{len(buildings)} buildings within {radius}m radius")
            return filtered

        elif method == "wfs":
            logger.warning("WFS is disabled on Swiss geo.admin.ch - using REST API instead")
            return self.get_buildings_around_point(x, y, radius, method="rest")

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
        from src.terrain_with_site import fetch_boundary_by_egrid

        # Get parcel boundary using existing function
        site_boundary, metadata = fetch_boundary_by_egrid(egrid)
        if site_boundary is None:
            logger.warning(f"No boundary found for EGRID {egrid}")
            return []

        # Create bbox with buffer
        bounds = site_boundary.bounds  # (minx, miny, maxx, maxy)
        bbox = (
            bounds[0] - buffer_m,
            bounds[1] - buffer_m,
            bounds[2] + buffer_m,
            bounds[3] + buffer_m
        )

        # Get buildings in bbox using REST API (WFS is disabled)
        buildings = self.get_buildings_rest(bbox)

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
    method: Literal["rest", "wfs", "stac"] = "rest"
) -> Tuple[List[BuildingFeature], Dict]:
    """
    Get buildings in bounding box

    Args:
        bbox_2056: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
        method: API method ('rest' recommended, 'wfs' disabled, 'stac' returns tiles only)

    Returns:
        Tuple of (buildings list, statistics dict)
    """
    loader = SwissBuildingLoader()

    if method == "rest":
        buildings = loader.get_buildings_rest(bbox_2056)
    elif method == "wfs":
        logger.warning("WFS is disabled on Swiss geo.admin.ch - using REST API instead")
        buildings = loader.get_buildings_rest(bbox_2056)
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
