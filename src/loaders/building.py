"""
CityGML Building Loader

Loads complete 3D buildings from Swiss STAC CityGML tiles.
Provides lod2Solid geometry with walls, roofs, and ground surfaces.

Also supports fallback to FileGDB format for regions without CityGML data
(e.g., Geneva, Lausanne which only have 2020+ GDB data).
"""

import logging
import tempfile
import zipfile
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from defusedxml.ElementTree import parse as safe_parse
    XML_SAFE = True
except ImportError:
    import xml.etree.ElementTree as ET
    safe_parse = ET.parse
    XML_SAFE = False
    logging.warning("defusedxml not available - using standard XML parser (less secure)")

try:
    import fiona
    FIONA_AVAILABLE = True
except ImportError:
    FIONA_AVAILABLE = False
    logging.warning("fiona not available - GDB building fallback disabled")

import requests
from shapely.geometry import Polygon, box, Point, shape
from pyproj import Transformer

logger = logging.getLogger(__name__)


@dataclass
class CityGMLBuilding:
    """Complete 3D building from CityGML lod2Solid"""
    id: str
    faces: List[List[Tuple[float, float, float]]]  # List of polygon faces with XYZ coords
    height_max: Optional[float] = None
    height_min: Optional[float] = None
    building_type: Optional[str] = None
    centroid: Optional[Tuple[float, float]] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    attributes: Optional[Dict] = None


class CityGMLBuildingLoader:
    """
    Load complete 3D buildings from Swiss STAC CityGML tiles
    
    Uses lod2Solid geometry which includes walls, roofs, and ground surfaces.
    """
    
    STAC_BASE = "https://data.geo.admin.ch/api/stac/v1"
    COLLECTION = "ch.swisstopo.swissbuildings3d_3_0"
    
    def __init__(self, timeout: int = 120):
        """
        Initialize CityGML loader
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    
    def get_buildings_in_bbox(
        self,
        bbox_2056: Tuple[float, float, float, float],
        max_tiles: int = 1
    ) -> List[CityGMLBuilding]:
        """
        Get complete 3D buildings in bounding box from CityGML tiles
        
        Args:
            bbox_2056: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
            max_tiles: Maximum number of tiles to process
            
        Returns:
            List of CityGMLBuilding objects with complete 3D geometry
        """
        logger.info(f"Fetching CityGML buildings in bbox: {bbox_2056}")
        
        # Convert bbox to WGS84 for STAC query
        min_lon, min_lat = self.transformer.transform(bbox_2056[0], bbox_2056[1])
        max_lon, max_lat = self.transformer.transform(bbox_2056[2], bbox_2056[3])
        bbox_wgs84 = (min_lon, min_lat, max_lon, max_lat)
        
        # Query STAC for tiles - request more than needed since some may not have CityGML
        url = f"{self.STAC_BASE}/collections/{self.COLLECTION}/items"
        # Request 3x more tiles to account for older tiles without CityGML assets
        query_limit = max(max_tiles * 3, 10)
        params = {'bbox': ','.join(map(str, bbox_wgs84)), 'limit': query_limit}
        
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        tiles = response.json().get('features', [])
        if not tiles:
            logger.warning("No STAC tiles found for bbox")
            return []
        
        # Filter to tiles that have CityGML assets (newer tiles)
        citygml_tiles = []
        for tile in tiles:
            has_citygml = any('citygml' in name.lower() for name in tile.get('assets', {}).keys())
            if has_citygml:
                citygml_tiles.append(tile)
        
        if not citygml_tiles:
            logger.info(f"Found {len(tiles)} tiles but none have CityGML assets - trying GDB fallback")
            # Fall back to GDB parsing for regions without CityGML (e.g., Geneva, Lausanne 2020+)
            return self._get_buildings_from_gdb(tiles, bbox_2056, max_tiles)
        
        logger.info(f"Found {len(citygml_tiles)} CityGML tiles (of {len(tiles)} total), processing up to {max_tiles}...")
        
        all_buildings = []
        processed_count = 0
        
        for tile in citygml_tiles:
            if processed_count >= max_tiles:
                break
            try:
                buildings = self._process_tile(tile, bbox_2056)
                all_buildings.extend(buildings)
                processed_count += 1
                logger.info(f"Processed tile {processed_count}/{min(len(citygml_tiles), max_tiles)}: {len(buildings)} buildings")
            except Exception as e:
                logger.error(f"Failed to process tile {tile.get('id')}: {e}")
                continue
        
        logger.info(f"Retrieved {len(all_buildings)} buildings from CityGML")
        return all_buildings
    
    def _process_tile(
        self,
        tile: Dict,
        bbox_2056: Tuple[float, float, float, float]
    ) -> List[CityGMLBuilding]:
        """
        Download and parse a CityGML tile
        
        Args:
            tile: STAC tile feature
            bbox_2056: Target bounding box for filtering
            
        Returns:
            List of buildings in the target area
        """
        # Find CityGML asset
        citygml_url = None
        for name, asset in tile.get('assets', {}).items():
            if 'citygml' in name.lower():
                citygml_url = asset['href']
                break
        
        if not citygml_url:
            raise ValueError(f"No CityGML asset found in tile {tile.get('id')}")
        
        # Download and extract
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, 'buildings.zip')
            
            response = requests.get(citygml_url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Safe extraction with path validation to prevent zip slip attacks
            with zipfile.ZipFile(zip_path, 'r') as zf:
                tmpdir_path = Path(tmpdir).resolve()
                for member in zf.namelist():
                    # Skip directories
                    if member.endswith('/'):
                        continue
                    
                    # Compute destination path
                    dest_path = (tmpdir_path / member).resolve()
                    
                    # Verify destination is inside tmpdir (prevent path traversal)
                    try:
                        dest_path.relative_to(tmpdir_path)
                    except ValueError:
                        # Path is outside tmpdir - skip this member
                        logger.warning(f"Skipping suspicious zip member: {member}")
                        continue
                    
                    # Create parent directories if needed
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Extract file safely
                    with zf.open(member) as source:
                        with open(dest_path, 'wb') as target:
                            target.write(source.read())
            
            # Find GML file (may be in subdirectory)
            gml_file = None
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith('.gml'):
                        gml_file = os.path.join(root, f)
                        break
                if gml_file:
                    break
            
            if not gml_file:
                raise ValueError("No GML file found in extracted archive")
            
            # Parse CityGML
            return self._parse_citygml(gml_file, bbox_2056)
    
    def _parse_citygml(
        self,
        gml_path: str,
        bbox_2056: Tuple[float, float, float, float]
    ) -> List[CityGMLBuilding]:
        """
        Parse CityGML file and extract buildings with lod2Solid
        
        Args:
            gml_path: Path to CityGML file
            bbox_2056: Bounding box for filtering (EPSG:2056)
            
        Returns:
            List of CityGMLBuilding objects
        """
        tree = safe_parse(gml_path)
        root = tree.getroot()
        
        # Define namespaces
        ns = {
            'bldg': 'http://www.opengis.net/citygml/building/2.0',
            'gml': 'http://www.opengis.net/gml',
            'gen': 'http://www.opengis.net/citygml/generics/2.0'
        }
        
        all_buildings_elem = root.findall('.//bldg:Building', ns)
        logger.debug(f"Found {len(all_buildings_elem)} buildings in CityGML file")
        
        target_box = box(*bbox_2056)
        buildings = []
        
        for b_elem in all_buildings_elem:
            # Check for lod2Solid
            lod2solid = b_elem.find('.//bldg:lod2Solid', ns)
            if lod2solid is None:
                continue
            
            solid = lod2solid.find('.//gml:Solid', ns)
            if solid is None:
                continue
            
            bldg_id = b_elem.get('{http://www.opengis.net/gml}id', 'unknown')
            
            # Extract attributes
            dach_max = b_elem.find('.//gen:doubleAttribute[@name="DACH_MAX"]/gen:value', ns)
            dach_min = b_elem.find('.//gen:doubleAttribute[@name="DACH_MIN"]/gen:value', ns)
            objektart = b_elem.find('.//gen:stringAttribute[@name="OBJEKTART"]/gen:value', ns)
            
            height_max = float(dach_max.text) if dach_max is not None else None
            height_min = float(dach_min.text) if dach_min is not None else None
            building_type = objektart.text if objektart is not None else None
            
            # Extract all attributes
            attrs = {}
            for attr in b_elem.findall('.//gen:*', ns):
                name = attr.get('name')
                value_elem = attr.find('gen:value', ns)
                if name and value_elem is not None:
                    attrs[name] = value_elem.text
            
            # Extract geometry faces
            faces_data = []
            all_points = []
            
            comp_surface = solid.find('.//gml:CompositeSurface', ns)
            if comp_surface is None:
                continue
            
            for surf_member in comp_surface.findall('.//gml:surfaceMember', ns):
                polygon = surf_member.find('.//gml:Polygon', ns)
                if polygon is None:
                    continue
                
                poslist = polygon.find('.//gml:posList', ns)
                if poslist is None:
                    continue
                
                coords_text = poslist.text.strip()
                coords_list = [float(x) for x in coords_text.split()]
                
                # Group into XYZ triples
                points = [(coords_list[i], coords_list[i+1], coords_list[i+2]) 
                         for i in range(0, len(coords_list), 3)]
                
                if len(points) >= 3:
                    faces_data.append(points)
                    all_points.extend(points)
            
            if not faces_data:
                continue
            
            # Calculate bounding box and centroid
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            zs = [p[2] for p in all_points]
            
            centroid_x = sum(xs) / len(xs)
            centroid_y = sum(ys) / len(ys)
            
            # Check if building intersects target area
            building_bbox = box(min(xs), min(ys), max(xs), max(ys))
            if not target_box.intersects(building_bbox):
                continue
            
            buildings.append(CityGMLBuilding(
                id=bldg_id,
                faces=faces_data,
                height_max=height_max,
                height_min=height_min,
                building_type=building_type,
                centroid=(centroid_x, centroid_y),
                z_min=min(zs),
                z_max=max(zs),
                attributes=attrs
            ))
        
        return buildings
    
    def _get_buildings_from_gdb(
        self,
        tiles: List[Dict],
        bbox_2056: Tuple[float, float, float, float],
        max_tiles: int
    ) -> List[CityGMLBuilding]:
        """
        Fallback: Get buildings from GDB tiles when CityGML is not available.
        
        This is used for regions like Geneva and Lausanne which only have
        2020+ data in FileGDB format (no CityGML).
        
        Args:
            tiles: List of STAC tile features
            bbox_2056: Bounding box for filtering
            max_tiles: Maximum number of tiles to process
            
        Returns:
            List of CityGMLBuilding objects with 3D geometry from GDB
        """
        if not FIONA_AVAILABLE:
            logger.warning("fiona not available - cannot parse GDB files. Install with: pip install fiona")
            return []
        
        # Filter to tiles that have GDB assets
        gdb_tiles = []
        for tile in tiles:
            assets = tile.get('assets', {})
            # Prefer older versioned tiles (e.g., 2020_1301-13) over yearly aggregates (2023, 2024)
            # as they have more detailed tile coverage
            gdb_asset = None
            for name, asset in assets.items():
                if 'gdb.zip' in name.lower():
                    gdb_asset = asset
                    break
            if gdb_asset:
                gdb_tiles.append((tile, gdb_asset))
        
        if not gdb_tiles:
            logger.warning("No GDB tiles found")
            return []
        
        # Sort to prefer older regional tiles over yearly Swiss-wide tiles
        # Regional tiles like "2020_1301-13" have better coverage for specific areas
        def tile_sort_key(item):
            tile_id = item[0].get('id', '')
            # Yearly tiles like "2023", "2024" should be last
            if tile_id.count('_') == 2:  # e.g., swissbuildings3d_3_0_2023
                return (1, tile_id)
            return (0, tile_id)
        
        gdb_tiles.sort(key=tile_sort_key)
        
        logger.info(f"Found {len(gdb_tiles)} GDB tiles, processing up to {max_tiles}...")
        print(f"  Processing {min(len(gdb_tiles), max_tiles)} GDB tile(s) (this may take 30-60 seconds per tile)...", flush=True)
        
        all_buildings = []
        processed_count = 0
        
        for tile, gdb_asset in gdb_tiles:
            if processed_count >= max_tiles:
                break
            try:
                print(f"  Downloading tile {processed_count + 1}/{min(len(gdb_tiles), max_tiles)}...", flush=True)
                buildings = self._process_gdb_tile(tile, gdb_asset, bbox_2056)
                all_buildings.extend(buildings)
                processed_count += 1
                logger.info(f"Processed GDB tile {processed_count}/{min(len(gdb_tiles), max_tiles)}: {len(buildings)} buildings")
                print(f"  ✓ Loaded {len(buildings)} buildings from tile {processed_count}", flush=True)
            except Exception as e:
                logger.error(f"Failed to process GDB tile {tile.get('id')}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        logger.info(f"Retrieved {len(all_buildings)} buildings from GDB")
        return all_buildings
    
    def _process_gdb_tile(
        self,
        tile: Dict,
        gdb_asset: Dict,
        bbox_2056: Tuple[float, float, float, float]
    ) -> List[CityGMLBuilding]:
        """
        Download and parse a GDB tile to extract 3D buildings.
        
        Args:
            tile: STAC tile feature
            gdb_asset: GDB asset info with download URL
            bbox_2056: Bounding box for filtering
            
        Returns:
            List of CityGMLBuilding objects
        """
        gdb_url = gdb_asset.get('href')
        if not gdb_url:
            raise ValueError(f"No GDB URL in asset for tile {tile.get('id')}")
        
        tile_id = tile.get('id', 'unknown')
        logger.info(f"Downloading GDB tile: {tile_id}")
        print(f"    Downloading {tile_id}...", flush=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, 'buildings.gdb.zip')
            
            response = requests.get(gdb_url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Track download size
            downloaded = 0
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            size_mb = downloaded / 1024 / 1024
            logger.info(f"Downloaded {size_mb:.1f} MB")
            print(f"    Downloaded {size_mb:.1f} MB, extracting...", flush=True)
            
            # Safe extraction with path validation
            with zipfile.ZipFile(zip_path, 'r') as zf:
                tmpdir_path = Path(tmpdir).resolve()
                for member in zf.namelist():
                    if member.endswith('/'):
                        continue
                    dest_path = (tmpdir_path / member).resolve()
                    try:
                        dest_path.relative_to(tmpdir_path)
                    except ValueError:
                        logger.warning(f"Skipping suspicious zip member: {member}")
                        continue
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as source:
                        with open(dest_path, 'wb') as target:
                            target.write(source.read())
            
            # Find GDB directory
            gdb_path = None
            for root, dirs, files in os.walk(tmpdir):
                for d in dirs:
                    if d.endswith('.gdb'):
                        gdb_path = os.path.join(root, d)
                        break
                if gdb_path:
                    break
            
            # Also check top level
            if not gdb_path:
                for item in os.listdir(tmpdir):
                    if item.endswith('.gdb'):
                        gdb_path = os.path.join(tmpdir, item)
                        break
            
            if not gdb_path:
                raise ValueError("No .gdb directory found in extracted archive")
            
            return self._parse_gdb(gdb_path, bbox_2056)
    
    def _parse_gdb(
        self,
        gdb_path: str,
        bbox_2056: Tuple[float, float, float, float]
    ) -> List[CityGMLBuilding]:
        """
        Parse GDB file and extract 3D buildings.
        
        Reads the Building_solid layer which contains complete 3D building geometry.
        
        Args:
            gdb_path: Path to .gdb directory
            bbox_2056: Bounding box for filtering
            
        Returns:
            List of CityGMLBuilding objects
        """
        layers = fiona.listlayers(gdb_path)
        logger.debug(f"GDB layers: {layers}")
        
        # Prefer Building_solid for complete 3D geometry
        target_layer = None
        for layer_name in ['Building_solid', 'Roof_solid', 'Wall']:
            if layer_name in layers:
                target_layer = layer_name
                break
        
        if not target_layer:
            logger.warning(f"No suitable building layer found in GDB. Available: {layers}")
            return []
        
            logger.info(f"Parsing GDB layer: {target_layer}")
            print(f"    Parsing {target_layer} layer...", flush=True)
        
        target_box = box(*bbox_2056)
        buildings = []
        
        with fiona.open(gdb_path, layer=target_layer) as src:
            for feat in src:
                try:
                    # Get geometry
                    geom_data = feat['geometry']
                    if not geom_data:
                        continue
                    
                    # Extract 3D coordinates from MultiPolygon
                    coords = geom_data.get('coordinates', [])
                    if not coords:
                        continue
                    
                    # Extract faces from MultiPolygon 3D geometry
                    faces = []
                    all_points = []
                    
                    for polygon_coords in coords:
                        # Each polygon has an exterior ring (and possibly holes)
                        if not polygon_coords:
                            continue
                        
                        exterior_ring = polygon_coords[0]  # First ring is exterior
                        if not exterior_ring or len(exterior_ring) < 3:
                            continue
                        
                        # Check if coordinates are 3D
                        if len(exterior_ring[0]) >= 3:
                            # 3D coordinates: (x, y, z)
                            face_points = [(float(c[0]), float(c[1]), float(c[2])) for c in exterior_ring]
                        else:
                            # 2D only - skip or use height from attributes
                            continue
                        
                        if len(face_points) >= 3:
                            faces.append(face_points)
                            all_points.extend(face_points)
                    
                    if not faces or not all_points:
                        continue
                    
                    # Calculate bounds
                    xs = [p[0] for p in all_points]
                    ys = [p[1] for p in all_points]
                    zs = [p[2] for p in all_points]
                    
                    # Check if building intersects target area
                    building_bbox = box(min(xs), min(ys), max(xs), max(ys))
                    if not target_box.intersects(building_bbox):
                        continue
                    
                    # Extract properties
                    props = feat.get('properties', {})
                    building_id = str(props.get('UUID') or props.get('EGID') or feat.get('id', 'unknown'))
                    
                    centroid_x = sum(xs) / len(xs)
                    centroid_y = sum(ys) / len(ys)
                    
                    buildings.append(CityGMLBuilding(
                        id=building_id,
                        faces=faces,
                        height_max=props.get('DACH_MAX'),
                        height_min=props.get('DACH_MIN'),
                        building_type=props.get('OBJEKTART'),
                        centroid=(centroid_x, centroid_y),
                        z_min=min(zs),
                        z_max=max(zs),
                        attributes=props
                    ))
                    
                except Exception as e:
                    logger.debug(f"Failed to parse GDB feature: {e}")
                    continue
        
        return buildings


def get_citygml_buildings_in_bbox(
    bbox_2056: Tuple[float, float, float, float],
    max_tiles: int = 1
) -> List[CityGMLBuilding]:
    """
    Convenience function to get CityGML buildings in bounding box
    
    Args:
        bbox_2056: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
        max_tiles: Maximum number of tiles to process
        
    Returns:
        List of CityGMLBuilding objects
    """
    loader = CityGMLBuildingLoader()
    return loader.get_buildings_in_bbox(bbox_2056, max_tiles=max_tiles)

