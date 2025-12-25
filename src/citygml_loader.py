"""
CityGML Building Loader

Loads complete 3D buildings from Swiss STAC CityGML tiles.
Provides lod2Solid geometry with walls, roofs, and ground surfaces.
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

import requests
from shapely.geometry import Polygon, box, Point
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
        
        # Query STAC for tiles
        url = f"{self.STAC_BASE}/collections/{self.COLLECTION}/items"
        params = {'bbox': ','.join(map(str, bbox_wgs84)), 'limit': max_tiles}
        
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        tiles = response.json().get('features', [])
        if not tiles:
            logger.warning("No STAC tiles found for bbox")
            return []
        
        logger.info(f"Found {len(tiles)} tiles, processing up to {max_tiles}...")
        
        all_buildings = []
        
        for tile_idx, tile in enumerate(tiles[:max_tiles]):
            try:
                buildings = self._process_tile(tile, bbox_2056)
                all_buildings.extend(buildings)
                logger.info(f"Processed tile {tile_idx + 1}/{len(tiles[:max_tiles])}: {len(buildings)} buildings")
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
            
            # Find GML file
            gml_file = None
            for f in os.listdir(tmpdir):
                if f.endswith('.gml'):
                    gml_file = os.path.join(tmpdir, f)
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

