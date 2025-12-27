"""
Swiss Water Loader - Combined FOEN Raster + swissTLM3D Lakes

- Rivers/streams: Extracted from FOEN Overland Flow Map raster (real shapes)
- Lakes: Fetched from swissTLM3D water network layer (polygon data)
"""

import logging
from typing import Optional, List
from dataclasses import dataclass

import requests
import numpy as np
from PIL import Image
from io import BytesIO
from shapely.geometry import Polygon, MultiPolygon, shape, box
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# FOEN Overland Flow Map WMS - river/stream shapes
FOEN_WMS_URL = "https://wms.geo.admin.ch/"
FOEN_WATER_LAYER = "ch.bafu.gefaehrdungskarte-oberflaechenabfluss"

# swissTLM3D Water Network - lakes and ponds
TLM3D_API_URL = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
TLM3D_WATER_LAYER = "ch.swisstopo.swisstlm3d-gewaessernetz"


@dataclass
class WaterFeature:
    """Represents a water feature (river, lake)."""
    id: str
    geometry: Polygon
    water_type: str
    name: Optional[str] = None
    gewiss_number: Optional[str] = None
    width: Optional[float] = None
    is_underground: bool = False
    attributes: Optional[dict] = None


class SwissWaterLoader:
    """Load water features from FOEN raster (rivers) + swissTLM3D (lakes)."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SwissSiteModel/1.0"
        })
    
    def get_water_in_bounds(
        self,
        bounds: tuple,
        fetch_elevations_func=None,
        max_features: int = 50,
        resolution: float = 2.0
    ) -> List[WaterFeature]:
        """
        Get water features within bounds.
        
        Combines:
        - Rivers/streams from FOEN raster (accurate shapes)
        - Lakes from swissTLM3D vector data
        """
        minx, miny, maxx, maxy = bounds
        
        # Fetch from both sources
        print("  Fetching FOEN water raster (rivers)...")
        river_features = self._extract_rivers_from_raster(
            minx - 100, miny - 100, maxx + 100, maxy + 100, resolution
        )
        print(f"    Found {len(river_features)} river polygons")
        
        print("  Fetching swissTLM3D lakes...")
        lake_features = self._fetch_lakes_from_tlm3d(bounds)
        print(f"    Found {len(lake_features)} lakes")
        
        # Combine: lakes first (they take priority), then rivers
        all_features = lake_features + river_features
        
        # Remove river features that overlap with lakes
        if lake_features and river_features:
            lake_union = unary_union([f.geometry for f in lake_features])
            filtered_rivers = []
            for rf in river_features:
                try:
                    # Keep river if it doesn't overlap significantly with lakes
                    overlap = rf.geometry.intersection(lake_union).area
                    if overlap < rf.geometry.area * 0.5:
                        # Subtract lake area from river
                        diff = rf.geometry.difference(lake_union)
                        if not diff.is_empty and diff.area > 100:
                            if diff.geom_type == 'MultiPolygon':
                                diff = max(diff.geoms, key=lambda g: g.area)
                            if diff.geom_type == 'Polygon':
                                rf.geometry = diff
                                filtered_rivers.append(rf)
                except Exception:
                    pass
            all_features = lake_features + filtered_rivers
        
        # Clip to bounds
        clip_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
        clipped = []
        for f in all_features[:max_features]:
            try:
                clipped_geom = f.geometry.intersection(clip_box)
                if not clipped_geom.is_empty and clipped_geom.area > 50:
                    if clipped_geom.geom_type == 'MultiPolygon':
                        clipped_geom = max(clipped_geom.geoms, key=lambda g: g.area)
                    if clipped_geom.geom_type == 'Polygon':
                        f.geometry = clipped_geom
                        clipped.append(f)
            except Exception:
                pass
        
        print(f"    Total: {len(clipped)} water features")
        
        # Fetch elevations
        if clipped and fetch_elevations_func:
            coords = [(f.geometry.centroid.x, f.geometry.centroid.y) for f in clipped]
            if coords:
                elevations = fetch_elevations_func(coords)
                for feature, elev in zip(clipped, elevations):
                    if feature.attributes is None:
                        feature.attributes = {}
                    feature.attributes['elevation'] = elev
        
        return clipped
    
    def _fetch_lakes_from_tlm3d(self, bounds: tuple) -> List[WaterFeature]:
        """Fetch lakes and ponds from swissTLM3D."""
        minx, miny, maxx, maxy = bounds
        
        # Buffer for edge cases
        buffer = 50
        geometry = f'{minx-buffer},{miny-buffer},{maxx+buffer},{maxy+buffer}'
        
        params = {
            'geometryType': 'esriGeometryEnvelope',
            'geometry': geometry,
            'geometryFormat': 'geojson',
            'layers': f'all:{TLM3D_WATER_LAYER}',
            'tolerance': 0,
            'sr': 2056,
            'returnGeometry': True,
            'f': 'json'
        }
        
        try:
            response = self.session.get(TLM3D_API_URL, params=params, timeout=30)
            if response.status_code != 200:
                return []
            
            data = response.json()
            features = []
            
            for result in data.get('results', []):
                attrs = result.get('attributes', {})
                geom_data = result.get('geometry', {})
                
                # Get polygon features (lakes, ponds, river surfaces)
                geom_type = geom_data.get('type', '')
                if 'Polygon' not in geom_type:
                    continue
                
                try:
                    geom = shape(geom_data)
                    if geom.geom_type == 'MultiPolygon':
                        geom = max(geom.geoms, key=lambda g: g.area)
                    
                    if geom.geom_type != 'Polygon' or geom.area < 500:
                        continue
                    
                    name = (attrs.get('name') or '').strip() or None
                    
                    # Determine water type by area (lakes are typically > 10000m²)
                    water_type = 'lake' if geom.area > 10000 else 'pond'
                    
                    feature = WaterFeature(
                        id=f"tlm3d_{result.get('id', len(features))}",
                        geometry=geom,
                        water_type=water_type,
                        name=name,
                        is_underground=False,
                        attributes={'source': 'swissTLM3D'}
                    )
                    features.append(feature)
                except Exception as e:
                    logger.debug(f"Error parsing TLM3D feature: {e}")
            
            # Sort by area (largest first)
            features.sort(key=lambda f: f.geometry.area, reverse=True)
            return features
            
        except Exception as e:
            logger.debug(f"TLM3D request failed: {e}")
            return []
    
    def _extract_rivers_from_raster(
        self,
        minx: float,
        miny: float, 
        maxx: float,
        maxy: float,
        resolution: float = 2.0
    ) -> List[WaterFeature]:
        """Extract river polygons from FOEN WMS raster."""
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available")
            return []
        
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)
        
        # Limit size
        max_dim = 1000
        if width > max_dim or height > max_dim:
            scale = max_dim / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
            resolution = (maxx - minx) / width
        
        params = {
            'SERVICE': 'WMS',
            'VERSION': '1.3.0',
            'REQUEST': 'GetMap',
            'LAYERS': FOEN_WATER_LAYER,
            'CRS': 'EPSG:2056',
            'BBOX': f'{minx},{miny},{maxx},{maxy}',
            'WIDTH': width,
            'HEIGHT': height,
            'FORMAT': 'image/png'
        }
        
        try:
            response = self.session.get(FOEN_WMS_URL, params=params, timeout=30)
            if response.status_code != 200:
                return []
            
            if 'image' not in response.headers.get('content-type', ''):
                return []
            
            img = Image.open(BytesIO(response.content))
            arr = np.array(img)
            
            if len(arr.shape) < 3:
                return []
            
            # Detect water (cyan: low R, high G, high B)
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
            water_mask = ((b > 200) & (g > 150) & (r < 100)).astype(np.uint8) * 255
            
            if np.sum(water_mask) == 0:
                return []
            
            contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            features = []
            for i, contour in enumerate(contours):
                if len(contour) < 3:
                    continue
                
                coords = []
                for point in contour:
                    px, py = point[0]
                    x = minx + px * resolution
                    y = maxy - py * resolution
                    coords.append((x, y))
                
                if len(coords) < 3:
                    continue
                
                try:
                    poly = Polygon(coords)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    
                    if poly.is_valid and poly.area > 100:
                        feature = WaterFeature(
                            id=f"foen_river_{i}",
                            geometry=poly,
                            water_type="river",
                            name=None,
                            is_underground=False,
                            attributes={'source': 'foen_raster'}
                        )
                        features.append(feature)
                except Exception:
                    pass
            
            features.sort(key=lambda f: f.geometry.area, reverse=True)
            return features
            
        except Exception as e:
            logger.debug(f"Raster extraction failed: {e}")
            return []
