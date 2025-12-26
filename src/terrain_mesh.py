"""
Terrain mesh generation and triangulation

Creates terrain meshes with site and road cutouts.
"""

import numpy as np
from shapely.geometry import Point
from scipy.spatial import Delaunay
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_circular_terrain_grid(center_x: float, center_y: float, radius: float = 500.0, 
                                 resolution: float = 10.0) -> Tuple[List[Tuple[float, float]], Tuple[float, float, float, float]]:
    """
    Create a grid of points covering a circular terrain area.
    
    Returns:
        coords: List of (x, y) coordinates within the circle
        circle_bounds: (minx, miny, maxx, maxy) bounding box of the circle
    """
    # Create bounding box for the circle
    minx = center_x - radius
    maxx = center_x + radius
    miny = center_y - radius
    maxy = center_y + radius
    
    # Create grid covering the bounding box
    x_range = np.arange(minx, maxx + resolution, resolution)
    y_range = np.arange(miny, maxy + resolution, resolution)
    
    # Filter points to only include those within the circle
    coords = []
    center_point = Point(center_x, center_y)
    circle = center_point.buffer(radius)
    
    for y in y_range:
        for x in x_range:
            point = Point(x, y)
            if circle.contains(point) or circle.boundary.distance(point) < resolution * 0.1:
                coords.append((x, y))
    
    print(f"Created circular grid: {len(coords)} points within {radius}m radius")
    print(f"Circle center: E {center_x:.1f}, N {center_y:.1f}")
    print(f"Coverage: E {minx:.1f} - {maxx:.1f}, N {miny:.1f} - {maxy:.1f}")
    print(f"Resolution: {resolution}m")
    
    return coords, (minx, miny, maxx, maxy)


def apply_road_recesses_to_terrain(roads, fetch_elevations_func=None):
    """
    Create road polygons and sample edge points for clean terrain mesh integration.
    Edge points are added to terrain mesh so triangulation naturally follows road boundaries.

    Args:
        roads: List of RoadFeature objects
        fetch_elevations_func: Function to fetch elevations for edge points

    Returns:
        road_polygons: Combined road polygon for centroid-based removal
        road_edge_coords: List of (x, y) coordinates along road edges
        road_edge_elevations: List of elevations for road edge points
    """
    from shapely.ops import unary_union
    
    DEFAULT_ROAD_WIDTH = 5.0  # meters
    EDGE_SAMPLE_INTERVAL = 3.0  # Sample every 3m along road edges
    
    if not roads:
        return None, [], []
    
    # Create buffered polygons for all roads and collect edge points
    road_polygons = []
    edge_coords = []
    
    for road in roads:
        if road.geometry is None or road.geometry.is_empty:
            continue
        
        # Use road's width if available, otherwise default
        width = road.width if road.width else DEFAULT_ROAD_WIDTH
        half_width = width / 2.0
        
        try:
            road_poly = road.geometry.buffer(half_width, cap_style=2, join_style=2)
            if road_poly.is_valid and not road_poly.is_empty:
                road_polygons.append(road_poly)
                
                # Sample points along road edge
                if road_poly.geom_type == 'Polygon':
                    exterior = road_poly.exterior
                    for dist in np.arange(0, exterior.length, EDGE_SAMPLE_INTERVAL):
                        pt = exterior.interpolate(dist)
                        edge_coords.append((pt.x, pt.y))
                elif road_poly.geom_type == 'MultiPolygon':
                    for poly in road_poly.geoms:
                        exterior = poly.exterior
                        for dist in np.arange(0, exterior.length, EDGE_SAMPLE_INTERVAL):
                            pt = exterior.interpolate(dist)
                            edge_coords.append((pt.x, pt.y))
        except Exception as e:
            logger.warning(f"Could not buffer road {road.id}: {e}")
            continue
    
    if not road_polygons:
        return None, [], []
    
    # Merge all road polygons
    try:
        combined_roads = unary_union(road_polygons)
    except Exception as e:
        logger.warning(f"Could not merge road polygons: {e}")
        return None, [], []
    
    print(f"  Created road cutout from {len(road_polygons)} road segments")
    print(f"  Road area: {combined_roads.area:.1f} m²")
    print(f"  Sampled {len(edge_coords)} road edge points")
    
    # Fetch elevations for edge points
    edge_elevations = []
    if edge_coords and fetch_elevations_func:
        print(f"  Fetching elevations for road edge points...")
        edge_elevations = fetch_elevations_func(edge_coords)
    
    return combined_roads, edge_coords, edge_elevations


def triangulate_terrain_with_cutout(
    coords: List[Tuple[float, float]], 
    elevations: List[float], 
    site_polygon, 
    site_boundary_coords: Optional[List[Tuple[float, float]]] = None, 
    site_boundary_elevations: Optional[List[float]] = None, 
    road_polygons=None,
    road_edge_coords: Optional[List[Tuple[float, float]]] = None, 
    road_edge_elevations: Optional[List[float]] = None
) -> List[List[Tuple[float, float, float]]]:
    """
    Create triangulated mesh from points, excluding triangles inside site/roads.
    Road edge points are added to the mesh so triangulation naturally follows boundaries.
    
    Args:
        coords: List of (x, y) coordinates for terrain grid
        elevations: List of elevations corresponding to coords
        site_polygon: Shapely Polygon representing site boundary
        site_boundary_coords: Optional list of (x, y) for site boundary
        site_boundary_elevations: Optional elevations for site boundary
        road_polygons: Optional polygon(s) for road areas
        road_edge_coords: Optional list of (x, y) along road edges
        road_edge_elevations: Optional elevations for road edges
    
    Returns list of triangles, each as [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]
    """
    # Start with terrain grid
    all_coords = list(coords)
    all_elevations = list(elevations)
    
    # Merge site boundary points
    if site_boundary_coords and site_boundary_elevations:
        all_coords.extend(site_boundary_coords)
        all_elevations.extend(site_boundary_elevations)
        print(f"  Merged {len(site_boundary_coords)} site boundary points")
    
    # Merge road edge points - this makes triangulation follow road edges!
    if road_edge_coords and road_edge_elevations:
        all_coords.extend(road_edge_coords)
        all_elevations.extend(road_edge_elevations)
        print(f"  Merged {len(road_edge_coords)} road edge points")
    
    # Triangulate all points
    points_2d = np.array(all_coords)
    tri = Delaunay(points_2d)
    
    triangles_3d = []
    excluded_site_count = 0
    excluded_road_count = 0
    
    for simplex in tri.simplices:
        i0, i1, i2 = simplex
        v0_2d = all_coords[i0]
        v1_2d = all_coords[i1]
        v2_2d = all_coords[i2]
        
        p0 = (*v0_2d, all_elevations[i0])
        p1 = (*v1_2d, all_elevations[i1])
        p2 = (*v2_2d, all_elevations[i2])
        
        # Calculate triangle centroid
        centroid_x = (v0_2d[0] + v1_2d[0] + v2_2d[0]) / 3.0
        centroid_y = (v0_2d[1] + v1_2d[1] + v2_2d[1]) / 3.0
        centroid = Point(centroid_x, centroid_y)
        
        # Exclude triangles inside site polygon
        if site_polygon.contains(centroid):
            excluded_site_count += 1
            continue
        
        # Exclude triangles inside roads (centroid-based - clean since edges are in mesh)
        if road_polygons is not None and road_polygons.contains(centroid):
            excluded_road_count += 1
            continue
        
        triangles_3d.append([p0, p1, p2])
    
    print(f"  Excluded {excluded_site_count} triangles inside site boundary")
    if road_polygons is not None:
        print(f"  Excluded {excluded_road_count} triangles inside roads")
    print(f"Created {len(triangles_3d)} terrain triangles")
    return triangles_3d

