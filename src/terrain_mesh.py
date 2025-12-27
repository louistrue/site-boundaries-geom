"""
Terrain mesh generation and triangulation

Creates terrain meshes with site and road cutouts.
"""

import numpy as np
from shapely.geometry import Point, Polygon
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


def apply_water_cutouts_to_terrain(waters, terrain_coords=None, terrain_elevations=None, 
                                    fetch_elevations_func=None, terrain_boundary=None):
    """
    Create water polygons for surface water cutouts in terrain mesh.
    Only surface water (not underground) is cut out from terrain.
    Water polygons are clipped to terrain boundary to ensure clean edges.
    
    Args:
        waters: List of WaterFeature objects
        terrain_coords: Existing terrain coordinates for interpolation
        terrain_elevations: Existing terrain elevations for interpolation
        fetch_elevations_func: Fallback function to fetch elevations
        terrain_boundary: Shapely Polygon representing terrain extent (for clipping lakes)
    
    Returns:
        water_polygons: Combined polygon of surface water for terrain cutout (clipped to boundary)
        water_edge_coords: List of (x, y) coordinates along water edges
        water_edge_elevations: List of elevations for water edge points
    """
    from shapely.ops import unary_union
    from shapely.geometry import LineString
    from scipy.interpolate import LinearNDInterpolator
    
    EDGE_SAMPLE_INTERVAL = 2.0  # Sample every 2m along water edges for very smooth boundaries
    DEFAULT_STREAM_WIDTH = 3.0  # meters
    
    if not waters:
        return None, [], []
    
    # Filter to only surface water (not underground)
    surface_waters = [w for w in waters if not w.is_underground]
    
    if not surface_waters:
        print(f"  No surface water to cut out ({len(waters)} underground features)")
        return None, [], []
    
    print(f"  Processing {len(surface_waters)} surface water features for terrain cutout...")
    
    water_polygons = []
    edge_coords = []
    
    for water in surface_waters:
        if water.geometry is None or water.geometry.is_empty:
            continue
        
        try:
            if isinstance(water.geometry, Polygon):
                # Already a polygon (lake or water surface)
                water_poly = water.geometry
            elif isinstance(water.geometry, LineString):
                # Stream/river - buffer to create polygon
                width = water.width if water.width and water.width > 0 else DEFAULT_STREAM_WIDTH
                half_width = width / 2.0
                water_poly = water.geometry.buffer(half_width, cap_style=2, join_style=2)
            else:
                continue
            
            if water_poly.is_valid and not water_poly.is_empty:
                # Clip water polygon to terrain boundary if provided (for lakes extending beyond radius)
                if terrain_boundary is not None:
                    try:
                        clipped = water_poly.intersection(terrain_boundary)
                        if clipped.is_empty:
                            continue
                        # Use the clipped version
                        if clipped.geom_type == 'Polygon':
                            water_poly = clipped
                        elif clipped.geom_type == 'MultiPolygon':
                            # Take largest polygon if multiple
                            water_poly = max(clipped.geoms, key=lambda p: p.area)
                        else:
                            continue
                    except Exception as e:
                        logger.warning(f"Could not clip water {water.id} to boundary: {e}")
                        # Continue with unclipped polygon
                
                water_polygons.append(water_poly)
                
                # Sample points along water edge
                if water_poly.geom_type == 'Polygon':
                    exterior = water_poly.exterior
                    for dist in np.arange(0, exterior.length, EDGE_SAMPLE_INTERVAL):
                        pt = exterior.interpolate(dist)
                        edge_coords.append((pt.x, pt.y))
                elif water_poly.geom_type == 'MultiPolygon':
                    for poly in water_poly.geoms:
                        exterior = poly.exterior
                        for dist in np.arange(0, exterior.length, EDGE_SAMPLE_INTERVAL):
                            pt = exterior.interpolate(dist)
                            edge_coords.append((pt.x, pt.y))
        except Exception as e:
            logger.warning(f"Could not process water {water.id}: {e}")
            continue
    
    if not water_polygons:
        return None, [], []
    
    # Merge all water polygons
    try:
        combined_water = unary_union(water_polygons)
    except Exception as e:
        logger.warning(f"Could not merge water polygons: {e}")
        return None, [], []
    
    print(f"  Created water cutout from {len(water_polygons)} surface water features")
    print(f"  Water cutout area: {combined_water.area:.1f} m²")
    print(f"  Sampled {len(edge_coords)} water edge points")
    
    # Get elevations for edge points - prefer interpolation over API calls
    edge_elevations = []
    if edge_coords:
        if terrain_coords is not None and terrain_elevations is not None and len(terrain_coords) > 0:
            # FAST: Interpolate from existing terrain data
            print("  Interpolating water edge elevations from terrain data...")
            try:
                terrain_points = np.array(terrain_coords)
                terrain_z = np.array(terrain_elevations)
                interpolator = LinearNDInterpolator(terrain_points, terrain_z)
                
                edge_points = np.array(edge_coords)
                edge_elevations = interpolator(edge_points)
                
                # Handle NaN values with nearest neighbor
                nan_mask = np.isnan(edge_elevations)
                if np.any(nan_mask):
                    from scipy.interpolate import NearestNDInterpolator
                    nearest = NearestNDInterpolator(terrain_points, terrain_z)
                    edge_elevations[nan_mask] = nearest(edge_points[nan_mask])
                
                edge_elevations = edge_elevations.tolist()
                print(f"  Interpolated {len(edge_elevations)} water edge elevations")
            except Exception as e:
                logger.warning(f"Water interpolation failed, falling back to API: {e}")
                if fetch_elevations_func:
                    edge_elevations = fetch_elevations_func(edge_coords)
        elif fetch_elevations_func:
            # SLOW fallback: Fetch via API
            print(f"  Fetching water edge elevations via API ({len(edge_coords)} calls)...")
            edge_elevations = fetch_elevations_func(edge_coords)
    
    return combined_water, edge_coords, edge_elevations


def apply_road_recesses_to_terrain(roads, terrain_coords=None, terrain_elevations=None, fetch_elevations_func=None):
    """
    Create road polygons and sample edge points for clean terrain mesh integration.
    Edge points are added to terrain mesh so triangulation naturally follows road boundaries.

    Args:
        roads: List of RoadFeature objects
        terrain_coords: Existing terrain coordinates for interpolation (FAST - no API calls!)
        terrain_elevations: Existing terrain elevations for interpolation
        fetch_elevations_func: Fallback function to fetch elevations (SLOW - API calls)

    Returns:
        road_polygons: Combined road polygon for centroid-based removal
        road_edge_coords: List of (x, y) coordinates along road edges
        road_edge_elevations: List of elevations for road edge points
    """
    from shapely.ops import unary_union
    from scipy.interpolate import LinearNDInterpolator
    
    DEFAULT_ROAD_WIDTH = 5.0  # meters
    EDGE_SAMPLE_INTERVAL = 2.0  # Sample every 2m along road edges for very smooth boundaries
    
    if not roads:
        return None, [], []
    
    # Create buffered polygons for all roads
    road_polygons = []
    
    for road in roads:
        if road.geometry is None or road.geometry.is_empty:
            continue
        
        # Use road's width if available, otherwise default
        width = road.width if road.width else DEFAULT_ROAD_WIDTH
        half_width = width / 2.0
        
        try:
            # Use round caps/joins for smooth edges (cap_style=1, join_style=1)
            road_poly = road.geometry.buffer(half_width, cap_style=1, join_style=1, resolution=8)
            if road_poly.is_valid and not road_poly.is_empty:
                road_polygons.append(road_poly)
        except Exception as e:
            logger.warning(f"Could not buffer road {road.id}: {e}")
            continue
    
    if not road_polygons:
        return None, [], []
    
    # Merge all road polygons and simplify for clean edges
    try:
        combined_roads = unary_union(road_polygons)
        # Simplify the merged polygon to remove jagged artifacts (0.5m tolerance)
        combined_roads = combined_roads.simplify(0.5, preserve_topology=True)
    except Exception as e:
        logger.warning(f"Could not merge road polygons: {e}")
        return None, [], []
    
    print(f"  Created road cutout from {len(road_polygons)} road segments")
    print(f"  Road area: {combined_roads.area:.1f} m²")
    
    # Sample edge points from the MERGED and SIMPLIFIED polygon for clean boundaries
    edge_coords = []
    if combined_roads.geom_type == 'Polygon':
        exterior = combined_roads.exterior
        for dist in np.arange(0, exterior.length, EDGE_SAMPLE_INTERVAL):
            pt = exterior.interpolate(dist)
            edge_coords.append((pt.x, pt.y))
    elif combined_roads.geom_type == 'MultiPolygon':
        for poly in combined_roads.geoms:
            exterior = poly.exterior
            for dist in np.arange(0, exterior.length, EDGE_SAMPLE_INTERVAL):
                pt = exterior.interpolate(dist)
                edge_coords.append((pt.x, pt.y))
    
    print(f"  Sampled {len(edge_coords)} road edge points")
    
    # Get elevations for edge points - prefer interpolation over API calls
    edge_elevations = []
    if edge_coords:
        if terrain_coords is not None and terrain_elevations is not None and len(terrain_coords) > 0:
            # FAST: Interpolate from existing terrain data (no API calls!)
            print("  Interpolating elevations from terrain data...")
            try:
                terrain_points = np.array(terrain_coords)
                terrain_z = np.array(terrain_elevations)
                interpolator = LinearNDInterpolator(terrain_points, terrain_z)
                
                edge_points = np.array(edge_coords)
                edge_elevations = interpolator(edge_points)
                
                # Handle NaN values (points outside interpolation range) with nearest neighbor
                nan_mask = np.isnan(edge_elevations)
                if np.any(nan_mask):
                    from scipy.interpolate import NearestNDInterpolator
                    nearest = NearestNDInterpolator(terrain_points, terrain_z)
                    edge_elevations[nan_mask] = nearest(edge_points[nan_mask])
                
                edge_elevations = edge_elevations.tolist()
                print(f"  Interpolated {len(edge_elevations)} elevations (0 API calls)")
            except Exception as e:
                logger.warning(f"Interpolation failed, falling back to API: {e}")
                if fetch_elevations_func:
                    edge_elevations = fetch_elevations_func(edge_coords)
        elif fetch_elevations_func:
            # SLOW fallback: Fetch via API
            print(f"  Fetching elevations via API ({len(edge_coords)} calls)...")
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
    road_edge_elevations: Optional[List[float]] = None,
    water_polygons=None,
    water_edge_coords: Optional[List[Tuple[float, float]]] = None,
    water_edge_elevations: Optional[List[float]] = None
) -> List[List[Tuple[float, float, float]]]:
    """
    Create triangulated terrain mesh with clean cutouts for site, roads, and water.
    Uses mapbox-earcut for proper polygon triangulation that respects holes.
    
    Args:
        coords: List of (x, y) coordinates for terrain grid
        elevations: List of elevations corresponding to coords
        site_polygon: Shapely Polygon representing site boundary
        site_boundary_coords: Optional list of (x, y) for site boundary
        site_boundary_elevations: Optional elevations for site boundary
        road_polygons: Optional polygon(s) for road areas (will be cut as holes)
        road_edge_coords: Optional list of (x, y) along road edges
        road_edge_elevations: Optional elevations for road edges
        water_polygons: Optional polygon(s) for surface water areas (will be cut as holes)
        water_edge_coords: Optional list of (x, y) along water edges
        water_edge_elevations: Optional elevations for water edges
    
    Returns list of triangles, each as [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]
    """
    import mapbox_earcut as earcut
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    
    # Calculate terrain extent
    coords_arr = np.array(coords)
    center_x = (coords_arr[:, 0].min() + coords_arr[:, 0].max()) / 2
    center_y = (coords_arr[:, 1].min() + coords_arr[:, 1].max()) / 2
    radius = np.sqrt((coords_arr[:, 0] - center_x)**2 + (coords_arr[:, 1] - center_y)**2).max()
    
    print(f"  Terrain extent: center ({center_x:.1f}, {center_y:.1f}), radius {radius:.1f}m")
    
    # Create terrain boundary polygon (circular)
    terrain_boundary = Point(center_x, center_y).buffer(radius, resolution=64)
    
    # Compute terrain polygon by subtracting holes (site, roads, water)
    terrain_poly = terrain_boundary
    
    # Subtract site polygon
    if site_polygon is not None and not site_polygon.is_empty:
        try:
            clipped_site = site_polygon.intersection(terrain_boundary)
            if not clipped_site.is_empty:
                terrain_poly = terrain_poly.difference(clipped_site.buffer(0.1))
                print("  Subtracted site boundary")
        except Exception as e:
            logger.warning(f"Could not subtract site: {e}")
    
    # Subtract road polygons
    if road_polygons is not None and not road_polygons.is_empty:
        try:
            clipped_roads = road_polygons.intersection(terrain_boundary)
            if not clipped_roads.is_empty:
                # Simplify roads slightly to avoid tiny segments
                simplified_roads = clipped_roads.simplify(0.5, preserve_topology=True)
                terrain_poly = terrain_poly.difference(simplified_roads)
                print("  Subtracted road areas")
        except Exception as e:
            logger.warning(f"Could not subtract roads: {e}")
    
    # Subtract water polygons
    if water_polygons is not None and not water_polygons.is_empty:
        try:
            clipped_water = water_polygons.intersection(terrain_boundary)
            if not clipped_water.is_empty:
                simplified_water = clipped_water.simplify(0.5, preserve_topology=True)
                terrain_poly = terrain_poly.difference(simplified_water)
                print("  Subtracted water areas")
        except Exception as e:
            logger.warning(f"Could not subtract water: {e}")
    
    # Handle result - could be Polygon, MultiPolygon, or GeometryCollection
    all_polygons = []
    if terrain_poly.geom_type == 'Polygon':
        if not terrain_poly.is_empty and terrain_poly.area > 1:
            all_polygons.append(terrain_poly)
    elif terrain_poly.geom_type == 'MultiPolygon':
        for poly in terrain_poly.geoms:
            if not poly.is_empty and poly.area > 1:
                all_polygons.append(poly)
    elif terrain_poly.geom_type == 'GeometryCollection':
        for geom in terrain_poly.geoms:
            if geom.geom_type == 'Polygon' and not geom.is_empty and geom.area > 1:
                all_polygons.append(geom)
    
    print(f"  Terrain split into {len(all_polygons)} polygon(s)")
    
    # Build elevation interpolator
    terrain_pts = np.array(coords)
    terrain_z = np.array(elevations)
    linear_interp = LinearNDInterpolator(terrain_pts, terrain_z)
    nearest_interp = NearestNDInterpolator(terrain_pts, terrain_z)
    
    def get_elevation(x, y):
        z = linear_interp([[x, y]])[0]
        if np.isnan(z):
            z = nearest_interp([[x, y]])[0]
        return float(z)
    
    # Triangulate each polygon using earcut
    triangles_3d = []
    
    for poly in all_polygons:
        try:
            # Get exterior ring (densified for smoother terrain)
            exterior = poly.exterior
            # Densify the exterior to add more vertices for terrain detail
            densified_exterior = _densify_ring(exterior, max_segment_length=15.0)
            # Force 2D coordinates for earcut (some sources have 3D coords)
            exterior_coords = np.array([(c[0], c[1]) for c in densified_exterior.coords])[:-1]
            
            # Collect all rings: exterior + holes
            all_ring_coords = [exterior_coords]
            ring_ends = [len(exterior_coords)]
            
            # Add interior rings (holes within this polygon)
            for interior in poly.interiors:
                densified_interior = _densify_ring(interior, max_segment_length=5.0)
                # Force 2D coordinates
                hole_coords = np.array([(c[0], c[1]) for c in densified_interior.coords])[:-1]
                if len(hole_coords) >= 3:
                    all_ring_coords.append(hole_coords)
                    ring_ends.append(ring_ends[-1] + len(hole_coords))
            
            # Stack all vertices
            vertices = np.vstack(all_ring_coords)
            rings_arr = np.array(ring_ends, dtype=np.uint32)
            
            # Run earcut triangulation
            tri_indices = earcut.triangulate_float64(vertices, rings_arr)
            num_tris = len(tri_indices) // 3
            
            # Convert to 3D triangles with elevations
            for i in range(0, len(tri_indices), 3):
                i0, i1, i2 = tri_indices[i], tri_indices[i+1], tri_indices[i+2]
                v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
                
                z0 = get_elevation(v0[0], v0[1])
                z1 = get_elevation(v1[0], v1[1])
                z2 = get_elevation(v2[0], v2[1])
                
                triangles_3d.append([
                    (v0[0], v0[1], z0),
                    (v1[0], v1[1], z1),
                    (v2[0], v2[1], z2)
                ])
            
        except Exception as e:
            logger.warning(f"Earcut failed for polygon: {e}")
            continue
    
    print(f"Created {len(triangles_3d)} terrain triangles (earcut)")
    return triangles_3d


def _densify_ring(ring, max_segment_length: float = 10.0):
    """Add vertices to a ring so no segment is longer than max_segment_length."""
    from shapely.geometry import LineString
    
    coords = list(ring.coords)
    new_coords = []
    
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        # Force 2D coordinates
        new_coords.append((p1[0], p1[1]))
        
        # Calculate distance
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = (dx*dx + dy*dy) ** 0.5
        
        if dist > max_segment_length:
            # Add intermediate points
            num_segments = int(np.ceil(dist / max_segment_length))
            for j in range(1, num_segments):
                t = j / num_segments
                new_coords.append((
                    p1[0] + t * dx,
                    p1[1] + t * dy
                ))
    
    # Add closing point (force 2D)
    new_coords.append((coords[-1][0], coords[-1][1]))
    
    return LineString(new_coords)


