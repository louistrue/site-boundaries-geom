#!/usr/bin/env python3
"""
Combined Terrain with Site Cutout

This script creates a single IFC file containing:
- Surrounding terrain mesh with a hole cut out for the site
- Site solid with smoothed surface, height-adjusted to align with terrain edges

Usage:
    python -m src.terrain_with_site --egrid CH999979659148 --radius 500 --resolution 20 --output combined.ifc
"""

import numpy as np
import requests
import ifcopenshell
import ifcopenshell.api
from shapely.geometry import shape, Point, Polygon
from shapely.ops import triangulate
from shapely.geometry.polygon import orient
import argparse
import time
import sys
import math
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def fetch_boundary_by_egrid(egrid):
    """
    Fetch the cadastral boundary (Polygon) and metadata for a given EGRID via geo.admin.ch API.
    Returns tuple: (Shapely geometry in EPSG:2056, metadata dict)
    """
    url = "https://api3.geo.admin.ch/rest/services/ech/MapServer/find"
    params = {
        "layer": "ch.kantone.cadastralwebmap-farbe",
        "searchText": egrid,
        "searchField": "egris_egrid",
        "returnGeometry": "true",
        "geometryFormat": "geojson",
        "sr": "2056"
    }
    
    print(f"Fetching boundary for EGRID {egrid}...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    if not data.get("results"):
        print(f"No results found for EGRID {egrid}")
        return None, None
    
    feature = data["results"][0]
    geometry = shape(feature["geometry"])
    
    # Extract cadastre metadata (API uses 'properties' when geometry is included)
    attributes = feature.get("properties", {}) or feature.get("attributes", {})
    
    # Calculate area from geometry (in m² since EPSG:2056 is in meters)
    area_m2 = geometry.area
    
    metadata = {
        "egrid": egrid,
        "canton": attributes.get("ak", ""),
        "parcel_number": attributes.get("number", ""),
        "local_id": attributes.get("identnd", ""),
        "geoportal_url": attributes.get("geoportal_url", ""),
        "realestate_type": attributes.get("realestate_type", ""),
        "area_m2": round(area_m2, 2),
        "perimeter_m": round(geometry.length, 2),
    }
    
    # Print metadata
    if metadata["canton"]:
        print(f"  Canton: {metadata['canton']}")
    if metadata["parcel_number"]:
        print(f"  Parcel Number: {metadata['parcel_number']}")
    print(f"  Area: {metadata['area_m2']:.1f} m² ({metadata['area_m2']/10000:.3f} ha)")
    print(f"  Perimeter: {metadata['perimeter_m']:.1f} m")
    
    return geometry, metadata


def _fetch_single_elevation(coord):
    """
    Fetch elevation for a single coordinate.
    
    Args:
        coord: Tuple of (x, y) coordinates
        
    Returns:
        Tuple of (index, elevation) or (index, None) on failure
    """
    x, y = coord
    url = "https://api3.geo.admin.ch/rest/services/height"
    try:
        res = requests.get(url, params={"easting": x, "northing": y, "sr": "2056"}, timeout=10)
        res.raise_for_status()
        h = float(res.json()["height"])
        return h
    except Exception:
        return None


def fetch_elevation_batch(coords, batch_size=50, delay=0.1, max_workers=15):
    """
    Fetch elevations for a list of coordinates via geo.admin.ch REST height service.
    Uses concurrent requests for faster processing.
    
    Args:
        coords: List of (x, y) coordinate tuples
        batch_size: Progress reporting interval (deprecated, kept for compatibility)
        delay: Delay between batches (deprecated, kept for compatibility)
        max_workers: Number of concurrent workers (default: 15)
        
    Returns:
        List of elevations in same order as coords
    """
    total = len(coords)
    elevations = [None] * total
    failed_count = 0
    
    print(f"Fetching elevations for {total} points (concurrent, {max_workers} workers)...")
    
    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(_fetch_single_elevation, coord): i 
            for i, coord in enumerate(coords)
        }
        
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                elevation = future.result()
                if elevation is not None:
                    elevations[index] = elevation
                else:
                    failed_count += 1
                    # Set to None for post-processing - don't access neighbors here due to race condition
                    elevations[index] = None
            except Exception:
                failed_count += 1
                # Set to None for post-processing - don't access neighbors here due to race condition
                elevations[index] = None
            
            completed += 1
            if completed % batch_size == 0:
                pct = completed / total * 100
                print(f"  Progress: {completed}/{total} ({pct:.1f}%)")
    
    # Post-processing: replace None values with nearest previous non-None value (or 0.0)
    # This is race-free since all async operations have completed
    for i in range(total):
        if elevations[i] is None:
            # Find nearest previous non-None value
            fallback_value = 0.0
            for j in range(i - 1, -1, -1):
                if elevations[j] is not None:
                    fallback_value = elevations[j]
                    break
            elevations[i] = fallback_value
    
    if failed_count > 0:
        print(f"  Warning: {failed_count} points failed to fetch elevation")
    
    return elevations


def create_circular_terrain_grid(center_x, center_y, radius=500.0, resolution=10.0):
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


def _circular_mean(values, window_size):
    """Smooth values with a circular mean filter."""
    n = len(values)
    if n == 0:
        return []

    window_size = min(window_size, n)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 1:
        window_size = 1

    half_window = window_size // 2
    smoothed = []
    for i in range(n):
        window = [values[(i + j) % n] for j in range(-half_window, half_window + 1)]
        smoothed.append(float(np.mean(window)))
    return smoothed


def _best_fit_plane(ext_coords):
    """Project coordinates onto a best-fit plane to flatten bumps while keeping tilt."""
    if len(ext_coords) < 3:
        return [c[2] for c in ext_coords]

    arr = np.array(ext_coords, dtype=float)
    A = np.column_stack((arr[:, 0], arr[:, 1], np.ones(len(arr))))
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, arr[:, 2], rcond=None)
    except np.linalg.LinAlgError:
        return [c[2] for c in ext_coords]

    plane_z = A @ coeffs
    return plane_z.tolist()


def triangulate_terrain_with_cutout(coords, elevations, site_polygon, site_boundary_coords=None, site_boundary_elevations=None):
    """
    Create triangulated mesh from points, excluding triangles inside site boundary.
    Includes site boundary vertices in triangulation to ensure precise cutout shape.
    
    Args:
        coords: List of (x, y) coordinates for terrain grid
        elevations: List of elevations corresponding to coords
        site_polygon: Shapely Polygon representing site boundary
        site_boundary_coords: Optional list of (x, y) coordinates for site boundary
        site_boundary_elevations: Optional list of elevations for site boundary
    
    Returns list of triangles, each as [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]
    """
    # Merge site boundary points with terrain grid if provided
    if site_boundary_coords and site_boundary_elevations:
        # Combine terrain and boundary coordinates
        all_coords = coords + site_boundary_coords
        all_elevations = elevations + site_boundary_elevations
        print(f"  Merged {len(site_boundary_coords)} site boundary points into terrain grid")
    else:
        all_coords = coords
        all_elevations = elevations
    
    try:
        from scipy.spatial import Delaunay
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    if use_scipy:
        # Use scipy's Delaunay triangulation on combined point set
        points_2d = np.array(all_coords)
        tri = Delaunay(points_2d)
        
        triangles_3d = []
        excluded_count = 0
        
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
            
            # Exclude triangles whose centroid is inside site polygon
            # This ensures clean cutout even with boundary points in triangulation
            if not site_polygon.contains(centroid):
                triangles_3d.append([p0, p1, p2])
            else:
                excluded_count += 1
        
        print(f"  Excluded {excluded_count} triangles with centroid inside site boundary")
        return triangles_3d
    else:
        # Fallback: Use Shapely's triangulate
        from shapely.geometry import MultiPoint
        
        multipoint = MultiPoint([Point(x, y) for x, y in all_coords])
        triangles_2d = triangulate(multipoint)
        
        triangles_3d = []
        excluded_count = 0
        
        for tri_2d in triangles_2d:
            tri_coords_2d = list(tri_2d.exterior.coords)[:-1]
            
            if len(tri_coords_2d) == 3:
                # Check if triangle centroid is inside site
                centroid = tri_2d.centroid
                if site_polygon.contains(centroid):
                    excluded_count += 1
                    continue
                
                # Map to 3D coordinates
                tri_3d = []
                for tx, ty in tri_coords_2d:
                    min_dist = float('inf')
                    closest_idx = 0
                    for idx, (cx, cy) in enumerate(all_coords):
                        dist = math.sqrt((tx - cx)**2 + (ty - cy)**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = idx
                    
                    tri_3d.append((all_coords[closest_idx][0], all_coords[closest_idx][1], all_elevations[closest_idx]))
                
                if len(tri_3d) == 3:
                    triangles_3d.append(tri_3d)
        
        print(f"  Excluded {excluded_count} triangles with centroid inside site boundary")
        return triangles_3d


def create_site_solid_coords(site_polygon, site_coords_3d, z_offset_adjustment=0.0):
    """
    Create smoothed site solid coordinates with height adjustment.
    
    Args:
        site_polygon: Shapely Polygon of site boundary
        site_coords_3d: List of (x, y, z) coordinates for site boundary
        z_offset_adjustment: Additional Z offset to align with terrain
    
    Returns:
        ext_coords: List of (x, y, z) coordinates for smoothed boundary
        base_elevation: Base elevation for solid bottom
        polygon_2d: 2D polygon for triangulation
        smoothed_boundary_2d: List of (x, y) for smoothed boundary
        smoothed_boundary_z: List of Z values for smoothed boundary
    """
    # Apply smoothing (same as site_solid.py)
    ext_coords = [(float(x), float(y), float(z)) for x, y, z in site_coords_3d]
    if ext_coords[0] == ext_coords[-1]:
        ext_coords = ext_coords[:-1]
    
    # Apply smoothing
    z_values = [c[2] for c in ext_coords]
    plane_z = _best_fit_plane(ext_coords)
    smoothed_z = _circular_mean(z_values, window_size=9)
    residuals = [sz - pz for sz, pz in zip(smoothed_z, plane_z)]
    smoothed_residuals = _circular_mean(residuals, window_size=9)
    
    # Heavily attenuate residuals (20% scale)
    residual_scale = 0.2
    flattened_z = [pz + residual_scale * rz for pz, rz in zip(plane_z, smoothed_residuals)]
    
    # Apply height adjustment to align with terrain
    adjusted_z = [z + z_offset_adjustment for z in flattened_z]
    
    ext_coords = [(ext_coords[i][0], ext_coords[i][1], adjusted_z[i]) for i in range(len(ext_coords))]
    
    base_elevation = min(z for _, _, z in ext_coords) - 2.0  # 2 meters below lowest point
    
    # Create 2D polygon for triangulation
    polygon_2d = Polygon([(x, y) for x, y, _ in ext_coords])
    if not polygon_2d.is_valid:
        polygon_2d = polygon_2d.buffer(0)
    
    # Extract smoothed boundary for terrain attachment
    smoothed_boundary_2d = [(x, y) for x, y, _ in ext_coords]
    smoothed_boundary_z = [z for _, _, z in ext_coords]
    
    return ext_coords, base_elevation, polygon_2d, smoothed_boundary_2d, smoothed_boundary_z


def calculate_height_offset(site_polygon, site_coords_3d, terrain_coords, terrain_elevations):
    """
    Calculate Z offset needed to align site solid edges with terrain.
    
    Returns the average offset to apply to smoothed site elevations.
    """
    # Sample terrain elevations at site boundary points
    boundary_terrain_z = []
    
    for x, y, _ in site_coords_3d:
        # Find closest terrain point
        min_dist = float('inf')
        closest_z = None
        for idx, (tx, ty) in enumerate(terrain_coords):
            dist = math.sqrt((tx - x)**2 + (ty - y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_z = terrain_elevations[idx]
        
        if closest_z is not None:
            boundary_terrain_z.append(closest_z)
    
    if not boundary_terrain_z:
        return 0.0
    
    # Get smoothed site elevations at boundary
    ext_coords = [(float(x), float(y), float(z)) for x, y, z in site_coords_3d]
    if ext_coords[0] == ext_coords[-1]:
        ext_coords = ext_coords[:-1]
    
    z_values = [c[2] for c in ext_coords]
    plane_z = _best_fit_plane(ext_coords)
    smoothed_z = _circular_mean(z_values, window_size=9)
    residuals = [sz - pz for sz, pz in zip(smoothed_z, plane_z)]
    smoothed_residuals = _circular_mean(residuals, window_size=9)
    residual_scale = 0.2
    smoothed_boundary_z = [pz + residual_scale * rz for pz, rz in zip(plane_z, smoothed_residuals)]
    
    # Calculate average offset
    avg_terrain_z = np.mean(boundary_terrain_z)
    avg_smoothed_z = np.mean(smoothed_boundary_z)
    z_offset = avg_terrain_z - avg_smoothed_z
    
    print(f"  Terrain avg Z at boundary: {avg_terrain_z:.2f}m")
    print(f"  Smoothed site avg Z: {avg_smoothed_z:.2f}m")
    print(f"  Height offset adjustment: {z_offset:.2f}m")
    
    return z_offset


def create_combined_ifc(terrain_triangles, site_solid_data, output_path, bounds, 
                        center_x, center_y, egrid=None, cadastre_metadata=None, return_model=False):
    """
    Create an IFC file with terrain (with hole) and/or site solid.
    terrain_triangles: List of triangles for terrain mesh, or None to skip terrain
    site_solid_data: Dict with site solid data, or None to skip site solid
    cadastre_metadata: dict with parcel info from cadastre API
    return_model: If True, return model object instead of writing to file
    
    Returns:
        If return_model=True: (model, offset_x, offset_y, offset_z)
        If return_model=False: (offset_x, offset_y, offset_z)
    """
    model = ifcopenshell.file(schema='IFC4')
    
    # Create OwnerHistory (required by many IFC viewers for psets)
    person = model.createIfcPerson(FamilyName="User")
    organization = model.createIfcOrganization(Name="Site Boundaries Tool")
    person_org = model.createIfcPersonAndOrganization(person, organization)
    application = model.createIfcApplication(
        organization, 
        "1.0", 
        "Site Boundaries Geometry Tool", 
        "SiteBoundariesGeom"
    )
    owner_history = model.createIfcOwnerHistory(
        OwningUser=person_org, 
        OwningApplication=application, 
        ChangeAction="ADDED",
        CreationDate=int(time.time())
    )
    
    def set_owner_history_on_pset(pset):
        """Set OwnerHistory on pset and its relationship."""
        pset.OwnerHistory = owner_history
        # Find and update the relationship
        for rel in model.by_type('IfcRelDefinesByProperties'):
            if rel.RelatingPropertyDefinition == pset:
                rel.OwnerHistory = owner_history
    
    # Project setup
    project = ifcopenshell.api.run("root.create_entity", model,
                                    ifc_class="IfcProject",
                                    name="Combined Terrain Model")
    project.OwnerHistory = owner_history
    
    length_unit = ifcopenshell.api.run("unit.add_si_unit", model)
    ifcopenshell.api.run("unit.assign_unit", model, units=[length_unit])
    
    context = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    body_context = ifcopenshell.api.run(
        "context.add_context",
        model,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=context,
    )
    footprint_context = ifcopenshell.api.run(
        "context.add_context",
        model,
        context_type="Plan",
        context_identifier="FootPrint",
        target_view="PLAN_VIEW",
        parent=context,
    )
    
    # CRS setup
    crs = model.createIfcProjectedCRS(
        Name="EPSG:2056",
        Description="Swiss LV95 / CH1903+",
        GeodeticDatum="CH1903+",
        VerticalDatum="LN02"
    )
    
    # Calculate offsets - center on site
    minx, miny, maxx, maxy = bounds
    offset_x = round(center_x, -2)  # Round to nearest 100m
    offset_y = round(center_y, -2)
    
    # Determine min_z from available data
    if terrain_triangles:
        min_z = min(p[2] for tri in terrain_triangles for p in tri)
    elif site_solid_data:
        min_z = min(z for _, _, z in site_solid_data['ext_coords'])
    else:
        min_z = 0.0
    offset_z = round(min_z)
    
    print(f"\nProject Origin: E={offset_x}, N={offset_y}, H={offset_z}")
    print(f"Site Center (relative to origin): E={center_x - offset_x:.1f}, N={center_y - offset_y:.1f}")
    
    model.createIfcMapConversion(
        SourceCRS=context,
        TargetCRS=crs,
        Eastings=float(offset_x),
        Northings=float(offset_y),
        OrthogonalHeight=float(offset_z)
    )
    
    # Create Site
    site_name = f"Site_{egrid}" if egrid else "Combined_Site"
    site = ifcopenshell.api.run("root.create_entity", model,
                                 ifc_class="IfcSite",
                                 name=site_name)
    site.OwnerHistory = owner_history
    ifcopenshell.api.run("aggregate.assign_object", model,
                          products=[site], relating_object=project)
    ifcopenshell.api.run("geometry.edit_object_placement", model, product=site)
    
    # Add site footprint representation for visibility
    if site_solid_data and site_solid_data.get('polygon_2d'):
        try:
            polygon_2d = site_solid_data['polygon_2d']
            coords_2d = list(polygon_2d.exterior.coords)
            if len(coords_2d) >= 3:
                # Remove duplicate closing point
                if coords_2d[0] == coords_2d[-1]:
                    coords_2d = coords_2d[:-1]
                
                footprint_points = [
                    model.createIfcCartesianPoint([
                        float(x - offset_x),
                        float(y - offset_y)
                    ])
                    for x, y in coords_2d
                ]
                # Close the polyline
                footprint_points.append(footprint_points[0])
                
                polyline = model.createIfcPolyLine(footprint_points)
                site_footprint_rep = model.createIfcShapeRepresentation(
                    footprint_context, "FootPrint", "Curve2D", [polyline])
                site.Representation = model.createIfcProductDefinitionShape(
                    None, None, [site_footprint_rep])
        except Exception as e:
            logger.debug(f"Could not add site footprint: {e}")
    
    # Create terrain mesh (with hole) if requested
    if terrain_triangles:
        print(f"\nCreating terrain mesh with {len(terrain_triangles)} triangles...")
        terrain = ifcopenshell.api.run("root.create_entity", model,
                                        ifc_class="IfcGeographicElement",
                                        name="Surrounding_Terrain")
        terrain.OwnerHistory = owner_history
        terrain.PredefinedType = "TERRAIN"
        
        terrain_origin = model.createIfcCartesianPoint([0., 0., 0.])
        terrain_axis = model.createIfcDirection([0., 0., 1.])
        terrain_ref_direction = model.createIfcDirection([1., 0., 0.])
        terrain_axis2_placement = model.createIfcAxis2Placement3D(terrain_origin, terrain_axis, terrain_ref_direction)
        terrain_placement = model.createIfcLocalPlacement(site.ObjectPlacement, terrain_axis2_placement)
        terrain.ObjectPlacement = terrain_placement
        
        ifcopenshell.api.run("spatial.assign_container", model,
                              products=[terrain], relating_structure=site)
        
        terrain_faces = []
        for tri in terrain_triangles:
            local_pts = [(p[0] - offset_x, p[1] - offset_y, p[2] - offset_z) for p in tri]
            tri_points = [
                model.createIfcCartesianPoint([float(p[0]), float(p[1]), float(p[2])])
                for p in local_pts
            ]
            tri_loop = model.createIfcPolyLoop(tri_points)
            face = model.createIfcFace([model.createIfcFaceOuterBound(tri_loop, True)])
            terrain_faces.append(face)
        
        terrain_shell = model.createIfcOpenShell(terrain_faces)
        terrain_shell_model = model.createIfcShellBasedSurfaceModel([terrain_shell])
        
        terrain_rep = model.createIfcShapeRepresentation(
            body_context, "Body", "SurfaceModel", [terrain_shell_model])
        terrain.Representation = model.createIfcProductDefinitionShape(
            None, None, [terrain_rep])
    
    # Create site solid
    if site_solid_data:
        try:
            print(f"Creating site solid...")
            ext_coords = site_solid_data['ext_coords']
            base_elevation = site_solid_data['base_elevation']
            polygon_2d = site_solid_data['polygon_2d']
            
            print(f"  Site boundary points: {len(ext_coords)}")
            
            site_terrain = ifcopenshell.api.run("root.create_entity", model,
                                                 ifc_class="IfcGeographicElement",
                                                 name=f"Site_Solid_{egrid}" if egrid else "Site_Solid")
            site_terrain.OwnerHistory = owner_history
            site_terrain.PredefinedType = "TERRAIN"
            
            site_origin = model.createIfcCartesianPoint([0., 0., 0.])
            site_axis = model.createIfcDirection([0., 0., 1.])
            site_ref_direction = model.createIfcDirection([1., 0., 0.])
            site_axis2_placement = model.createIfcAxis2Placement3D(site_origin, site_axis, site_ref_direction)
            site_terrain_placement = model.createIfcLocalPlacement(site.ObjectPlacement, site_axis2_placement)
            site_terrain.ObjectPlacement = site_terrain_placement
            
            ifcopenshell.api.run("spatial.assign_container", model,
                                  products=[site_terrain], relating_structure=site)
            
            # Create local coordinates
            print(f"  Converting to local coordinates...")
            local_ext_coords = [(float(x - offset_x), float(y - offset_y), float(z - offset_z)) 
                                for x, y, z in ext_coords]
            local_base_elevation = base_elevation - offset_z
            
            z_lookup = {(round(x, 6), round(y, 6)): z for x, y, z in local_ext_coords}
            
            def get_vertex_z(x, y):
                key = (round(x, 6), round(y, 6))
                if key in z_lookup:
                    return z_lookup[key]
                return min(local_ext_coords, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)[2]
            
            site_ifc_faces = []
            
            # Create local 2D polygon for triangulation
            print(f"  Creating local polygon...")
            local_polygon_2d = Polygon([(x, y) for x, y, _ in local_ext_coords])
            if not local_polygon_2d.is_valid:
                print(f"  Polygon invalid, buffering...")
                local_polygon_2d = local_polygon_2d.buffer(0)
            
            if local_polygon_2d.is_empty:
                print(f"  Warning: Local polygon is empty, skipping site solid")
            else:
                # Create triangulated top faces
                print(f"  Triangulating top surface...")
                tri_list = list(triangulate(local_polygon_2d))
                print(f"  Generated {len(tri_list)} triangles")
                
                for tri in tri_list:
                    if not local_polygon_2d.contains(tri.centroid):
                        continue
                    
                    oriented_tri = orient(tri, sign=1.0)
                    tri_coords = list(oriented_tri.exterior.coords)[:-1]
                    tri_points = [
                        model.createIfcCartesianPoint([float(x), float(y), float(get_vertex_z(x, y))])
                        for x, y in tri_coords
                    ]
                    tri_loop = model.createIfcPolyLoop(tri_points)
                    face = model.createIfcFace([model.createIfcFaceOuterBound(tri_loop, True)])
                    site_ifc_faces.append(face)
                
                print(f"  Created {len(site_ifc_faces)} top faces")
                
                # Create side faces (skirt)
                print(f"  Creating side faces...")
                for i in range(len(local_ext_coords)):
                    p1 = local_ext_coords[i]
                    p2 = local_ext_coords[(i + 1) % len(local_ext_coords)]
                    
                    side_pts = [
                        model.createIfcCartesianPoint([float(p1[0]), float(p1[1]), float(p1[2])]),
                        model.createIfcCartesianPoint([float(p1[0]), float(p1[1]), float(local_base_elevation)]),
                        model.createIfcCartesianPoint([float(p2[0]), float(p2[1]), float(local_base_elevation)]),
                        model.createIfcCartesianPoint([float(p2[0]), float(p2[1]), float(p2[2])])
                    ]
                    side_loop = model.createIfcPolyLoop(side_pts)
                    side_face = model.createIfcFace([model.createIfcFaceOuterBound(side_loop, True)])
                    site_ifc_faces.append(side_face)
                
                # Create bottom face
                print(f"  Creating bottom face...")
                bot_points = [model.createIfcCartesianPoint([float(p[0]), float(p[1]), float(local_base_elevation)]) 
                              for p in reversed(local_ext_coords)]
                bot_loop = model.createIfcPolyLoop(bot_points)
                bot_face = model.createIfcFace([model.createIfcFaceOuterBound(bot_loop, True)])
                site_ifc_faces.append(bot_face)
                
                print(f"  Creating closed shell with {len(site_ifc_faces)} faces...")
                site_shell = model.createIfcClosedShell(site_ifc_faces)
                print(f"  Creating faceted brep...")
                site_solid = model.createIfcFacetedBrep(site_shell)
                
                site_rep = model.createIfcShapeRepresentation(
                    body_context, "Body", "Brep", [site_solid])
                site_terrain.Representation = model.createIfcProductDefinitionShape(
                    None, None, [site_rep])
                
                print(f"  Created site solid with {len(site_ifc_faces)} faces")
                
                # Add cadastre metadata using IFC schema best practices
                if cadastre_metadata:
                    # --- IfcSite schema attributes ---
                    # LandTitleNumber: Official land registration identifier (EGRID)
                    if cadastre_metadata.get('egrid'):
                        site.LandTitleNumber = cadastre_metadata['egrid']
                    
                    # LongName: Human-readable parcel identifier
                    if cadastre_metadata.get('parcel_number'):
                        site.LongName = f"{cadastre_metadata.get('canton', '')} {cadastre_metadata['parcel_number']}"
                    
                    # Description: Brief summary
                    site.Description = f"Swiss cadastral parcel in Canton {cadastre_metadata.get('canton', 'CH')}"
                    
                    # --- Pset_LandRegistration (IFC standard pset for land parcels) ---
                    pset_land = ifcopenshell.api.run("pset.add_pset", model, 
                                                      product=site, 
                                                      name="Pset_LandRegistration")
                    set_owner_history_on_pset(pset_land)
                    land_props = {
                        'LandID': cadastre_metadata.get('parcel_number', ''),
                        'LandTitleID': cadastre_metadata.get('egrid', ''),
                        'IsPermanentID': True  # EGRID is a permanent identifier
                    }
                    ifcopenshell.api.run("pset.edit_pset", model, 
                                          pset=pset_land, 
                                          properties=land_props)
                    print(f"  Added Pset_LandRegistration: {list(land_props.keys())}")
                    
                    # --- Qto_SiteBaseQuantities (IFC standard quantity set) ---
                    if cadastre_metadata.get('area_m2') or cadastre_metadata.get('perimeter_m'):
                        qto = ifcopenshell.api.run("pset.add_qto", model, 
                                                    product=site, 
                                                    name="Qto_SiteBaseQuantities")
                        set_owner_history_on_pset(qto)
                        quantities = {}
                        if cadastre_metadata.get('area_m2'):
                            quantities['GrossArea'] = cadastre_metadata['area_m2']
                        if cadastre_metadata.get('perimeter_m'):
                            quantities['GrossPerimeter'] = cadastre_metadata['perimeter_m']
                        
                        ifcopenshell.api.run("pset.edit_qto", model, 
                                              qto=qto, 
                                              properties=quantities)
                        print(f"  Added Qto_SiteBaseQuantities: {list(quantities.keys())}")
                    
                    # --- Site solid element description ---
                    site_terrain.Description = f"Site terrain solid - Parcel {cadastre_metadata.get('parcel_number', '')}"
                    
                    # --- Pset_SiteCommon (IFC standard - comprehensive site properties) ---
                    pset_common = ifcopenshell.api.run("pset.add_pset", model, 
                                                        product=site, 
                                                        name="Pset_SiteCommon")
                    set_owner_history_on_pset(pset_common)
                    
                    common_props = {}
                    if cadastre_metadata.get('local_id'):
                        common_props['Reference'] = cadastre_metadata['local_id']
                    if cadastre_metadata.get('area_m2'):
                        # TotalArea: Total planned area for the site
                        common_props['TotalArea'] = cadastre_metadata['area_m2']
                        # BuildableArea: Maximum buildable area (use total area as default if not specified)
                        # In Swiss cadastre, this might be different, but we use total area as fallback
                        common_props['BuildableArea'] = cadastre_metadata['area_m2']
                    
                    ifcopenshell.api.run("pset.edit_pset", model, 
                                          pset=pset_common, 
                                          properties=common_props)
                    print(f"  Added Pset_SiteCommon: {list(common_props.keys())}")
                    
                    # --- CPset_SwissCadastre (Custom property set - NOT in IFC schema) ---
                    if cadastre_metadata.get('geoportal_url') or cadastre_metadata.get('canton'):
                        pset_swiss = ifcopenshell.api.run("pset.add_pset", model, 
                                                           product=site, 
                                                           name="CPset_SwissCadastre")
                        set_owner_history_on_pset(pset_swiss)
                        swiss_props = {}
                        if cadastre_metadata.get('geoportal_url'):
                            swiss_props['GeoportalURL'] = cadastre_metadata['geoportal_url']
                        if cadastre_metadata.get('canton'):
                            swiss_props['Canton'] = cadastre_metadata['canton']
                        if cadastre_metadata.get('parcel_number'):
                            swiss_props['ParcelNumber'] = cadastre_metadata['parcel_number']
                        
                        ifcopenshell.api.run("pset.edit_pset", model, 
                                              pset=pset_swiss, 
                                              properties=swiss_props)
                        print(f"  Added CPset_SwissCadastre: {list(swiss_props.keys())}")
                    
        except Exception as e:
            import traceback
            print(f"  ERROR creating site solid: {e}")
            traceback.print_exc()
            print(f"  Continuing without site solid...")
    
    # Set OwnerHistory on all relationships that are missing it
    for entity in model:
        if hasattr(entity, 'OwnerHistory') and entity.OwnerHistory is None:
            try:
                entity.OwnerHistory = owner_history
            except Exception:
                pass  # Some entities may not accept OwnerHistory
    
    if return_model:
        print(f"\nIFC model created in memory")
        if terrain_triangles:
            print(f"  Terrain triangles: {len(terrain_triangles)}")
        if site_solid_data:
            print(f"  Site solid: created")
        return model, offset_x, offset_y, offset_z
    else:
        model.write(output_path)
        print(f"\nIFC file created: {output_path}")
        if terrain_triangles:
            print(f"  Terrain triangles: {len(terrain_triangles)}")
        if site_solid_data:
            print(f"  Site solid: created")
        return offset_x, offset_y, offset_z


def run_combined_terrain_workflow(
    egrid=None,
    center_x=None,
    center_y=None,
    radius=500.0,
    resolution=10.0,
    densify=2.0,
    attach_to_solid=False,
    include_terrain=True,
    include_site_solid=True,
    output_path="combined_terrain.ifc",
    return_model=False,
):
    """
    Run the combined terrain generation workflow using the existing processing pipeline.
    """
    if not egrid:
        raise ValueError("EGRID is required for combined terrain generation.")

    # Fetch site boundary from cadastre (required for combined workflow)
    cadastre_metadata = None
    site_geometry = None
    site_geometry, cadastre_metadata = fetch_boundary_by_egrid(egrid)
    if site_geometry is None:
        raise ValueError(f"Failed to fetch site boundary for EGRID {egrid}")
    centroid = site_geometry.centroid
    center_x = center_x if center_x is not None else centroid.x
    center_y = center_y if center_y is not None else centroid.y
    bounds = site_geometry.bounds
    print(f"Site bounds: E {bounds[0]:.1f}-{bounds[2]:.1f}, N {bounds[1]:.1f}-{bounds[3]:.1f}")
    print(f"Site centroid: E {center_x:.1f}, N {center_y:.1f}")

    # Initialize variables
    terrain_triangles = None
    site_solid_data = None
    circle_bounds = None
    terrain_coords = None
    terrain_elevations = None
    site_coords_3d = None
    smoothed_boundary_2d = None
    smoothed_boundary_z = None
    z_offset = None  # Will be calculated once if needed

    # Get site boundary 3D coordinates (needed for both terrain and site solid)
    ring = site_geometry.exterior
    distances = np.arange(0, ring.length, densify)
    if distances[-1] < ring.length:
        distances = np.append(distances, ring.length)

    site_points = [ring.interpolate(d) for d in distances]
    site_coords_2d = [(p.x, p.y) for p in site_points]

    print(f"\nFetching site boundary elevations ({len(site_coords_2d)} points)...")
    site_elevations = fetch_elevation_batch(site_coords_2d)
    site_coords_3d = [(x, y, z) for (x, y), z in zip(site_coords_2d, site_elevations)]

    # Create terrain if requested
    if include_terrain:
        # Create terrain grid
        terrain_coords, circle_bounds = create_circular_terrain_grid(
            center_x, center_y, radius=radius, resolution=resolution
        )

        if len(terrain_coords) == 0:
            raise ValueError("No points generated in circular area.")

        # Fetch terrain elevations
        print("\nFetching terrain elevations...")
        terrain_elevations = fetch_elevation_batch(terrain_coords)

        # Calculate height offset if site solid is also included (needed for alignment)
        if include_site_solid and z_offset is None:
            print("\nCalculating height offset for site solid...")
            z_offset = calculate_height_offset(
                site_geometry, site_coords_3d, terrain_coords, terrain_elevations
            )

        # Prepare site solid boundary for terrain attachment
        if include_site_solid and attach_to_solid:
            # Need to prepare smoothed boundary for attachment
            if z_offset is None:
                z_offset = 0.0
            (
                _,
                _,
                _,
                smoothed_boundary_2d,
                smoothed_boundary_z,
            ) = create_site_solid_coords(
                site_geometry, site_coords_3d, z_offset_adjustment=z_offset
            )
            print("  Using smoothed site solid boundary for terrain attachment")
            boundary_coords = smoothed_boundary_2d
            boundary_elevations = smoothed_boundary_z
        else:
            print("  Using raw API elevations for terrain boundary")
            boundary_coords = site_coords_2d
            boundary_elevations = site_elevations

        # Triangulate terrain with site cutout
        print("\nTriangulating terrain mesh (excluding site area)...")
        terrain_triangles = triangulate_terrain_with_cutout(
            terrain_coords,
            terrain_elevations,
            site_geometry,
            site_boundary_coords=boundary_coords,
            site_boundary_elevations=boundary_elevations,
        )
        print(f"Created {len(terrain_triangles)} terrain triangles")

    # Create site solid if requested
    if include_site_solid:
        # Use calculated z_offset if available, otherwise use 0.0
        if z_offset is None:
            z_offset = 0.0
            print("\nNo terrain provided, using site boundary elevations directly")

        # Prepare site solid data
        print("\nPreparing smoothed site solid...")
        (
            ext_coords,
            base_elevation,
            polygon_2d,
            smoothed_boundary_2d,
            smoothed_boundary_z,
        ) = create_site_solid_coords(
            site_geometry, site_coords_3d, z_offset_adjustment=z_offset
        )
        site_solid_data = {
            "ext_coords": ext_coords,
            "base_elevation": base_elevation,
            "polygon_2d": polygon_2d,
        }
        print(f"Site solid prepared with {len(ext_coords)} boundary points")

    # Determine bounds for IFC creation
    if circle_bounds is None:
        # Use site bounds if no terrain
        bounds = site_geometry.bounds
        circle_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])

    # Generate combined IFC
    print("\nGenerating IFC file...")
    if return_model:
        model, offset_x, offset_y, offset_z = create_combined_ifc(
            terrain_triangles,
            site_solid_data,
            output_path,
            circle_bounds,
            center_x,
            center_y,
            egrid=egrid,
            cadastre_metadata=cadastre_metadata,
            return_model=True,
        )
        print(f"\nSuccess! Combined terrain IFC model created in memory")
        return model, site_geometry, cadastre_metadata, (offset_x, offset_y, offset_z)
    else:
        create_combined_ifc(
            terrain_triangles,
            site_solid_data,
            output_path,
            circle_bounds,
            center_x,
            center_y,
            egrid=egrid,
            cadastre_metadata=cadastre_metadata,
            return_model=False,
        )
        print(f"\nSuccess! Combined terrain IFC saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create combined terrain with site cutout"
    )
    parser.add_argument("--egrid", help="EGRID number")
    parser.add_argument("--center-x", type=float, help="Center easting (EPSG:2056)")
    parser.add_argument("--center-y", type=float, help="Center northing (EPSG:2056)")
    parser.add_argument("--radius", type=float, default=500.0,
                        help="Radius of circular terrain area (meters), default: 500")
    parser.add_argument("--resolution", type=float, default=10.0,
                        help="Grid resolution (meters), default: 10")
    parser.add_argument("--densify", type=float, default=2.0,
                        help="Site boundary densification interval (meters), default: 2.0 (lower=faster, higher=more precise)")
    parser.add_argument("--attach-to-solid", action="store_true",
                        help="Attach terrain to smoothed site solid edges (less bumpy)")
    parser.add_argument("--output", default="combined_terrain.ifc",
                        help="Output IFC file path")
    
    args = parser.parse_args()
    
    try:
        run_combined_terrain_workflow(
            egrid=args.egrid,
            center_x=args.center_x,
            center_y=args.center_y,
            radius=args.radius,
            resolution=args.resolution,
            densify=args.densify,
            attach_to_solid=args.attach_to_solid,
            output_path=args.output,
        )
    except requests.Timeout as exc:
        print(f"Upstream request timed out: {exc}")
        sys.exit(1)
    except requests.HTTPError as exc:
        print(f"Upstream request failed: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
