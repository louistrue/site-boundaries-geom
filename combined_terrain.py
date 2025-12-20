#!/usr/bin/env python3
"""
Combined Terrain with Site Cutout

This script creates a single IFC file containing:
- Surrounding terrain mesh with a hole cut out for the site
- Site solid with smoothed surface, height-adjusted to align with terrain edges

Usage:
    python combined_terrain.py --egrid CH999979659148 --radius 500 --resolution 20 --output combined.ifc
"""

import numpy as np
import requests
import ifcopenshell
import ifcopenshell.api
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.ops import triangulate
from shapely.geometry.polygon import orient
import argparse
import time
import sys
import math


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


def fetch_elevation_batch(coords, batch_size=50, delay=0.1):
    """
    Fetch elevations for a list of coordinates via geo.admin.ch REST height service.
    """
    url = "https://api3.geo.admin.ch/rest/services/height"
    elevations = []
    failed_count = 0
    total = len(coords)
    
    print(f"Fetching elevations for {total} points...")
    
    for i, (x, y) in enumerate(coords):
        try:
            res = requests.get(url, params={"easting": x, "northing": y, "sr": "2056"}, timeout=10)
            res.raise_for_status()
            h = float(res.json()["height"])
            elevations.append(h)
        except Exception:
            elevations.append(elevations[-1] if elevations else 0.0)
            failed_count += 1
        
        if (i + 1) % batch_size == 0:
            pct = (i + 1) / total * 100
            print(f"  Progress: {i + 1}/{total} ({pct:.1f}%)")
            time.sleep(delay)
    
    if failed_count > 0:
        print(f"  Warning: {failed_count} points failed to fetch elevation")
    
    return elevations


def _geometry_has_z(coords):
    """Recursively check if a GeoJSON coordinate array contains Z values."""
    if not coords:
        return False
    first = coords[0]
    if isinstance(first, (float, int)):
        return len(coords) >= 3
    return any(_geometry_has_z(part) for part in coords)


def _collect_z_range_from_geojson(geometry):
    """Collect minimum and maximum Z values from a GeoJSON geometry."""
    z_values = []

    def _walk(values):
        if isinstance(values, (list, tuple)):
            if values and isinstance(values[0], (int, float)):
                if len(values) >= 3:
                    z_values.append(float(values[2]))
            else:
                for value in values:
                    _walk(value)

    _walk(geometry.get("coordinates", []))
    if not z_values:
        return None, None
    return min(z_values), max(z_values)


def _nearest_terrain_height(x, y, terrain_coords, terrain_elevations, default=0.0):
    """Find nearest terrain elevation for a point."""
    if not terrain_coords or not terrain_elevations:
        return default

    best_idx = 0
    best_dist = float("inf")
    for idx, (tx, ty) in enumerate(terrain_coords):
        dist = (tx - x) ** 2 + (ty - y) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return float(terrain_elevations[best_idx])


def _get_attr_float(attrs, keys):
    """Return the first available attribute from keys as float."""
    for key in keys:
        value = attrs.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _estimate_building_height(attrs, default_height=10.0):
    """Heuristic to estimate building height from swissBUILDINGS3D attributes."""
    height_keys = [
        "height",
        "HEIGHT",
        "H_GEB",
        "h_mean",
        "hmax",
        "z_max",
        "zmin",
        "zmax",
        "hroof",
        "roof_height",
        "buildingheight",
    ]
    height = _get_attr_float(attrs, height_keys)
    if height is None or height <= 0:
        return default_height
    return height


def _close_ring(ring):
    """Ensure a ring is closed by repeating the first coordinate if needed."""
    if not ring:
        return []
    if ring[0] != ring[-1]:
        return ring + [ring[0]]
    return ring


def fetch_buildings_in_radius(center_x, center_y, radius, layer="ch.swisstopo.swissbuildings3d_3_0", max_buildings=250):
    """
    Fetch swissBUILDINGS3D 3.0 Beta features in a circular search area.
    """
    bbox = (
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    )
    params = {
        "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "geometryType": "esriGeometryEnvelope",
        "layers": f"all:{layer}",
        "tolerance": 0,
        "returnGeometry": "true",
        "sr": "2056",
        "mapExtent": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "imageDisplay": "1000,1000,96",
        "geometryFormat": "geojson",
    }
    url = "https://api3.geo.admin.ch/rest/services/all/MapServer/identify"

    print(f"\nFetching swissBUILDINGS3D 3.0 Beta buildings within {radius}m...")
    try:
        response = requests.get(url, params=params, timeout=25)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(f"  Warning: Failed to fetch swissBUILDINGS3D data: {exc}")
        return []

    results = data.get("results", [])
    if not results:
        print("  No buildings returned for swissBUILDINGS3D query")
        return []

    circle = Point(center_x, center_y).buffer(radius)
    buildings = []
    for feature in results:
        geom_json = feature.get("geometry")
        if not geom_json:
            continue

        geom_shape = shape(geom_json)
        if not geom_shape.is_valid:
            geom_shape = geom_shape.buffer(0)
        if geom_shape.is_empty:
            continue
        if not circle.intersects(geom_shape):
            continue

        attrs = feature.get("properties", {}) or feature.get("attributes", {}) or {}
        buildings.append({
            "geometry": geom_shape,
            "geometry_json": geom_json,
            "attributes": attrs,
            "layer": layer
        })

    if not buildings:
        print("  No buildings intersect the requested circle")
        return []

    # Limit building count by distance to center for performance
    if max_buildings and len(buildings) > max_buildings:
        buildings.sort(key=lambda b: b["geometry"].centroid.distance(Point(center_x, center_y)))
        print(f"  Limiting buildings from {len(buildings)} to closest {max_buildings}")
        buildings = buildings[:max_buildings]

    print(f"  Prepared {len(buildings)} swissBUILDINGS3D features for processing")
    return buildings


def prepare_building_geometries(building_features, terrain_coords, terrain_elevations):
    """Normalize swissBUILDINGS3D geometries for IFC creation."""
    prepared = []
    for feature in building_features:
        geom_json = feature["geometry_json"]
        has_z = _geometry_has_z(geom_json.get("coordinates", []))
        min_z, max_z = _collect_z_range_from_geojson(geom_json) if has_z else (None, None)

        attrs = feature.get("attributes", {})
        geom_shape = feature["geometry"]

        base_z = min_z if min_z is not None else _nearest_terrain_height(
            geom_shape.centroid.x, geom_shape.centroid.y, terrain_coords, terrain_elevations, default=0.0
        )
        height = (max_z - min_z) if (min_z is not None and max_z is not None) else _estimate_building_height(attrs)

        building_id = (
            attrs.get("egid")
            or attrs.get("EGID")
            or attrs.get("id")
            or attrs.get("gml_id")
            or f"building_{len(prepared) + 1}"
        )

        prepared.append({
            "geometry_json": geom_json,
            "shape": geom_shape,
            "attributes": attrs,
            "layer": feature.get("layer", ""),
            "has_z": has_z,
            "base_z": base_z,
            "height": height,
            "min_z": min_z if min_z is not None else base_z,
            "max_z": max_z if max_z is not None else base_z + height,
            "id": str(building_id),
        })
    return prepared


def _geojson_polygons(geometry_json):
    """Return list of polygon coordinate arrays from a GeoJSON geometry."""
    geom_type = geometry_json.get("type")
    coords = geometry_json.get("coordinates", [])
    if geom_type == "Polygon":
        return [coords]
    if geom_type in ("MultiPolygon", "MultiSurface", "CompositeSurface"):
        return coords
    return []


def _ring_to_points(model, ring, offset_x, offset_y, offset_z, fallback_z):
    """Convert a coordinate ring to IFC Cartesian points in local coordinates."""
    local_points = []
    for coord in _close_ring(ring):
        z_val = coord[2] if len(coord) >= 3 else fallback_z
        local_points.append(model.createIfcCartesianPoint([
            float(coord[0] - offset_x),
            float(coord[1] - offset_y),
            float(float(z_val) - offset_z)
        ]))
    return local_points


def _create_faces_from_geojson(model, geometry_json, offset_x, offset_y, offset_z, fallback_z):
    """Create IFC faces from GeoJSON polygons (used when Z is present in geometry)."""
    faces = []
    for polygon in _geojson_polygons(geometry_json):
        if not polygon:
            continue
        try:
            outer_loop = model.createIfcPolyLoop(
                _ring_to_points(model, polygon[0], offset_x, offset_y, offset_z, fallback_z)
            )
            bounds = [model.createIfcFaceOuterBound(outer_loop, True)]
            for hole in polygon[1:]:
                hole_loop = model.createIfcPolyLoop(
                    _ring_to_points(model, hole, offset_x, offset_y, offset_z, fallback_z)
                )
                bounds.append(model.createIfcFaceInnerBound(hole_loop, True))
            faces.append(model.createIfcFace(bounds))
        except Exception as exc:
            print(f"    Warning: Failed to convert building polygon to IFC face: {exc}")
            continue
    return faces


def _create_extruded_building_shell(model, polygon, base_z, height, offset_x, offset_y, offset_z):
    """Create a closed shell for an extruded building volume."""
    if height is None or height <= 0:
        return None, 0
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None, 0

    ext_coords = list(polygon.exterior.coords)
    if len(ext_coords) < 4:
        return None, 0

    local_ext = [(float(x - offset_x), float(y - offset_y), float(base_z - offset_z)) for x, y in ext_coords]
    local_roof = [(float(x - offset_x), float(y - offset_y), float(base_z + height - offset_z)) for x, y in ext_coords]

    faces = []

    # Roof faces (triangulated)
    tri_list = list(triangulate(polygon))
    for tri in tri_list:
        if not polygon.contains(tri.centroid):
            continue
        tri_coords = list(tri.exterior.coords)[:-1]
        tri_points = [
            model.createIfcCartesianPoint([float(x - offset_x), float(y - offset_y), float(base_z + height - offset_z)])
            for x, y in tri_coords
        ]
        tri_loop = model.createIfcPolyLoop(tri_points)
        faces.append(model.createIfcFace([model.createIfcFaceOuterBound(tri_loop, True)]))

    # Side faces
    for i in range(len(local_ext) - 1):
        p1 = local_ext[i]
        p2 = local_ext[i + 1]
        r1 = local_roof[i]
        r2 = local_roof[i + 1]
        side_pts = [
            model.createIfcCartesianPoint([float(p1[0]), float(p1[1]), float(p1[2])]),
            model.createIfcCartesianPoint([float(r1[0]), float(r1[1]), float(r1[2])]),
            model.createIfcCartesianPoint([float(r2[0]), float(r2[1]), float(r2[2])]),
            model.createIfcCartesianPoint([float(p2[0]), float(p2[1]), float(p2[2])])
        ]
        side_loop = model.createIfcPolyLoop(side_pts)
        faces.append(model.createIfcFace([model.createIfcFaceOuterBound(side_loop, True)]))

    # Bottom face
    bot_points = [model.createIfcCartesianPoint([float(p[0]), float(p[1]), float(p[2])]) for p in reversed(local_ext)]
    bot_loop = model.createIfcPolyLoop(bot_points)
    faces.append(model.createIfcFace([model.createIfcFaceOuterBound(bot_loop, True)]))

    if not faces:
        return None, 0

    shell = model.createIfcClosedShell(faces)
    return shell, len(faces)


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
    # Apply smoothing (same as workflow.py)
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
                        center_x, center_y, egrid=None, cadastre_metadata=None, buildings=None):
    """
    Create a single IFC file with both terrain (with hole) and site solid.
    cadastre_metadata: dict with parcel info from cadastre API
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
    terrain_min_z = min(p[2] for tri in terrain_triangles for p in tri)
    building_min_z = terrain_min_z
    if buildings:
        building_min_z = min(b.get("min_z", terrain_min_z) for b in buildings)
    offset_z = round(min(terrain_min_z, building_min_z))
    
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
    
    # Create terrain mesh (with hole)
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
    
    # Add swissBUILDINGS3D buildings if provided
    if buildings:
        print(f"\nAdding {len(buildings)} swissBUILDINGS3D buildings to IFC...")
        for idx, building in enumerate(buildings, 1):
            building_name = f"Building_{building.get('id', idx)}"
            building_elem = ifcopenshell.api.run(
                "root.create_entity", model, ifc_class="IfcGeographicElement", name=building_name
            )
            building_elem.OwnerHistory = owner_history
            building_elem.ObjectType = "Building"
            building_elem.PredefinedType = "USERDEFINED"

            origin = model.createIfcCartesianPoint([0.0, 0.0, 0.0])
            axis = model.createIfcDirection([0.0, 0.0, 1.0])
            ref_dir = model.createIfcDirection([1.0, 0.0, 0.0])
            placement = model.createIfcAxis2Placement3D(origin, axis, ref_dir)
            building_loc = model.createIfcLocalPlacement(site.ObjectPlacement, placement)
            building_elem.ObjectPlacement = building_loc

            ifcopenshell.api.run("spatial.assign_container", model,
                                  products=[building_elem], relating_structure=site)

            rep = None
            if building.get("has_z"):
                faces = _create_faces_from_geojson(
                    model, building["geometry_json"], offset_x, offset_y, offset_z, building.get("base_z", 0.0)
                )
                if faces:
                    shell = model.createIfcOpenShell(faces)
                    shell_model = model.createIfcShellBasedSurfaceModel([shell])
                    shape_rep = model.createIfcShapeRepresentation(
                        body_context, "Body", "SurfaceModel", [shell_model]
                    )
                    rep = model.createIfcProductDefinitionShape(None, None, [shape_rep])
            else:
                polygons = []
                if isinstance(building.get("shape"), Polygon):
                    polygons = [building["shape"]]
                elif isinstance(building.get("shape"), MultiPolygon):
                    polygons = list(building["shape"].geoms)

                rep_items = []
                total_faces = 0
                for poly in polygons:
                    shell, face_count = _create_extruded_building_shell(
                        model, poly, building.get("base_z", 0.0), building.get("height", 0.0),
                        offset_x, offset_y, offset_z
                    )
                    total_faces += face_count
                    if shell:
                        rep_items.append(model.createIfcFacetedBrep(shell))

                if rep_items:
                    shape_rep = model.createIfcShapeRepresentation(
                        body_context, "Body", "Brep", rep_items
                    )
                    rep = model.createIfcProductDefinitionShape(None, None, [shape_rep])
                    print(f"  Building {idx}: created {total_faces} faces from {len(rep_items)} footprint(s)")

            if rep:
                building_elem.Representation = rep
            else:
                print(f"  Building {idx}: No geometry created, skipping representation")

            # Attach lightweight metadata
            attrs = building.get("attributes", {})
            pset_props = {}
            egid_value = attrs.get("egid") or attrs.get("EGID")
            if egid_value:
                pset_props["EGID"] = str(egid_value)
            if building.get("layer"):
                pset_props["SourceLayer"] = building["layer"]
            if building.get("height"):
                pset_props["HeightEstimate"] = float(building["height"])

            if pset_props:
                pset_buildings = ifcopenshell.api.run(
                    "pset.add_pset", model, product=building_elem, name="CPset_SwissBUILDINGS3D"
                )
                set_owner_history_on_pset(pset_buildings)
                ifcopenshell.api.run("pset.edit_pset", model, pset=pset_buildings, properties=pset_props)

        print(f"Completed adding {len(buildings)} buildings")
    
    # Set OwnerHistory on all relationships that are missing it
    for entity in model:
        if hasattr(entity, 'OwnerHistory') and entity.OwnerHistory is None:
            try:
                entity.OwnerHistory = owner_history
            except:
                pass  # Some entities may not accept OwnerHistory
    
    model.write(output_path)
    print(f"\nCombined IFC file created: {output_path}")
    print(f"  Terrain triangles: {len(terrain_triangles)}")
    if site_solid_data:
        print(f"  Site solid: created")
    
    return offset_x, offset_y, offset_z


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
    parser.add_argument("--densify", type=float, default=0.5,
                        help="Site boundary densification interval (meters), default: 0.5")
    parser.add_argument("--attach-to-solid", action="store_true",
                        help="Attach terrain to smoothed site solid edges (less bumpy)")
    parser.add_argument("--include-buildings", action="store_true",
                        help="Include swissBUILDINGS3D 3.0 Beta buildings within the terrain radius")
    parser.add_argument("--building-radius", type=float,
                        help="Radius for fetching swissBUILDINGS3D buildings (defaults to terrain radius)")
    parser.add_argument("--building-layer", default="ch.swisstopo.swissbuildings3d_3_0",
                        help="GeoAdmin layer name for swissBUILDINGS3D 3.0 Beta")
    parser.add_argument("--max-buildings", type=int, default=250,
                        help="Limit the number of buildings to include (closest to center)")
    parser.add_argument("--output", default="combined_terrain.ifc",
                        help="Output IFC file path")
    
    args = parser.parse_args()
    
    if not args.egrid and not (args.center_x and args.center_y):
        print("Error: Either --egrid or both --center-x and --center-y must be provided")
        sys.exit(1)
    
    # Fetch site boundary
    cadastre_metadata = None
    if args.egrid:
        site_geometry, cadastre_metadata = fetch_boundary_by_egrid(args.egrid)
        if site_geometry is None:
            print("Failed to fetch site boundary")
            sys.exit(1)
        centroid = site_geometry.centroid
        center_x = centroid.x
        center_y = centroid.y
        bounds = site_geometry.bounds
        print(f"Site bounds: E {bounds[0]:.1f}-{bounds[2]:.1f}, N {bounds[1]:.1f}-{bounds[3]:.1f}")
        print(f"Site centroid: E {center_x:.1f}, N {center_y:.1f}")
    else:
        center_x = args.center_x
        center_y = args.center_y
        site_geometry = None
    
    # Create terrain grid
    terrain_coords, circle_bounds = create_circular_terrain_grid(
        center_x, center_y, radius=args.radius, resolution=args.resolution
    )
    
    if len(terrain_coords) == 0:
        print("Error: No points generated in circular area")
        sys.exit(1)
    
    # Fetch terrain elevations
    print("\nFetching terrain elevations...")
    terrain_elevations = fetch_elevation_batch(terrain_coords)

    # Optionally fetch nearby buildings
    buildings_prepared = []
    building_radius = args.building_radius if args.building_radius else args.radius
    if args.include_buildings:
        building_features = fetch_buildings_in_radius(
            center_x, center_y, building_radius, layer=args.building_layer, max_buildings=args.max_buildings
        )
        if building_features:
            buildings_prepared = prepare_building_geometries(building_features, terrain_coords, terrain_elevations)
            print(f"Prepared {len(buildings_prepared)} buildings for IFC export")
    
    # Get site boundary 3D coordinates
    site_solid_data = None
    smoothed_boundary_2d = None
    smoothed_boundary_z = None
    
    if site_geometry:
        ring = site_geometry.exterior
        distances = np.arange(0, ring.length, args.densify)
        if distances[-1] < ring.length:
            distances = np.append(distances, ring.length)
        
        site_points = [ring.interpolate(d) for d in distances]
        site_coords_2d = [(p.x, p.y) for p in site_points]
        
        print(f"\nFetching site boundary elevations ({len(site_coords_2d)} points)...")
        site_elevations = fetch_elevation_batch(site_coords_2d)
        site_coords_3d = [(x, y, z) for (x, y), z in zip(site_coords_2d, site_elevations)]
        
        # Calculate height offset
        print("\nCalculating height offset for site solid...")
        z_offset = calculate_height_offset(
            site_geometry, site_coords_3d, terrain_coords, terrain_elevations
        )
        
        # Prepare site solid data BEFORE triangulation (so we can use smoothed boundary)
        print("\nPreparing smoothed site solid...")
        ext_coords, base_elevation, polygon_2d, smoothed_boundary_2d, smoothed_boundary_z = create_site_solid_coords(
            site_geometry, site_coords_3d, z_offset_adjustment=z_offset
        )
        site_solid_data = {
            'ext_coords': ext_coords,
            'base_elevation': base_elevation,
            'polygon_2d': polygon_2d
        }
        print(f"Site solid prepared with {len(ext_coords)} boundary points")
    else:
        print("Warning: No site geometry provided, skipping site solid")
        site_coords_3d = []
        z_offset = 0.0
    
    # Triangulate terrain with site cutout
    print("\nTriangulating terrain mesh (excluding site area)...")
    if site_geometry:
        # Choose boundary elevations based on --attach-to-solid option
        if args.attach_to_solid and smoothed_boundary_2d and smoothed_boundary_z:
            print("  Using smoothed site solid boundary for terrain attachment")
            boundary_coords = smoothed_boundary_2d
            boundary_elevations = smoothed_boundary_z
        else:
            print("  Using raw API elevations for terrain boundary")
            boundary_coords = site_coords_2d
            boundary_elevations = site_elevations
        
        # Pass site boundary coordinates to ensure precise cutout shape
        terrain_triangles = triangulate_terrain_with_cutout(
            terrain_coords, terrain_elevations, site_geometry,
            site_boundary_coords=boundary_coords,
            site_boundary_elevations=boundary_elevations
        )
    else:
        # No cutout if no site geometry
        terrain_triangles = []
        for i in range(len(terrain_coords) - 1):
            for j in range(len(terrain_coords) - 1):
                # Simple grid triangulation fallback
                pass
        print("Warning: Cannot create terrain without site geometry")
        sys.exit(1)
    
    print(f"Created {len(terrain_triangles)} terrain triangles")
    
    # Generate combined IFC
    print("\nGenerating combined IFC file...")
    create_combined_ifc(
        terrain_triangles, site_solid_data, args.output, circle_bounds,
        center_x, center_y, egrid=args.egrid, cadastre_metadata=cadastre_metadata, buildings=buildings_prepared
    )
    
    print(f"\nSuccess! Combined terrain IFC saved to: {args.output}")


if __name__ == "__main__":
    main()
