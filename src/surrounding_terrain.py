#!/usr/bin/env python3
"""
Terrain Loader for Swiss Cadastre Surroundings

This script fetches terrain data for the surroundings of a site from the Swiss
geo.admin.ch elevation API and creates a valid IFC terrain surface.

This is an additional module that does not modify the existing site_solid.py setup.

Usage:
    python -m src.surrounding_terrain --egrid CH999979659148 --radius 100 --resolution 10 --output terrain.ifc
"""

import numpy as np
import requests
import ifcopenshell
import ifcopenshell.api
from shapely.geometry import shape, Point
from shapely.ops import triangulate
import argparse
import time
import sys
import math


def fetch_boundary_by_egrid(egrid):
    """
    Fetch the cadastral boundary (Polygon) for a given EGRID via geo.admin.ch API.
    Returns Shapely geometry in EPSG:2056.
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
        return None
    
    feature = data["results"][0]
    return shape(feature["geometry"])


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


def create_circular_terrain_grid(center_x, center_y, radius=500.0, resolution=10.0):
    """
    Create a grid of points covering a circular terrain area.
    
    Args:
        center_x: Center easting coordinate
        center_y: Center northing coordinate
        radius: Radius of the circular area in meters
        resolution: Grid spacing in meters
    
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
    
    # Verify center point is included
    center_included = any(abs(x - center_x) < resolution and abs(y - center_y) < resolution 
                          for x, y in coords)
    
    print(f"Created circular grid: {len(coords)} points within {radius}m radius")
    print(f"Circle center: E {center_x:.1f}, N {center_y:.1f}")
    if center_included:
        print(f"  ✓ Center point included in grid")
    print(f"Coverage: E {minx:.1f} - {maxx:.1f}, N {miny:.1f} - {maxy:.1f}")
    print(f"Resolution: {resolution}m")
    
    return coords, (minx, miny, maxx, maxy)


def triangulate_points(coords, elevations):
    """
    Create triangulated mesh from irregular points using Delaunay triangulation.
    
    Args:
        coords: List of (x, y) coordinates
        elevations: List of elevations corresponding to coords
    
    Returns list of triangles, each as [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]
    """
    try:
        from scipy.spatial import Delaunay
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    if use_scipy:
        # Use scipy's Delaunay triangulation (more robust)
        points_2d = np.array(coords)
        tri = Delaunay(points_2d)
        
        triangles_3d = []
        for simplex in tri.simplices:
            i0, i1, i2 = simplex
            p0 = (*coords[i0], elevations[i0])
            p1 = (*coords[i1], elevations[i1])
            p2 = (*coords[i2], elevations[i2])
            triangles_3d.append([p0, p1, p2])
        
        return triangles_3d
    else:
        # Fallback: Use Shapely's triangulate
        from shapely.geometry import MultiPoint
        
        # Create MultiPoint from all coordinates
        multipoint = MultiPoint([Point(x, y) for x, y in coords])
        triangles_2d = triangulate(multipoint)
        
        # Map triangles back to 3D coordinates
        triangles_3d = []
        for tri_2d in triangles_2d:
            tri_coords_2d = list(tri_2d.exterior.coords)[:-1]  # Remove duplicate closing point
            
            if len(tri_coords_2d) == 3:
                # Find closest points in our coordinate list for each triangle vertex
                tri_3d = []
                for tx, ty in tri_coords_2d:
                    # Find the closest point in our grid
                    min_dist = float('inf')
                    closest_idx = 0
                    for idx, (cx, cy) in enumerate(coords):
                        dist = math.sqrt((tx - cx)**2 + (ty - cy)**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = idx
                    
                    # Use the elevation from the closest point
                    tri_3d.append((coords[closest_idx][0], coords[closest_idx][1], elevations[closest_idx]))
                
                if len(tri_3d) == 3:
                    triangles_3d.append(tri_3d)
        
        return triangles_3d


def create_terrain_ifc(triangles, output_path, bounds, egrid=None, site_geometry=None, center_x=None, center_y=None):
    """
    Create an IFC file with terrain as IfcGeographicElement.
    
    Args:
        triangles: List of triangles
        output_path: Path to save IFC file
        bounds: Bounding box (minx, miny, maxx, maxy)
        egrid: Optional EGRID for naming
        site_geometry: Optional Shapely geometry of site boundary
        center_x, center_y: Center coordinates for verification
    """
    model = ifcopenshell.file(schema='IFC4')
    
    # Project setup
    project = ifcopenshell.api.run("root.create_entity", model, 
                                    ifc_class="IfcProject", 
                                    name="Terrain Model")
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
    
    # Calculate offsets - use center as reference, round to nearest 100m
    minx, miny, maxx, maxy = bounds
    if center_x and center_y:
        # Center the project origin on the site center
        offset_x = round(center_x, -2)  # Round to nearest 100m
        offset_y = round(center_y, -2)
    else:
        # Fallback to bounds center
        offset_x = round((minx + maxx) / 2, -2)
        offset_y = round((miny + maxy) / 2, -2)
    
    min_z = min(p[2] for tri in triangles for p in tri)
    offset_z = round(min_z)
    
    print(f"Project Origin: E={offset_x}, N={offset_y}, H={offset_z}")
    if center_x and center_y:
        print(f"Site Center (relative to origin): E={center_x - offset_x:.1f}, N={center_y - offset_y:.1f}")
    
    model.createIfcMapConversion(
        SourceCRS=context,
        TargetCRS=crs,
        Eastings=float(offset_x),
        Northings=float(offset_y),
        OrthogonalHeight=float(offset_z)
    )
    
    # Create Site
    site_name = f"Site_{egrid}" if egrid else "Terrain_Site"
    site = ifcopenshell.api.run("root.create_entity", model, 
                                 ifc_class="IfcSite", 
                                 name=site_name)
    ifcopenshell.api.run("aggregate.assign_object", model, 
                          products=[site], relating_object=project)
    ifcopenshell.api.run("geometry.edit_object_placement", model, product=site)
    
    # Add site boundary footprint if provided
    if site_geometry and hasattr(site_geometry, 'exterior'):
        try:
            # Get average elevation for the boundary
            avg_z = sum(p[2] for tri in triangles for p in tri) / (len(triangles) * 3) - offset_z
            
            boundary_coords = list(site_geometry.exterior.coords)
            local_boundary = [
                (x - offset_x, y - offset_y, avg_z)
                for x, y in boundary_coords
            ]
            
            boundary_points = [
                model.createIfcCartesianPoint([float(p[0]), float(p[1]), float(p[2])])
                for p in local_boundary
            ]
            polyline = model.createIfcPolyLine(boundary_points)
            rep_footprint = model.createIfcShapeRepresentation(
                footprint_context, "FootPrint", "Curve2D", [polyline])
            site.Representation = model.createIfcProductDefinitionShape(
                None, None, [rep_footprint])
            print("  Added site boundary footprint to Site")
        except Exception as e:
            print(f"  Warning: Could not add site boundary: {e}")
    
    # Create IfcGeographicElement for terrain
    terrain_name = f"Terrain_{egrid}" if egrid else "Surrounding_Terrain"
    terrain = ifcopenshell.api.run("root.create_entity", model,
                                    ifc_class="IfcGeographicElement",
                                    name=terrain_name)
    terrain.PredefinedType = "TERRAIN"
    
    # Create placement
    origin = model.createIfcCartesianPoint([0., 0., 0.])
    axis = model.createIfcDirection([0., 0., 1.])
    ref_direction = model.createIfcDirection([1., 0., 0.])
    axis2_placement = model.createIfcAxis2Placement3D(origin, axis, ref_direction)
    terrain_placement = model.createIfcLocalPlacement(site.ObjectPlacement, axis2_placement)
    terrain.ObjectPlacement = terrain_placement
    
    ifcopenshell.api.run("spatial.assign_container", model,
                          products=[terrain], relating_structure=site)
    
    # Create triangle faces
    print(f"Creating {len(triangles)} triangular faces...")
    faces = []
    
    for tri in triangles:
        local_pts = [(p[0] - offset_x, p[1] - offset_y, p[2] - offset_z) for p in tri]
        tri_points = [
            model.createIfcCartesianPoint([float(p[0]), float(p[1]), float(p[2])])
            for p in local_pts
        ]
        tri_loop = model.createIfcPolyLoop(tri_points)
        face = model.createIfcFace([model.createIfcFaceOuterBound(tri_loop, True)])
        faces.append(face)
    
    # Create shell and representation using OpenShell for terrain surface
    shell = model.createIfcOpenShell(faces)
    shell_model = model.createIfcShellBasedSurfaceModel([shell])
    
    rep_terrain = model.createIfcShapeRepresentation(
        body_context, "Body", "SurfaceModel", [shell_model])
    terrain.Representation = model.createIfcProductDefinitionShape(
        None, None, [rep_terrain])
    
    model.write(output_path)
    print(f"\nTerrain IFC file created: {output_path}")
    print(f"  Total triangles: {len(triangles)}")
    
    return offset_x, offset_y, offset_z


def load_surrounding_terrain(egrid=None, center_coords=None,
                             radius=500.0, resolution=10.0,
                             output_path="terrain_surroundings.ifc"):
    """
    Main function to load terrain for site surroundings in a circular area.
    
    Args:
        egrid: Swiss EGRID to look up site boundary (center will be site centroid)
        center_coords: Alternative to egrid - (easting, northing) center point
        radius: Radius of circular area in meters (default: 500m)
        resolution: Grid resolution (meters) - lower = more detail but slower
        output_path: Where to save the IFC file
    
    Returns:
        Path to created IFC file
    """
    # Get center point
    site_geometry = None
    if egrid:
        site_geometry = fetch_boundary_by_egrid(egrid)
        if site_geometry is None:
            print("Failed to fetch site boundary")
            return None
        # Use centroid of the site as center - this ensures site is centered in circle
        centroid = site_geometry.centroid
        center_x = centroid.x
        center_y = centroid.y
        bounds = site_geometry.bounds
        print(f"Site bounds: E {bounds[0]:.1f}-{bounds[2]:.1f}, N {bounds[1]:.1f}-{bounds[3]:.1f}")
        print(f"Site centroid (circle center): E {center_x:.1f}, N {center_y:.1f}")
    elif center_coords:
        center_x, center_y = center_coords
    else:
        print("Error: Either egrid or center_coords must be provided")
        return None
    
    # Create circular terrain grid
    coords, circle_bounds = create_circular_terrain_grid(
        center_x, center_y, radius=radius, resolution=resolution
    )
    
    if len(coords) == 0:
        print("Error: No points generated in circular area")
        return None
    
    # Fetch elevations
    print("\nFetching elevation data from Swiss height service...")
    elevations = fetch_elevation_batch(coords)
    
    # Create triangulated mesh using Delaunay triangulation
    print("\nTriangulating terrain mesh...")
    triangles = triangulate_points(coords, elevations)
    print(f"Created {len(triangles)} triangles")
    
    # Generate IFC - pass site geometry and center for proper centering
    print("\nGenerating IFC file...")
    site_geom = site_geometry if egrid else None
    create_terrain_ifc(triangles, output_path, circle_bounds, 
                       egrid=egrid, site_geometry=site_geom,
                       center_x=center_x, center_y=center_y)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Load surrounding terrain from Swiss cadastre in a circular area and export to IFC"
    )
    parser.add_argument("--egrid",
                        help="EGRID number - terrain will be centered on site centroid")
    parser.add_argument("--center-x", type=float,
                        help="Center easting (EPSG:2056) if no EGRID")
    parser.add_argument("--center-y", type=float,
                        help="Center northing (EPSG:2056) if no EGRID")
    parser.add_argument("--radius", type=float, default=500.0,
                        help="Radius of circular terrain area (meters), default: 500")
    parser.add_argument("--resolution", type=float, default=10.0,
                        help="Grid resolution (meters), default: 10. Lower = more detail but slower.")
    parser.add_argument("--output", default="terrain_surroundings.ifc",
                        help="Output IFC file path")
    
    args = parser.parse_args()
    
    center_coords = None
    if args.center_x and args.center_y:
        center_coords = (args.center_x, args.center_y)
    
    if not args.egrid and not center_coords:
        print("Error: Either --egrid or both --center-x and --center-y must be provided")
        sys.exit(1)
    
    result = load_surrounding_terrain(
        egrid=args.egrid,
        center_coords=center_coords,
        radius=args.radius,
        resolution=args.resolution,
        output_path=args.output
    )
    
    if result:
        print(f"\nSuccess! Terrain IFC saved to: {result}")
    else:
        print("\nFailed to create terrain IFC")
        sys.exit(1)


if __name__ == "__main__":
    main()
