import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import Polygon, shape
from shapely.geometry.polygon import orient
from shapely.ops import triangulate
import ifcopenshell
import ifcopenshell.api
import argparse
import os
import requests
import json

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

def fetch_boundary_by_egrid(egrid):
    """
    Fetch the cadastral boundary (Polygon) for a given EGRID via geo.admin.ch API.
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
    
    # Take the first result
    feature = data["results"][0]
    geometry = shape(feature["geometry"])
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({
        "egrid": [egrid],
        "geometry": [geometry]
    }, crs="EPSG:2056")
    
    return gdf

def fetch_elevation_api(coords):
    """
    Fetch elevations for a list of coordinates via geo.admin.ch REST height service.
    """
    url = "https://api3.geo.admin.ch/rest/services/height"
    elevations = []
    
    # The API might have rate limits or batch limits. 
    # For many points, it's better to use local DEM if possible.
    # Here we'll do them one by one or in small batches if the API supports it.
    # Actually, the height API takes single points. 
    # Documentation says: ?easting=...&northing=...
    
    print(f"Sampling {len(coords)} points from height service...")
    for i, (x, y) in enumerate(coords):
        params = {
            "easting": x,
            "northing": y,
            "sr": "2056"
        }
        res = requests.get(url, params=params)
        res.raise_for_status()
        h = float(res.json()["height"])
        elevations.append(h)
        if i > 0 and i % 20 == 0:
            print(f"  Processed {i}/{len(coords)} points...")
            
    return elevations

def drape_cadastral_to_3d(gdf, dem_path=None, densify_interval=0.5):
    """
    Convert 2D cadastral polygons to 3D using terrain sampling.
    """
    if gdf.crs != "EPSG:2056":
        gdf = gdf.to_crs("EPSG:2056")
    
    geometries_3d = []
    
    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            # 1. Densify exterior ring
            ring = geom.exterior
            distances = np.arange(0, ring.length, densify_interval)
            if distances[-1] < ring.length:
                distances = np.append(distances, ring.length)
            
            points = [ring.interpolate(d) for d in distances]
            coords = [(p.x, p.y) for p in points]
            
            # 2. Sample elevations
            if dem_path and os.path.exists(dem_path):
                with rasterio.open(dem_path) as dem:
                    elevations = [val[0] for val in dem.sample(coords)]
            else:
                # Use API height service
                elevations = fetch_elevation_api(coords)
            
            coords_3d = [(x, y, z) for (x, y), z in zip(coords, elevations)]
            
            # 3. Handle holes
            interiors_3d = []
            for interior in geom.interiors:
                i_distances = np.arange(0, interior.length, densify_interval)
                if i_distances[-1] < interior.length:
                    i_distances = np.append(i_distances, interior.length)
                i_points = [interior.interpolate(d) for d in i_distances]
                i_coords = [(p.x, p.y) for p in i_points]
                
                if dem_path and os.path.exists(dem_path):
                    with rasterio.open(dem_path) as dem:
                        i_elevations = [val[0] for val in dem.sample(i_coords)]
                else:
                    i_elevations = fetch_elevation_api(i_coords)
                    
                i_coords_3d = [(x, y, z) for (x, y), z in zip(i_coords, i_elevations)]
                interiors_3d.append(i_coords_3d)

            geometries_3d.append(Polygon(coords_3d, interiors_3d))
        else:
            geometries_3d.append(geom)
            
    gdf_3d = gdf.copy()
    gdf_3d.geometry = geometries_3d
    return gdf_3d

def create_cadastral_ifc(gdf_3d, output_path, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """
    Create a georeferenced IFC file.
    """
    model = ifcopenshell.file(schema='IFC4')
    
    project = ifcopenshell.api.run("root.create_entity", model, 
                                    ifc_class="IfcProject", name="Cadastral Site")
    length_unit = ifcopenshell.api.run("unit.add_si_unit", model)
    ifcopenshell.api.run("unit.assign_unit", model, units=[length_unit])
    context = ifcopenshell.api.run("context.add_context", model, 
                                    context_type="Model")
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
    
    crs = model.createIfcProjectedCRS(
        Name="EPSG:2056",
        Description="Swiss LV95 / CH1903+",
        GeodeticDatum="CH1903+",
        VerticalDatum="LN02"
    )
    
    # Calculate origin if not provided
    if offset_x == 0.0 and len(gdf_3d) > 0:
        bounds = gdf_3d.total_bounds
        offset_x = round(bounds[0], -2) # Round to nearest 100m
        offset_y = round(bounds[1], -2)
        # Get min Z for offset_z
        z_coords = []
        for g in gdf_3d.geometry:
            z_coords.extend([c[2] for c in g.exterior.coords])
        offset_z = round(min(z_coords)) if z_coords else 0.0

    print(f"Using Project Origin: E={offset_x}, N={offset_y}, H={offset_z}")

    map_conversion = model.createIfcMapConversion(
        SourceCRS=context,
        TargetCRS=crs,
        Eastings=float(offset_x),
        Northings=float(offset_y),
        OrthogonalHeight=float(offset_z)
    )
    
    for idx, row in gdf_3d.iterrows():
        name = row.get('egrid', f"Parcel_{idx}")
        site = ifcopenshell.api.run("root.create_entity", model,
                                     ifc_class="IfcSite", 
                                     name=str(name))
        # Aggregate site to project for proper spatial hierarchy
        ifcopenshell.api.run("aggregate.assign_object", model, products=[site], relating_object=project)

        site_placement = ifcopenshell.api.run(
            "geometry.edit_object_placement", model, product=site
        )
        
        # 1. Keep PolyLine Footprint on Site for visibility (what the user liked)
        coords = [(float(x - offset_x), float(y - offset_y), float(z - offset_z)) 
                  for x, y, z in row.geometry.exterior.coords]
        footprint_points = [model.createIfcCartesianPoint([c[0], c[1]]) for c in coords]
        polyline = model.createIfcPolyLine(footprint_points)
        rep_site = model.createIfcShapeRepresentation(
            footprint_context, "FootPrint", "Curve2D", [polyline])
        site.Representation = model.createIfcProductDefinitionShape(
            None, None, [rep_site])

        # 2. Create IfcGeographicElement for proper Terrain Representation
        # Add Local Placement (Crucial for visibility!)
        terrain = ifcopenshell.api.run("root.create_entity", model,
                                        ifc_class="IfcGeographicElement",
                                        name=f"Terrain_{name}")
        terrain.PredefinedType = "TERRAIN"

        # Create placement relative to site
        # Since edit_object_placement doesn't support relative_to, we create it manually
        origin = model.createIfcCartesianPoint([0., 0., 0.])
        axis = model.createIfcDirection([0., 0., 1.])
        ref_direction = model.createIfcDirection([1., 0., 0.])
        axis2_placement = model.createIfcAxis2Placement3D(origin, axis, ref_direction)
        # site.ObjectPlacement is the placement to reference
        terrain_placement = model.createIfcLocalPlacement(site.ObjectPlacement, axis2_placement)
        terrain.ObjectPlacement = terrain_placement
        
        # Assign terrain to site
        ifcopenshell.api.run("spatial.assign_container", model,
                              products=[terrain], relating_structure=site)

        # 3. Create Solid Geometry (Manifold Brep) with a "Skirt"
        # We'll extrude the boundary down to a base elevation to make it a solid
        ext_coords = [(float(x - offset_x), float(y - offset_y), float(z - offset_z)) 
                      for x, y, z in row.geometry.exterior.coords]
        if ext_coords[0] == ext_coords[-1]:
            ext_coords = ext_coords[:-1]
            
        # Apply aggressive smoothing: fit plane for tilt, then heavily damp residual bumps
        z_values = [c[2] for c in ext_coords]
        plane_z = _best_fit_plane(ext_coords)

    # Smooth raw heights and residuals separately to kill small bumps
        smoothed_z = _circular_mean(z_values, window_size=9)
        residuals = [sz - pz for sz, pz in zip(smoothed_z, plane_z)]
        smoothed_residuals = _circular_mean(residuals, window_size=9)

        # Heavily attenuate residuals so the top stays flat but keeps overall orientation
        residual_scale = 0.2
        flattened_z = [pz + residual_scale * rz for pz, rz in zip(plane_z, smoothed_residuals)]

        ext_coords = [(ext_coords[i][0], ext_coords[i][1], flattened_z[i]) for i in range(len(ext_coords))]
            
        base_elevation = min(z for _, _, z in ext_coords) - 2.0 # 2 meters below lowest point
        
        # Create triangulated top faces to handle non-planar geometry
        polygon_2d = Polygon([(x, y) for x, y, _ in ext_coords])
        if not polygon_2d.is_valid:
            polygon_2d = polygon_2d.buffer(0)
        if polygon_2d.is_empty:
            print(f"Warning: invalid footprint for {name}, skipping geometry.")
            continue

        z_lookup = {(round(x, 6), round(y, 6)): z for x, y, z in ext_coords}

        def get_vertex_z(x, y):
            key = (round(x, 6), round(y, 6))
            if key in z_lookup:
                return z_lookup[key]
            return min(ext_coords, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)[2]

        faces = []

        for tri in triangulate(polygon_2d):
            # Filter out triangles that extend beyond the polygon boundary
            # Delaunay triangulation fills the convex hull, so we need to check containment
            if not polygon_2d.contains(tri.centroid):
                continue
            
            oriented_tri = orient(tri, sign=1.0)
            tri_coords = list(oriented_tri.exterior.coords)[:-1]  # drop closing vertex
            tri_points = [
                model.createIfcCartesianPoint([float(x), float(y), float(get_vertex_z(x, y))])
                for x, y in tri_coords
            ]
            tri_loop = model.createIfcPolyLoop(tri_points)
            faces.append(model.createIfcFace([model.createIfcFaceOuterBound(tri_loop, True)]))
        
        # Create side faces (the "skirt")
        for i in range(len(ext_coords)):
            p1 = ext_coords[i]
            p2 = ext_coords[(i + 1) % len(ext_coords)]
            
            # 4 points for the side quad
            side_pts = [
                model.createIfcCartesianPoint([p1[0], p1[1], p1[2]]),
                model.createIfcCartesianPoint([p1[0], p1[1], base_elevation]),
                model.createIfcCartesianPoint([p2[0], p2[1], base_elevation]),
                model.createIfcCartesianPoint([p2[0], p2[1], p2[2]])
            ]
            side_loop = model.createIfcPolyLoop(side_pts)
            side_face = model.createIfcFace([model.createIfcFaceOuterBound(side_loop, True)])
            faces.append(side_face)
            
        # Create bottom face
        bot_points = [model.createIfcCartesianPoint([p[0], p[1], base_elevation]) for p in reversed(ext_coords)]
        bot_loop = model.createIfcPolyLoop(bot_points)
        bot_face = model.createIfcFace([model.createIfcFaceOuterBound(bot_loop, True)])
        faces.append(bot_face)
        
        shell = model.createIfcClosedShell(faces)
        solid = model.createIfcFacetedBrep(shell)
        
        rep_terrain = model.createIfcShapeRepresentation(
            body_context, "Body", "Brep", [solid])
        terrain.Representation = model.createIfcProductDefinitionShape(
            None, None, [rep_terrain])

    model.write(output_path)
    print(f"IFC file successfully created at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Swiss Cadastral data to 3D IFC.")
    parser.add_argument("--egrid", help="EGRID number to fetch boundary for")
    parser.add_argument("--cadastral", help="Path to local cadastral file (optional if --egrid used)")
    parser.add_argument("--dem", help="Path to local DEM file (optional, will use API if omitted)")
    parser.add_argument("--output", default="output.ifc", help="Path to output IFC file")
    parser.add_argument("--densify", type=float, default=0.5, help="Densification interval in meters")
    parser.add_argument("--offset-x", type=float, default=0.0, help="Custom Easting offset")
    parser.add_argument("--offset-y", type=float, default=0.0, help="Custom Northing offset")
    parser.add_argument("--offset-z", type=float, default=0.0, help="Custom Height offset")
    
    args = parser.parse_args()
    
    if args.egrid:
        gdf = fetch_boundary_by_egrid(args.egrid)
    elif args.cadastral:
        gdf = gpd.read_file(args.cadastral)
    else:
        print("Error: Either --egrid or --cadastral must be provided.")
        return

    if gdf is None:
        return

    print(f"Processing geometries...")
    gdf_3d = drape_cadastral_to_3d(gdf, args.dem, args.densify)
    create_cadastral_ifc(gdf_3d, args.output, args.offset_x, args.offset_y, args.offset_z)

if __name__ == "__main__":
    main()
