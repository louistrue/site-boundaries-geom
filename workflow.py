import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import Polygon, shape
import ifcopenshell
import ifcopenshell.api
import argparse
import os
import requests
import json

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
        
        # 1. Keep PolyLine Footprint on Site for visibility (what the user liked)
        coords = [(float(x - offset_x), float(y - offset_y), float(z - offset_z)) 
                  for x, y, z in row.geometry.exterior.coords]
        points = [model.createIfcCartesianPoint(list(c)) for c in coords]
        polyline = model.createIfcPolyLine(points)
        rep_site = model.createIfcShapeRepresentation(
            context, "FootPrint", "Curve2D", [polyline])
        site.Representation = model.createIfcProductDefinitionShape(
            None, None, [rep_site])

        # 2. Create IfcGeographicElement for proper Terrain Representation
        # Add Local Placement (Crucial for visibility!)
        placement = ifcopenshell.api.run("geometry.edit_object_placement", model, product=site) # Default relative to site
        terrain = ifcopenshell.api.run("root.create_entity", model,
                                        ifc_class="IfcGeographicElement",
                                        name=f"Terrain_{name}")
        terrain.ObjectPlacement = placement
        terrain.PredefinedType = "TERRAIN"
        
        # Assign terrain to site
        ifcopenshell.api.run("spatial.assign_container", model,
                              products=[terrain], relating_structure=site)

        # 3. Create Solid Geometry (Manifold Brep) with a "Skirt"
        # We'll extrude the boundary down to a base elevation to make it a solid
        ext_coords = [(float(x - offset_x), float(y - offset_y), float(z - offset_z)) 
                      for x, y, z in row.geometry.exterior.coords]
        if ext_coords[0] == ext_coords[-1]:
            ext_coords = ext_coords[:-1]
            
        # Apply median filter to smooth Z-coordinates of the solid terrain
        window_size = 5
        half_window = window_size // 2
        z_values = [c[2] for c in ext_coords]
        smoothed_z = []
        n = len(z_values)
        for i in range(n):
            # Use wrap-around for the closed loop
            window = [z_values[(i + j) % n] for j in range(-half_window, half_window + 1)]
            smoothed_z.append(float(np.median(window)))
        ext_coords = [(ext_coords[i][0], ext_coords[i][1], smoothed_z[i]) for i in range(n)]
            
        base_elevation = min(z for _, _, z in ext_coords) - 2.0 # 2 meters below lowest point
        
        # Create the top face
        top_points = [model.createIfcCartesianPoint(list(c)) for c in ext_coords]
        top_loop = model.createIfcPolyLoop(top_points)
        top_face = model.createIfcFace([model.createIfcFaceOuterBound(top_loop, True)])
        
        faces = [top_face]
        
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
        solid = model.createIfcManifoldSolidBrep(shell)
        
        rep_terrain = model.createIfcShapeRepresentation(
            context, "Body", "Brep", [solid])
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
