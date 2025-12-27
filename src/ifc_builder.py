"""
IFC model builder

Creates IFC files with terrain, site solid, roads, trees, water, and buildings.
"""

# Patch dataclasses for Python 3.9 compatibility
# ifcopenshell uses slots=True in dataclass decorators, which is not supported in Python 3.9
# This monkey-patch removes the slots parameter before calling the original dataclass decorator
# TODO: Remove this patch when Python 3.9 support is dropped (Python 3.10+ supports slots=True)
import dataclasses
_original_dataclass = dataclasses.dataclass
def _patched_dataclass(*args, **kwargs):
    """Patch dataclass to ignore slots parameter for Python 3.9 compatibility."""
    kwargs.pop('slots', None)
    return _original_dataclass(*args, **kwargs)
dataclasses.dataclass = _patched_dataclass

import time
import logging
import ifcopenshell
import ifcopenshell.api
from shapely.geometry import Polygon
from shapely.ops import triangulate
from shapely.geometry.polygon import orient
from typing import List, Optional, Tuple, Dict

# Import from modular structure
from src.elevation import fetch_elevation_batch

logger = logging.getLogger(__name__)


def get_vertex_z(x: float, y: float, z_lookup: Dict[Tuple[float, float], float], 
                 local_ext_coords: List[Tuple[float, float, float]]) -> float:
    """Get Z coordinate for a vertex, using lookup or finding closest."""
    key = (round(x, 6), round(y, 6))
    if key in z_lookup:
        return z_lookup[key]
    return min(local_ext_coords, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)[2]


def create_combined_ifc(terrain_triangles, site_solid_data, output_path, bounds, 
                        center_x, center_y, egrid=None, cadastre_metadata=None,
                        roads=None, forest_points=None, waters=None, buildings=None,
                        railways=None, bridges=None,
                        base_elevation=0.0, road_recess_depth=0.0, return_model=False):
    """
    Create an IFC file with terrain (with hole) and/or site solid, optionally with roads, trees, water, and buildings.
    
    Args:
        terrain_triangles: List of triangles for terrain mesh, or None to skip terrain
        site_solid_data: Dict with site solid data, or None to skip site solid
        output_path: Path to output IFC file
        bounds: Bounding box (minx, miny, maxx, maxy)
        center_x, center_y: Center coordinates
        egrid: Swiss EGRID identifier
        cadastre_metadata: dict with parcel info from cadastre API
        roads: List of RoadFeature objects, or None to skip roads
        forest_points: List of ForestPoint objects from NFI data, or None to skip forest
        waters: List of WaterFeature objects, or None to skip water
        buildings: List of CityGMLBuilding objects, or None to skip buildings
        railways: List of RailwayFeature objects, or None to skip railways
        bridges: List of BridgeFeature objects, or None to skip bridges
        base_elevation: Base elevation for roads and trees
        road_recess_depth: How much terrain is recessed for roads (roads sit in recess)
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
    _minx, _miny, _maxx, _maxy = bounds  # Unpacked but individual values not used
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
                        model.createIfcCartesianPoint([float(x), float(y), float(get_vertex_z(x, y, z_lookup, local_ext_coords))])
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
                    if cadastre_metadata.get('egrid'):
                        site.LandTitleNumber = cadastre_metadata['egrid']
                    
                    if cadastre_metadata.get('parcel_number'):
                        site.LongName = f"{cadastre_metadata.get('canton', '')} {cadastre_metadata['parcel_number']}"
                    
                    site.Description = f"Swiss cadastral parcel in Canton {cadastre_metadata.get('canton', 'CH')}"
                    
                    # --- Pset_LandRegistration (IFC standard pset for land parcels) ---
                    pset_land = ifcopenshell.api.run("pset.add_pset", model, 
                                                      product=site, 
                                                      name="Pset_LandRegistration")
                    set_owner_history_on_pset(pset_land)
                    land_props = {
                        'LandID': cadastre_metadata.get('parcel_number', ''),
                        'LandTitleID': cadastre_metadata.get('egrid', ''),
                        'IsPermanentID': True
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
                        common_props['TotalArea'] = cadastre_metadata['area_m2']
                        common_props['BuildableArea'] = cadastre_metadata['area_m2']
                    
                    ifcopenshell.api.run("pset.edit_pset", model, 
                                          pset=pset_common, 
                                          properties=common_props)
                    print(f"  Added Pset_SiteCommon: {list(common_props.keys())}")
                    
                    # --- CPset_SwissCadastre (Custom property set) ---
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
    
    # Add roads if provided (with terrain-projected elevations)
    if roads:
        print(f"\nAdding {len(roads)} roads to IFC model...")
        try:
            from src.roads_vegetation_to_ifc import roads_to_ifc
            
            ifc_roads = roads_to_ifc(
                model, roads, site, body_context,
                offset_x, offset_y, offset_z,
                fetch_elevations_func=fetch_elevation_batch
            )
            print(f"  Added {len(ifc_roads)} road elements (in {road_recess_depth}m recess)")
        except Exception as e:
            import traceback
            print(f"  ERROR adding roads: {e}")
            traceback.print_exc()
            print(f"  Continuing without roads...")

    # Add forest trees if provided
    if forest_points:
        print(f"\nAdding {len(forest_points)} forest trees to IFC model...")
        try:
            from src.roads_vegetation_to_ifc import forest_to_ifc
            
            ifc_trees = forest_to_ifc(
                model, forest_points, site, body_context,
                offset_x, offset_y, offset_z
            )
            # Count deciduous vs coniferous using tree_type property
            deciduous = sum(1 for p in forest_points if hasattr(p, 'tree_type') and p.tree_type == "deciduous")
            coniferous = len(forest_points) - deciduous
            print(f"  Added {len(ifc_trees)} trees ({deciduous} deciduous, {coniferous} coniferous)")
        except Exception as e:
            import traceback
            print(f"  ERROR adding forest trees: {e}")
            traceback.print_exc()
            print(f"  Continuing without forest...")

    # Add water features if provided
    if waters:
        print(f"\nAdding {len(waters)} water features to IFC model...")
        try:
            from src.roads_vegetation_to_ifc import waters_to_ifc
            
            ifc_waters = waters_to_ifc(
                model, waters, site, body_context,
                offset_x, offset_y, offset_z,
                fetch_elevations_func=fetch_elevation_batch
            )
            print(f"  Added {len(ifc_waters)} water elements")
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"  Continuing without water...")

    # Add railways if provided
    if railways:
        print(f"\nAdding {len(railways)} railways to IFC model...")
        try:
            from src.roads_vegetation_to_ifc import railways_to_ifc
            
            ifc_railways = railways_to_ifc(
                model, railways, site, body_context,
                offset_x, offset_y, offset_z,
                fetch_elevations_func=fetch_elevation_batch
            )
            print(f"  Added {len(ifc_railways)} railway elements")
        except Exception as e:
            import traceback
            print(f"  ERROR adding railways: {e}")
            traceback.print_exc()
            print(f"  Continuing without railways...")

    # Add bridges if provided
    if bridges:
        print(f"\nAdding {len(bridges)} bridges to IFC model...")
        try:
            from src.roads_vegetation_to_ifc import bridges_to_ifc
            
            ifc_bridges = bridges_to_ifc(
                model, bridges, site, body_context,
                offset_x, offset_y, offset_z,
                fetch_elevations_func=fetch_elevation_batch
            )
            print(f"  Added {len(ifc_bridges)} bridge elements")
        except Exception as e:
            import traceback
            print(f"  ERROR adding bridges: {e}")
            traceback.print_exc()
            print(f"  Continuing without bridges...")

    # Add buildings if provided
    if buildings:
        print(f"\nAdding {len(buildings)} buildings to IFC model...")
        try:
            from src.citygml_to_ifc import citygml_buildings_to_ifc
            
            ifc_buildings = citygml_buildings_to_ifc(
                model, buildings, site, body_context, footprint_context,
                offset_x, offset_y, offset_z
            )
            print(f"  Added {len(ifc_buildings)} buildings")
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"  Continuing without buildings...")

    # Set OwnerHistory on all relationships that are missing it
    for entity in model:
        if hasattr(entity, 'OwnerHistory') and entity.OwnerHistory is None:
            try:
                entity.OwnerHistory = owner_history
            except Exception as e:
                # Some entities may not accept OwnerHistory (e.g., geometric representations)
                # This is expected behavior for certain IFC entity types
                pass
    
    if return_model:
        print(f"\nIFC model created in memory")
        if terrain_triangles:
            print(f"  Terrain triangles: {len(terrain_triangles)}")
        if site_solid_data:
            print(f"  Site solid: created")
        if roads:
            print(f"  Roads: {len(roads)}")
        return model, offset_x, offset_y, offset_z
    else:
        model.write(output_path)
        print(f"\nIFC file created: {output_path}")
        if terrain_triangles:
            print(f"  Terrain triangles: {len(terrain_triangles)}")
        if site_solid_data:
            print(f"  Site solid: created")
        return offset_x, offset_y, offset_z

