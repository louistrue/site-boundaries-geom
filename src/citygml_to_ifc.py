"""
Convert CityGML buildings to IFC following best practices

Creates proper IFC hierarchy with types, property sets, and complete 3D geometry.
"""

import logging
from typing import List, Optional, Tuple

import ifcopenshell
import ifcopenshell.api

from src.citygml_loader import CityGMLBuilding


logger = logging.getLogger(__name__)


def citygml_building_to_ifc(
    model: ifcopenshell.file,
    building: CityGMLBuilding,
    site: ifcopenshell.entity_instance,
    body_context,
    footprint_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
) -> Optional[ifcopenshell.entity_instance]:
    """
    Convert CityGML building to IFC following best practices
    
    Creates:
    - IfcBuilding with proper placement
    - IfcFacetedBrep representation from lod2Solid geometry
    - Property sets (Pset_BuildingCommon, CPset_SwissBuilding)
    - Footprint representation
    
    Args:
        model: IFC model
        building: CityGMLBuilding with complete 3D geometry
        site: Parent IfcSite
        body_context: IFC Body context
        footprint_context: IFC FootPrint context
        offset_x, offset_y, offset_z: Project origin offsets
        
    Returns:
        IfcBuilding entity or None if conversion fails
    """
    try:
        # Create IfcBuilding
        building_name = building.id[:30] if building.id else f"Building_{id(building)}"
        ifc_building = ifcopenshell.api.run(
            "root.create_entity",
            model,
            ifc_class="IfcBuilding",
            name=building_name
        )
        
        # Set building type if available
        if building.building_type:
            ifc_building.Description = building.building_type
        
        # Create placement relative to site
        origin = model.createIfcCartesianPoint([0.0, 0.0, 0.0])
        axis = model.createIfcDirection([0.0, 0.0, 1.0])
        ref_direction = model.createIfcDirection([1.0, 0.0, 0.0])
        axis2_placement = model.createIfcAxis2Placement3D(origin, axis, ref_direction)
        building_placement = model.createIfcLocalPlacement(
            site.ObjectPlacement,
            axis2_placement
        )
        ifc_building.ObjectPlacement = building_placement
        
        # Assign building to site
        ifcopenshell.api.run(
            "aggregate.assign_object",
            model,
            products=[ifc_building],
            relating_object=site
        )
        
        # Create representations
        representations = []
        
        # 1. Create 3D BRep representation from lod2Solid faces
        body_rep = _create_brep_from_citygml_faces(
            model, building, body_context, offset_x, offset_y, offset_z
        )
        if body_rep:
            representations.append(body_rep)
        else:
            logger.warning(f"Failed to create BRep for building {building.id}")
            return None
        
        # 2. Create 2D footprint representation
        footprint_rep = _create_footprint_from_citygml(
            model, building, footprint_context, offset_x, offset_y
        )
        if footprint_rep:
            representations.append(footprint_rep)
        
        # Assign representations
        if representations:
            ifc_building.Representation = model.createIfcProductDefinitionShape(
                None, None, representations
            )
        else:
            logger.warning(f"Building {building.id}: no representations created")
            return None
        
        # Add property sets following IFC best practices
        _add_building_properties(model, ifc_building, building)
        
        logger.debug(f"Created IfcBuilding: {building.id}")
        return ifc_building
        
    except Exception as e:
        logger.error(f"Failed to convert CityGML building {building.id}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def _create_brep_from_citygml_faces(
    model: ifcopenshell.file,
    building: CityGMLBuilding,
    body_context,
    offset_x: float,
    offset_y: float,
    offset_z: float,
) -> Optional[ifcopenshell.entity_instance]:
    """
    Create IfcFacetedBrep from CityGML lod2Solid faces
    
    Args:
        model: IFC model
        building: CityGMLBuilding with faces
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
        
    Returns:
        IfcShapeRepresentation or None
    """
    try:
        faces = []
        
        for face_points in building.faces:
            if len(face_points) < 3:
                continue
            
            # Remove duplicate closing point if present (first and last points are effectively the same)
            # Use tolerance for float comparisons
            points_to_use = face_points
            if len(face_points) > 3:
                first = face_points[0]
                last = face_points[-1]
                # Check if last point is effectively duplicate of first (within tolerance)
                tolerance = 1e-6
                if (abs(first[0] - last[0]) < tolerance and 
                    abs(first[1] - last[1]) < tolerance and 
                    abs(first[2] - last[2]) < tolerance):
                    points_to_use = face_points[:-1]
            
            # Apply offset and create IFC points
            local_points = [
                model.createIfcCartesianPoint([
                    float(p[0] - offset_x),
                    float(p[1] - offset_y),
                    float(p[2] - offset_z)
                ])
                for p in points_to_use
            ]
            
            if len(local_points) < 3:
                continue
            
            # Create poly loop and face
            loop = model.createIfcPolyLoop(local_points)
            face = model.createIfcFace([
                model.createIfcFaceOuterBound(loop, True)
            ])
            faces.append(face)
        
        if not faces:
            return None
        
        # Create closed shell and BRep
        shell = model.createIfcClosedShell(faces)
        brep = model.createIfcFacetedBrep(shell)
        
        rep = model.createIfcShapeRepresentation(
            body_context,
            "Body",
            "Brep",
            [brep]
        )
        
        return rep
        
    except Exception as e:
        logger.warning(f"Failed to create BRep from CityGML faces: {e}")
        return None


def _create_footprint_from_citygml(
    model: ifcopenshell.file,
    building: CityGMLBuilding,
    footprint_context,
    offset_x: float,
    offset_y: float,
) -> Optional[ifcopenshell.entity_instance]:
    """
    Create 2D footprint representation from CityGML building
    
    Uses ground surface faces to create footprint curve.
    
    Args:
        model: IFC model
        building: CityGMLBuilding
        footprint_context: IFC FootPrint context
        offset_x, offset_y: Project origin offsets
        
    Returns:
        IfcShapeRepresentation or None
    """
    try:
        # Find ground faces (horizontal faces at lowest Z)
        if building.z_min is None:
            return None
        
        ground_faces = []
        for face_points in building.faces:
            if len(face_points) < 3:
                continue
            
            # Check if face is horizontal (all Z values similar)
            z_values = [p[2] for p in face_points]
            z_range = max(z_values) - min(z_values)
            
            # Consider it ground if horizontal and near z_min
            if z_range < 0.5 and abs(min(z_values) - building.z_min) < 1.0:
                ground_faces.append(face_points)
        
        if not ground_faces:
            # Fallback: use centroid and create simple footprint
            if building.centroid:
                # Create a simple point representation
                point = model.createIfcCartesianPoint([
                    float(building.centroid[0] - offset_x),
                    float(building.centroid[1] - offset_y)
                ])
                geom = model.createIfcGeometricPointSet([point])
                rep = model.createIfcShapeRepresentation(
                    footprint_context,
                    "FootPrint",
                    "GeometricPointSet",
                    [geom]
                )
                return rep
            return None
        
        # Use largest ground face as footprint
        largest_face = max(ground_faces, key=len)
        
        # Remove duplicate closing point if present (first and last points are effectively the same)
        # Use tolerance for float comparisons
        points_to_use = largest_face
        if len(largest_face) > 3:
            first = largest_face[0]
            last = largest_face[-1]
            # Check if last point is effectively duplicate of first (within tolerance)
            tolerance = 1e-6
            if (abs(first[0] - last[0]) < tolerance and 
                abs(first[1] - last[1]) < tolerance):
                points_to_use = largest_face[:-1]
        
        # Create 2D polyline
        points_2d = [
            model.createIfcCartesianPoint([
                float(p[0] - offset_x),
                float(p[1] - offset_y)
            ])
            for p in points_to_use
        ]
        
        # Close the polyline if not already closed (check by comparing coordinates)
        if len(points_2d) > 0:
            first_point = points_2d[0]
            last_point = points_2d[-1]
            tolerance = 1e-6
            if (abs(first_point.Coordinates[0] - last_point.Coordinates[0]) > tolerance or
                abs(first_point.Coordinates[1] - last_point.Coordinates[1]) > tolerance):
                points_2d.append(points_2d[0])
        
        polyline = model.createIfcPolyLine(points_2d)
        
        rep = model.createIfcShapeRepresentation(
            footprint_context,
            "FootPrint",
            "Curve2D",
            [polyline]
        )
        
        return rep
        
    except Exception as e:
        logger.debug(f"Failed to create footprint from CityGML: {e}")
        return None


def _add_building_properties(
    model: ifcopenshell.file,
    ifc_building: ifcopenshell.entity_instance,
    building: CityGMLBuilding
):
    """
    Add property sets following IFC best practices
    
    Creates:
    - Pset_BuildingCommon (standard IFC properties)
    - CPset_SwissBuilding (Swiss-specific properties)
    
    Args:
        model: IFC model
        ifc_building: IfcBuilding entity
        building: CityGMLBuilding with attributes
    """
    # Pset_BuildingCommon
    pset_common = ifcopenshell.api.run(
        "pset.add_pset",
        model,
        product=ifc_building,
        name="Pset_BuildingCommon"
    )
    
    properties = {}
    
    # Height
    if building.height_max and building.height_min:
        properties["TotalHeight"] = building.height_max - building.height_min
    
    if building.z_max and building.z_min:
        properties["EaveHeight"] = building.z_max - building.z_min
    
    # Building type
    if building.building_type:
        properties["BuildingType"] = building.building_type
    
    if properties:
        ifcopenshell.api.run(
            "pset.edit_pset",
            model,
            pset=pset_common,
            properties=properties
        )
    
    # Swiss-specific properties
    if building.attributes:
        pset_swiss = ifcopenshell.api.run(
            "pset.add_pset",
            model,
            product=ifc_building,
            name="CPset_SwissBuilding"
        )
        
        swiss_props = {}
        
        # Map common Swiss attributes
        if "OBJEKTART" in building.attributes:
            swiss_props["ObjectType"] = building.attributes["OBJEKTART"]
        if "GEBAEUDE_NUTZUNG" in building.attributes:
            swiss_props["BuildingUse"] = building.attributes["GEBAEUDE_NUTZUNG"]
        if "HERKUNFT" in building.attributes:
            swiss_props["Source"] = building.attributes["HERKUNFT"]
        if "HERKUNFT_JAHR" in building.attributes:
            swiss_props["SourceYear"] = building.attributes["HERKUNFT_JAHR"]
        
        if swiss_props:
            ifcopenshell.api.run(
                "pset.edit_pset",
                model,
                pset=pset_swiss,
                properties=swiss_props
            )


def citygml_buildings_to_ifc(
    model: ifcopenshell.file,
    buildings: List[CityGMLBuilding],
    site: ifcopenshell.entity_instance,
    body_context,
    footprint_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
) -> List[ifcopenshell.entity_instance]:
    """
    Convert multiple CityGML buildings to IFC
    
    Args:
        model: IFC model
        buildings: List of CityGMLBuilding objects
        site: Parent IfcSite
        body_context: IFC Body context
        footprint_context: IFC FootPrint context
        offset_x, offset_y, offset_z: Project origin offsets
        
    Returns:
        List of IfcBuilding entities
    """
    ifc_buildings = []
    
    logger.info(f"Converting {len(buildings)} CityGML buildings to IFC...")
    
    for building in buildings:
        try:
            ifc_building = citygml_building_to_ifc(
                model, building, site, body_context, footprint_context,
                offset_x, offset_y, offset_z
            )
            if ifc_building:
                ifc_buildings.append(ifc_building)
        except Exception as e:
            logger.warning(f"Failed to convert building {building.id}: {e}")
            continue
    
    logger.info(f"Successfully converted {len(ifc_buildings)}/{len(buildings)} buildings")
    return ifc_buildings

