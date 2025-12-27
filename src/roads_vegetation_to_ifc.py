"""
Convert Swiss road and vegetation data to IFC elements

This module provides functions to convert RoadFeature and VegetationFeature objects
into IFC geographic element representations.
"""

import logging
from typing import List, Optional, Tuple

import ifcopenshell
import ifcopenshell.api
from shapely.geometry import LineString, Polygon

from src.loaders.road import RoadFeature
from src.loaders.water import WaterFeature
from typing import TYPE_CHECKING

# VegetationFeature is no longer used - replaced by TreeFeature from forest loader
# But we need a type alias for legacy function signatures
if TYPE_CHECKING:
    from typing import Protocol
    
    class VegetationFeature(Protocol):
        """Protocol for vegetation features (legacy compatibility)."""
        id: str
        geometry: object
        height: Optional[float]
        vegetation_type: str
        canopy_area: Optional[float]
        tree_species: Optional[str]
        density: Optional[float]
        is_coniferous: bool
else:
    # Dummy type for runtime (legacy functions not used)
    VegetationFeature = type('VegetationFeature', (), {})


logger = logging.getLogger(__name__)

# Default values
DEFAULT_ROAD_WIDTH = 5.0  # meters
DEFAULT_ROAD_THICKNESS = 0.15  # meters - road pavement thickness
DEFAULT_VEGETATION_HEIGHT = 3.0  # meters
MIN_VEGETATION_HEIGHT = 0.5  # meters
ROAD_SURFACE_OFFSET = 0.02  # meters - tiny offset above recessed terrain to avoid z-fighting

# Tree geometry defaults
DEFAULT_TREE_HEIGHT = 10.0  # meters - total tree height
DEFAULT_TREE_TRUNK_HEIGHT = 2.5  # meters - trunk height
DEFAULT_TREE_TRUNK_RADIUS = 0.3  # meters - trunk radius
DEFAULT_TREE_CANOPY_RADIUS = 4.0  # meters - canopy radius
DEFAULT_HEDGE_HEIGHT = 2.5  # meters - hedge height
DEFAULT_HEDGE_WIDTH = 1.5  # meters - hedge width (already buffered in loader)

# Forest tree defaults (from NFI data)
FOREST_TREE_HEIGHT_MIN = 8.0  # meters
FOREST_TREE_HEIGHT_MAX = 18.0  # meters
FOREST_TRUNK_HEIGHT_RATIO = 0.25  # trunk is 25% of total height
FOREST_TRUNK_RADIUS = 0.25  # meters
FOREST_CANOPY_RADIUS_RATIO = 0.35  # canopy radius is 35% of height

# Road color (dark asphalt)
ROAD_COLOR_RGB = (0.15, 0.15, 0.15)  # Dark gray/black

# Water colors - darker/greenish tint for natural water appearance
WATER_SURFACE_COLOR_RGB = (0.12, 0.28, 0.22)  # Dark teal/greenish - surface water
WATER_UNDERGROUND_COLOR_RGB = (0.08, 0.18, 0.25)  # Even darker blue-gray - underground water
WATER_DEPTH_OFFSET = -0.1  # meters - water sits slightly below terrain
UNDERGROUND_WATER_OFFSET = -1.5  # meters - underground water shown below terrain

# Tree colors
DECIDUOUS_COLOR_RGB = (0.2, 0.5, 0.2)  # Medium green
CONIFEROUS_COLOR_RGB = (0.1, 0.35, 0.15)  # Dark green
TRUNK_COLOR_RGB = (0.4, 0.25, 0.1)  # Brown


def _create_road_style(model: ifcopenshell.file) -> ifcopenshell.entity_instance:
    """Create a dark asphalt surface style for roads."""
    # Create surface color
    colour = model.createIfcColourRgb(None, *ROAD_COLOR_RGB)
    
    # Create surface style rendering
    rendering = model.createIfcSurfaceStyleRendering(
        colour,
        0.0,  # Transparency (0 = opaque)
        None,  # DiffuseColour
        None,  # TransmissionColour
        None,  # DiffuseTransmissionColour
        None,  # ReflectionColour
        None,  # SpecularColour
        None,  # SpecularHighlight
        "FLAT"  # ReflectanceMethod
    )
    
    # Create surface style
    surface_style = model.createIfcSurfaceStyle(
        "RoadAsphalt",
        "BOTH",
        [rendering]
    )
    
    return surface_style


def _create_water_style(model: ifcopenshell.file, is_underground: bool = False) -> ifcopenshell.entity_instance:
    """Create a water surface style with darker/greenish tint.
    
    Args:
        model: IFC model
        is_underground: If True, creates a more transparent style for underground water
    
    Returns:
        IfcSurfaceStyle for water
    """
    if is_underground:
        # Underground water: more transparent, darker color
        colour = model.createIfcColourRgb(None, *WATER_UNDERGROUND_COLOR_RGB)
        transparency = 0.6  # More transparent to indicate underground
        style_name = "WaterUnderground"
    else:
        # Surface water: darker greenish tint
        colour = model.createIfcColourRgb(None, *WATER_SURFACE_COLOR_RGB)
        transparency = 0.25  # Slightly transparent
        style_name = "WaterSurface"
    
    # Create surface style rendering
    rendering = model.createIfcSurfaceStyleRendering(
        colour,
        transparency,
        None,  # DiffuseColour
        None,  # TransmissionColour
        None,  # DiffuseTransmissionColour
        None,  # ReflectionColour
        None,  # SpecularColour
        None,  # SpecularHighlight
        "FLAT"  # ReflectanceMethod
    )
    
    # Create surface style
    surface_style = model.createIfcSurfaceStyle(
        style_name,
        "BOTH",
        [rendering]
    )
    
    return surface_style


def _apply_style_to_representation(
    model: ifcopenshell.file,
    representation: ifcopenshell.entity_instance,
    style: ifcopenshell.entity_instance
):
    """Apply a surface style to all items in a representation."""
    if representation is None:
        return
    
    for item in representation.Items:
        # Create styled item linking the geometry to the style
        model.createIfcStyledItem(item, [style], None)


def road_to_ifc(
    model: ifcopenshell.file,
    road: RoadFeature,
    site: ifcopenshell.entity_instance,
    body_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    elevations: Optional[List[float]] = None
) -> Optional[ifcopenshell.entity_instance]:
    """
    Convert a RoadFeature to an IfcGeographicElement with road geometry

    Args:
        model: IFC model
        road: RoadFeature to convert
        site: Parent IfcSite element
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
        elevations: List of elevations for each road coordinate (terrain-projected)

    Returns:
        IfcGeographicElement for road or None if failed
    """
    if road.geometry is None or road.geometry.is_empty:
        logger.warning(f"Skipping road {road.id}: no geometry")
        return None

    try:
        # Get coordinates from LineString
        coords = list(road.geometry.coords)
        if len(coords) < 2:
            logger.warning(f"Skipping road {road.id}: too few coordinates")
            return None
    except Exception as e:
        logger.warning(f"Skipping road {road.id}: invalid geometry - {e}")
        return None

    # Create IfcGeographicElement for road
    road_name = road.name or road.road_number or f"Road_{road.id}"
    ifc_road = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class="IfcGeographicElement",
        name=road_name
    )
    ifc_road.PredefinedType = "USERDEFINED"
    ifc_road.ObjectType = "ROAD"

    if road.road_class:
        ifc_road.Description = str(road.road_class)

    # Create placement relative to site
    origin = model.createIfcCartesianPoint([0.0, 0.0, 0.0])
    axis = model.createIfcDirection([0.0, 0.0, 1.0])
    ref_direction = model.createIfcDirection([1.0, 0.0, 0.0])
    axis2_placement = model.createIfcAxis2Placement3D(origin, axis, ref_direction)
    road_placement = model.createIfcLocalPlacement(
        site.ObjectPlacement,
        axis2_placement
    )
    ifc_road.ObjectPlacement = road_placement

    # Assign road to site
    ifcopenshell.api.run(
        "spatial.assign_container",
        model,
        products=[ifc_road],
        relating_structure=site
    )

    # Create representations
    representations = []

    # Roads sit AT terrain level - they extend DOWN by their thickness
    # This creates roads that embed into the terrain naturally
    z_adjustment = ROAD_SURFACE_OFFSET  # Tiny offset to sit just above terrain

    # Create 3D polyline representation - use elevations if provided
    if elevations and len(elevations) == len(coords):
        # Use terrain-projected elevations - road surface at terrain level
        road_points = [
            model.createIfcCartesianPoint([
                float(x - offset_x),
                float(y - offset_y),
                float(z - offset_z + z_adjustment)
            ])
            for (x, y), z in zip(coords, elevations)
        ]
        coords_3d = [(x, y, z + z_adjustment) for (x, y), z in zip(coords, elevations)]
    else:
        # Fallback to flat elevation (not terrain-projected)
        default_z = z_adjustment
        road_points = [
            model.createIfcCartesianPoint([
                float(x - offset_x),
                float(y - offset_y),
                default_z
            ])
            for x, y in coords
        ]
        coords_3d = [(x, y, offset_z + z_adjustment) for x, y in coords]

    polyline = model.createIfcPolyLine(road_points)
    road_rep = model.createIfcShapeRepresentation(
        body_context, "Body", "Curve3D", [polyline]
    )
    representations.append(road_rep)

    # Optionally create surface representation if width is available
    road_width = road.width if road.width and road.width > 0 else DEFAULT_ROAD_WIDTH
    if road_width > 0 and len(coords_3d) >= 2:
        try:
            # Create a simple surface representation by buffering the line
            # For simplicity, create a surface along the road centerline
            surface_rep = _create_road_surface_3d(
                model, body_context, coords_3d, road_width,
                offset_x, offset_y, offset_z
            )
            if surface_rep:
                representations.append(surface_rep)
        except Exception as e:
            logger.debug(f"Could not create road surface for {road.id}: {e}")

    # Assign representation
    if representations:
        ifc_road.Representation = model.createIfcProductDefinitionShape(
            None, None, representations
        )
        
        # Apply dark asphalt style to road surface
        road_style = _create_road_style(model)
        for rep in representations:
            _apply_style_to_representation(model, rep, road_style)
    else:
        logger.warning(f"Road {road.id}: no representations created")
        return None

    # Add properties
    _add_road_properties(model, ifc_road, road, road_width)

    logger.debug(f"Created IfcGeographicElement (ROAD): {road_name}")
    return ifc_road


def _create_road_surface_3d(
    model: ifcopenshell.file,
    body_context,
    coords_3d: List[Tuple[float, float, float]],
    width: float,
    offset_x: float,
    offset_y: float,
    offset_z: float,
    thickness: float = DEFAULT_ROAD_THICKNESS
) -> Optional[ifcopenshell.entity_instance]:
    """Create a 3D solid road representation with thickness following terrain"""
    if len(coords_3d) < 2:
        return None

    half_width = width / 2.0
    
    # Create solid road segments with top, bottom, and side faces
    all_faces = []
    
    for i in range(len(coords_3d) - 1):
        x1, y1, z1 = coords_3d[i]
        x2, y2, z2 = coords_3d[i + 1]

        # Calculate perpendicular direction (in 2D)
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        if length == 0:
            continue

        # Perpendicular vector (normalized)
        perp_x = -dy / length
        perp_y = dx / length

        # Top surface corners (at road level)
        t1 = (x1 + perp_x * half_width - offset_x, y1 + perp_y * half_width - offset_y, z1 - offset_z)
        t2 = (x2 + perp_x * half_width - offset_x, y2 + perp_y * half_width - offset_y, z2 - offset_z)
        t3 = (x2 - perp_x * half_width - offset_x, y2 - perp_y * half_width - offset_y, z2 - offset_z)
        t4 = (x1 - perp_x * half_width - offset_x, y1 - perp_y * half_width - offset_y, z1 - offset_z)
        
        # Bottom surface corners (thickness below road level)
        b1 = (t1[0], t1[1], t1[2] - thickness)
        b2 = (t2[0], t2[1], t2[2] - thickness)
        b3 = (t3[0], t3[1], t3[2] - thickness)
        b4 = (t4[0], t4[1], t4[2] - thickness)

        def make_face(pts):
            points = [model.createIfcCartesianPoint([float(p[0]), float(p[1]), float(p[2])]) for p in pts]
            poly_loop = model.createIfcPolyLoop(points)
            return model.createIfcFace([model.createIfcFaceOuterBound(poly_loop, True)])

        # Top face (road surface)
        all_faces.append(make_face([t1, t2, t3, t4]))
        
        # Bottom face
        all_faces.append(make_face([b4, b3, b2, b1]))  # Reversed for outward normal
        
        # Side faces
        all_faces.append(make_face([t1, t4, b4, b1]))  # Left side
        all_faces.append(make_face([t2, t1, b1, b2]))  # Start cap
        all_faces.append(make_face([t3, t2, b2, b3]))  # Right side
        all_faces.append(make_face([t4, t3, b3, b4]))  # End cap

    if not all_faces:
        return None

    # Create closed shell for solid representation
    try:
        closed_shell = model.createIfcClosedShell(all_faces)
        brep = model.createIfcFacetedBrep(closed_shell)
        return model.createIfcShapeRepresentation(
            body_context, "Body", "Brep", [brep]
        )
    except Exception as e:
        logger.warning(f"Could not create road solid, falling back to surface: {e}")
        # Fallback to surface representation
        shell = model.createIfcOpenShell(all_faces)
        shell_model = model.createIfcShellBasedSurfaceModel([shell])
        return model.createIfcShapeRepresentation(
            body_context, "Body", "SurfaceModel", [shell_model]
        )


def _add_road_properties(
    model: ifcopenshell.file,
    ifc_road: ifcopenshell.entity_instance,
    road: RoadFeature,
    width: float
):
    """Add property sets to road element"""
    pset = ifcopenshell.api.run(
        "pset.add_pset",
        model,
        product=ifc_road,
        name="Pset_RoadProperties"
    )

    properties = {
        "Width": width
    }

    if road.road_class:
        properties["RoadClass"] = road.road_class

    if road.name:
        properties["Name"] = road.name

    if road.road_number:
        properties["RoadNumber"] = road.road_number

    if road.surface_type:
        properties["SurfaceType"] = road.surface_type

    ifcopenshell.api.run(
        "pset.edit_pset",
        model,
        pset=pset,
        properties=properties
    )


def roads_to_ifc(
    model: ifcopenshell.file,
    roads: List[RoadFeature],
    site: ifcopenshell.entity_instance,
    body_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    fetch_elevations_func=None
) -> List[ifcopenshell.entity_instance]:
    """
    Convert multiple roads to IFC with terrain-projected elevations

    Args:
        model: IFC model
        roads: List of RoadFeature objects
        site: Parent IfcSite element
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
        fetch_elevations_func: Function to fetch elevations for coordinates (optional)

    Returns:
        List of created IfcGeographicElement entities
    """
    ifc_roads = []
    
    # Collect all road coordinates for batch elevation fetching
    if fetch_elevations_func:
        all_coords = []
        coord_counts = []
        for road in roads:
            if road.geometry and not road.geometry.is_empty:
                coords = list(road.geometry.coords)
                all_coords.extend([(x, y) for x, y in coords])
                coord_counts.append(len(coords))
            else:
                coord_counts.append(0)
        
        # Fetch elevations for all road points at once
        if all_coords:
            logger.info(f"Fetching elevations for {len(all_coords)} road points...")
            all_elevations = fetch_elevations_func(all_coords)
            
            # Split elevations back to individual roads
            idx = 0
            road_elevations = []
            for count in coord_counts:
                if count > 0:
                    road_elevations.append(all_elevations[idx:idx + count])
                    idx += count
                else:
                    road_elevations.append(None)
        else:
            road_elevations = [None] * len(roads)
    else:
        road_elevations = [None] * len(roads)
    
    # Create IFC elements for each road
    for road, elevations in zip(roads, road_elevations):
        ifc_road = road_to_ifc(
            model, road, site, body_context,
            offset_x, offset_y, offset_z, elevations
        )
        if ifc_road:
            ifc_roads.append(ifc_road)

    logger.info(f"Created {len(ifc_roads)}/{len(roads)} road elements")
    return ifc_roads


def water_to_ifc(
    model: ifcopenshell.file,
    water: WaterFeature,
    site: ifcopenshell.entity_instance,
    body_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    elevations: Optional[List[float]] = None
) -> Optional[ifcopenshell.entity_instance]:
    """
    Convert a WaterFeature to an IfcGeographicElement with water geometry
    
    Args:
        model: IFC model
        water: WaterFeature to convert
        site: Parent IfcSite element
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
        elevations: List of elevations for water coordinates (terrain-projected)
    
    Returns:
        IfcGeographicElement for water or None if failed
    """
    if water.geometry is None or water.geometry.is_empty:
        logger.warning(f"Skipping water {water.id}: no geometry")
        return None
    
    try:
        from shapely.geometry import LineString, Polygon
    except ImportError:
        logger.error("Shapely required for water conversion")
        return None
    
    # Create IfcGeographicElement for water
    water_name = water.name or f"{water.water_type.capitalize()}_{water.id}"
    ifc_water = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class="IfcGeographicElement",
        name=water_name
    )
    ifc_water.PredefinedType = "USERDEFINED"
    ifc_water.ObjectType = "WATER"
    
    if water.water_type:
        ifc_water.Description = water.water_type.replace("_", " ").title()
    
    # Check if this is underground water
    is_underground = water.is_underground
    
    # Create placement relative to site
    origin = model.createIfcCartesianPoint([0.0, 0.0, 0.0])
    axis = model.createIfcDirection([0.0, 0.0, 1.0])
    ref_direction = model.createIfcDirection([1.0, 0.0, 0.0])
    axis2_placement = model.createIfcAxis2Placement3D(origin, axis, ref_direction)
    water_placement = model.createIfcLocalPlacement(
        site.ObjectPlacement,
        axis2_placement
    )
    ifc_water.ObjectPlacement = water_placement
    
    # Assign water to site
    ifcopenshell.api.run(
        "spatial.assign_container",
        model,
        products=[ifc_water],
        relating_structure=site
    )
    
    # Underground water sits much lower; surface water sits slightly below terrain
    z_adjustment = UNDERGROUND_WATER_OFFSET if is_underground else WATER_DEPTH_OFFSET
    
    representations = []
    
    if isinstance(water.geometry, LineString):
        # Stream/river - create buffered surface
        coords = list(water.geometry.coords)
        if len(coords) < 2:
            logger.warning(f"Skipping water {water.id}: too few coordinates")
            return None
        
        # Get width
        width = water.width if water.width and water.width > 0 else 2.0
        
        # Create 3D coordinates (handle both 2D and 3D input coords)
        if elevations and len(elevations) == len(coords):
            coords_3d = [(c[0], c[1], z + z_adjustment) for c, z in zip(coords, elevations)]
        else:
            # Fallback to flat elevation
            default_z = offset_z + z_adjustment
            coords_3d = [(c[0], c[1], default_z) for c in coords]
        
        # Create surface representation by buffering the line
        surface_rep = _create_water_surface_3d(
            model, body_context, coords_3d, width,
            offset_x, offset_y, offset_z
        )
        if surface_rep:
            representations.append(surface_rep)
    
    elif isinstance(water.geometry, Polygon):
        # Lake - create polygon surface
        coords_2d = list(water.geometry.exterior.coords)
        if len(coords_2d) < 3:
            logger.warning(f"Skipping water {water.id}: invalid polygon")
            return None
        
        # Get elevation (handle both 2D and 3D input coords)
        if elevations and len(elevations) >= len(coords_2d):
            coords_3d = [(c[0], c[1], z + z_adjustment) for c, z in zip(coords_2d, elevations)]
        else:
            # Use single elevation from attributes or default
            elev = water.attributes.get('elevation', offset_z) if water.attributes else offset_z
            coords_3d = [(c[0], c[1], elev + z_adjustment) for c in coords_2d]
        
        # Create polygon surface
        surface_rep = _create_water_polygon_surface(
            model, body_context, coords_3d,
            offset_x, offset_y, offset_z
        )
        if surface_rep:
            representations.append(surface_rep)
    
    else:
        logger.warning(f"Skipping water {water.id}: unsupported geometry type")
        return None
    
    # Assign representation
    if representations:
        ifc_water.Representation = model.createIfcProductDefinitionShape(
            None, None, representations
        )
        
        # Apply water style (different for surface vs underground)
        water_style = _create_water_style(model, is_underground=is_underground)
        for rep in representations:
            _apply_style_to_representation(model, rep, water_style)
    else:
        logger.warning(f"Water {water.id}: no representations created")
        return None
    
    # Add properties
    _add_water_properties(model, ifc_water, water)
    
    logger.debug(f"Created IfcGeographicElement (WATER): {water_name}")
    return ifc_water


def _create_water_surface_3d(
    model: ifcopenshell.file,
    body_context,
    coords_3d: List[Tuple[float, float, float]],
    width: float,
    offset_x: float,
    offset_y: float,
    offset_z: float
) -> Optional[ifcopenshell.entity_instance]:
    """Create a 3D water surface by buffering a line."""
    if len(coords_3d) < 2:
        return None
    
    half_width = width / 2.0
    all_faces = []
    
    for i in range(len(coords_3d) - 1):
        x1, y1, z1 = coords_3d[i]
        x2, y2, z2 = coords_3d[i + 1]
        
        # Calculate perpendicular direction (in 2D)
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        if length == 0:
            continue
        
        # Perpendicular vector (normalized)
        perp_x = -dy / length
        perp_y = dx / length
        
        # Surface corners (flat water surface)
        p1 = (x1 + perp_x * half_width - offset_x, y1 + perp_y * half_width - offset_y, z1 - offset_z)
        p2 = (x2 + perp_x * half_width - offset_x, y2 + perp_y * half_width - offset_y, z2 - offset_z)
        p3 = (x2 - perp_x * half_width - offset_x, y2 - perp_y * half_width - offset_y, z2 - offset_z)
        p4 = (x1 - perp_x * half_width - offset_x, y1 - perp_y * half_width - offset_y, z1 - offset_z)
        
        def make_face(pts):
            points = [model.createIfcCartesianPoint([float(p[0]), float(p[1]), float(p[2])]) for p in pts]
            poly_loop = model.createIfcPolyLoop(points)
            return model.createIfcFace([model.createIfcFaceOuterBound(poly_loop, True)])
        
        # Single face for water surface
        all_faces.append(make_face([p1, p2, p3, p4]))
    
    if not all_faces:
        return None

    # Create open shell for surface (not a closed volume)
    open_shell = model.createIfcOpenShell(all_faces)
    surface_model = model.createIfcShellBasedSurfaceModel([open_shell])

    return model.createIfcShapeRepresentation(
        body_context, "Body", "SurfaceModel", [surface_model]
    )


def _create_water_polygon_surface(
    model: ifcopenshell.file,
    body_context,
    coords_3d: List[Tuple[float, float, float]],
    offset_x: float,
    offset_y: float,
    offset_z: float
) -> Optional[ifcopenshell.entity_instance]:
    """Create a flat polygon surface for lakes."""
    if len(coords_3d) < 3:
        return None
    
    # Create face from polygon
    points = [
        model.createIfcCartesianPoint([
            float(x - offset_x),
            float(y - offset_y),
            float(z - offset_z)
        ])
        for x, y, z in coords_3d[:-1]  # Exclude duplicate last point
    ]
    
    poly_loop = model.createIfcPolyLoop(points)
    face = model.createIfcFace([model.createIfcFaceOuterBound(poly_loop, True)])

    # Create open shell for surface (not a closed volume)
    open_shell = model.createIfcOpenShell([face])
    surface_model = model.createIfcShellBasedSurfaceModel([open_shell])

    return model.createIfcShapeRepresentation(
        body_context, "Body", "SurfaceModel", [surface_model]
    )


def _add_water_properties(
    model: ifcopenshell.file,
    ifc_water: ifcopenshell.entity_instance,
    water: WaterFeature
):
    """Add property sets to water element."""
    pset = ifcopenshell.api.run(
        "pset.add_pset",
        model,
        product=ifc_water,
        name="Pset_WaterProperties"
    )

    properties = {
        "IsUnderground": water.is_underground
    }

    if water.water_type:
        properties["WaterType"] = water.water_type

    if water.name:
        properties["Name"] = water.name

    if water.gewiss_number is not None:
        properties["GEWISSNumber"] = water.gewiss_number

    if water.width:
        properties["Width"] = water.width

    ifcopenshell.api.run(
        "pset.edit_pset",
        model,
        pset=pset,
        properties=properties
    )


def waters_to_ifc(
    model: ifcopenshell.file,
    waters: List[WaterFeature],
    site: ifcopenshell.entity_instance,
    body_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    fetch_elevations_func=None
) -> List[ifcopenshell.entity_instance]:
    """
    Convert multiple water features to IFC with terrain-projected elevations
    
    Args:
        model: IFC model
        waters: List of WaterFeature objects
        site: Parent IfcSite element
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
        fetch_elevations_func: Function to fetch elevations for coordinates (optional)
    
    Returns:
        List of created IfcGeographicElement entities
    """
    ifc_waters = []
    
    # Collect coordinates for batch elevation fetching
    if fetch_elevations_func:
        all_coords = []
        coord_counts = []
        for water in waters:
            if water.geometry and not water.geometry.is_empty:
                from shapely.geometry import LineString, Polygon
                if isinstance(water.geometry, LineString):
                    coords = list(water.geometry.coords)
                    all_coords.extend([(c[0], c[1]) for c in coords])
                    coord_counts.append(len(coords))
                elif isinstance(water.geometry, Polygon):
                    coords = list(water.geometry.exterior.coords)
                    all_coords.extend([(c[0], c[1]) for c in coords])
                    coord_counts.append(len(coords))
                else:
                    coord_counts.append(0)
            else:
                coord_counts.append(0)
        
        # Fetch elevations for all water points at once
        if all_coords:
            logger.info(f"Fetching elevations for {len(all_coords)} water points...")
            all_elevations = fetch_elevations_func(all_coords)
            
            # Split elevations back to individual waters
            idx = 0
            water_elevations = []
            for count in coord_counts:
                if count > 0:
                    water_elevations.append(all_elevations[idx:idx + count])
                    idx += count
                else:
                    water_elevations.append(None)
        else:
            water_elevations = [None] * len(waters)
    else:
        water_elevations = [None] * len(waters)
    
    # Create IFC elements for each water feature
    for water, elevations in zip(waters, water_elevations):
        ifc_water = water_to_ifc(
            model, water, site, body_context,
            offset_x, offset_y, offset_z, elevations
        )
        if ifc_water:
            ifc_waters.append(ifc_water)
    
    logger.info(f"Created {len(ifc_waters)}/{len(waters)} water elements")
    return ifc_waters


def _create_tree_geometry(
    model: ifcopenshell.file,
    body_context,
    center_x: float,
    center_y: float,
    base_z: float,
    tree_height: float = DEFAULT_TREE_HEIGHT,
    trunk_height: float = DEFAULT_TREE_TRUNK_HEIGHT,
    trunk_radius: float = DEFAULT_TREE_TRUNK_RADIUS,
    canopy_radius: float = DEFAULT_TREE_CANOPY_RADIUS
) -> List[ifcopenshell.entity_instance]:
    """
    Create 3D tree geometry: cylinder trunk + cone canopy
    
    Args:
        model: IFC model
        body_context: IFC Body context
        center_x, center_y: Tree center coordinates (already offset-adjusted)
        base_z: Base elevation (already offset-adjusted)
        tree_height: Total tree height
        trunk_height: Trunk height
        trunk_radius: Trunk radius
        canopy_radius: Canopy radius
    
    Returns:
        List of IfcExtrudedAreaSolid entities (trunk + canopy)
    """
    solids = []
    
    # Create trunk (cylinder)
    # Create circular profile for trunk
    import math
    num_segments = 8  # 8-sided circle for trunk
    trunk_profile_points = []
    for i in range(num_segments):
        angle = 2 * math.pi * i / num_segments
        x = center_x + trunk_radius * math.cos(angle)
        y = center_y + trunk_radius * math.sin(angle)
        trunk_profile_points.append(model.createIfcCartesianPoint([x, y]))
    # Close the circle
    trunk_profile_points.append(trunk_profile_points[0])
    
    trunk_polyline = model.createIfcPolyLine(trunk_profile_points)
    trunk_profile = model.createIfcArbitraryClosedProfileDef(
        "AREA",
        None,
        trunk_polyline
    )
    
    trunk_direction = model.createIfcDirection([0.0, 0.0, 1.0])
    trunk_position = model.createIfcAxis2Placement3D(
        model.createIfcCartesianPoint([0.0, 0.0, base_z]),
        trunk_direction,
        None
    )
    
    trunk_solid = model.createIfcExtrudedAreaSolid(
        trunk_profile,
        trunk_position,
        trunk_direction,
        trunk_height
    )
    solids.append(trunk_solid)
    
    # Create canopy (cone)
    # Top of trunk is at base_z + trunk_height
    canopy_base_z = base_z + trunk_height
    canopy_height = tree_height - trunk_height
    
    # Create circular profile for canopy base (larger radius)
    canopy_profile_points = []
    for i in range(num_segments * 2):  # More segments for smoother canopy
        angle = 2 * math.pi * i / (num_segments * 2)
        x = center_x + canopy_radius * math.cos(angle)
        y = center_y + canopy_radius * math.sin(angle)
        canopy_profile_points.append(model.createIfcCartesianPoint([x, y]))
    canopy_profile_points.append(canopy_profile_points[0])
    
    canopy_polyline = model.createIfcPolyLine(canopy_profile_points)
    canopy_profile = model.createIfcArbitraryClosedProfileDef(
        "AREA",
        None,
        canopy_polyline
    )
    
    canopy_direction = model.createIfcDirection([0.0, 0.0, 1.0])
    canopy_position = model.createIfcAxis2Placement3D(
        model.createIfcCartesianPoint([0.0, 0.0, canopy_base_z]),
        canopy_direction,
        None
    )
    
    # Note: Tapered extrusion (frustum) was planned but not implemented
    # The canopy is created as a simple extrusion instead
    
    # Create swept disk solid for canopy (simpler cone approximation)
    # Use IfcRevolvedAreaSolid to create a cone by revolving a triangle
    # Actually, let's use a simpler approach: create a faceted cone using IfcFacetedBrep
    # For now, use extruded solid with tapered profile (frustum)
    canopy_solid = model.createIfcExtrudedAreaSolid(
        canopy_profile,
        canopy_position,
        canopy_direction,
        canopy_height
    )
    solids.append(canopy_solid)
    
    return solids


def tree_to_ifc(
    model: ifcopenshell.file,
    vegetation: VegetationFeature,
    site: ifcopenshell.entity_instance,
    body_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    base_elevation: float = 0.0
) -> Optional[ifcopenshell.entity_instance]:
    """
    Convert a tree (Point geometry) to an IfcGeographicElement with 3D tree geometry
    
    Args:
        model: IFC model
        vegetation: VegetationFeature with Point geometry (tree)
        site: Parent IfcSite element
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
        base_elevation: Base elevation for tree (ground level)
    
    Returns:
        IfcGeographicElement for tree or None if failed
    """
    if vegetation.geometry is None or vegetation.geometry.is_empty:
        logger.warning(f"Skipping tree {vegetation.id}: no geometry")
        return None
    
    # Get tree center from geometry centroid
    try:
        centroid = vegetation.geometry.centroid
        center_x = float(centroid.x - offset_x)
        center_y = float(centroid.y - offset_y)
    except Exception as e:
        logger.warning(f"Skipping tree {vegetation.id}: invalid geometry - {e}")
        return None
    
    # Determine height to use
    if vegetation.height and vegetation.height > MIN_VEGETATION_HEIGHT:
        tree_height = vegetation.height
        height_source = "actual"
    else:
        tree_height = DEFAULT_TREE_HEIGHT
        height_source = "default"
    
    # Create IfcGeographicElement for tree
    tree_name = f"Tree_{vegetation.id}"
    if vegetation.vegetation_type:
        tree_name = f"{vegetation.vegetation_type}_{vegetation.id}"
    
    ifc_tree = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class="IfcGeographicElement",
        name=tree_name
    )
    ifc_tree.PredefinedType = "USERDEFINED"
    ifc_tree.ObjectType = "TREE"
    
    if vegetation.vegetation_type:
        ifc_tree.Description = str(vegetation.vegetation_type)
    
    # Create placement relative to site
    origin = model.createIfcCartesianPoint([0.0, 0.0, 0.0])
    axis = model.createIfcDirection([0.0, 0.0, 1.0])
    ref_direction = model.createIfcDirection([1.0, 0.0, 0.0])
    axis2_placement = model.createIfcAxis2Placement3D(origin, axis, ref_direction)
    tree_placement = model.createIfcLocalPlacement(
        site.ObjectPlacement,
        axis2_placement
    )
    ifc_tree.ObjectPlacement = tree_placement
    
    # Assign tree to site
    ifcopenshell.api.run(
        "spatial.assign_container",
        model,
        products=[ifc_tree],
        relating_structure=site
    )
    
    # Create 3D tree geometry
    base_z = float(base_elevation - offset_z)
    
    # Calculate canopy radius from geometry if available
    canopy_radius = DEFAULT_TREE_CANOPY_RADIUS
    if vegetation.canopy_area:
        # Estimate radius from area: area = π * r²
        import math
        canopy_radius = math.sqrt(vegetation.canopy_area / math.pi)
        canopy_radius = max(2.0, min(canopy_radius, 8.0))  # Clamp between 2-8m
    
    tree_solids = _create_tree_geometry(
        model, body_context,
        center_x, center_y, base_z,
        tree_height=tree_height,
        canopy_radius=canopy_radius
    )
    
    # Create representation with multiple solids
    body_rep = model.createIfcShapeRepresentation(
        body_context, "Body", "SweptSolid", tree_solids
    )
    
    ifc_tree.Representation = model.createIfcProductDefinitionShape(
        None, None, [body_rep]
    )
    
    # Add properties
    _add_vegetation_properties(model, ifc_tree, vegetation, tree_height, height_source)
    
    logger.debug(f"Created IfcGeographicElement (TREE): {tree_name}")
    return ifc_tree


def hedge_to_ifc(
    model: ifcopenshell.file,
    vegetation: VegetationFeature,
    site: ifcopenshell.entity_instance,
    body_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    base_elevation: float = 0.0
) -> Optional[ifcopenshell.entity_instance]:
    """
    Convert a hedge (LineString geometry, buffered to Polygon) to an IfcGeographicElement
    
    Args:
        model: IFC model
        vegetation: VegetationFeature with Polygon geometry (buffered hedge)
        site: Parent IfcSite element
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
        base_elevation: Base elevation for hedge (ground level)
    
    Returns:
        IfcGeographicElement for hedge or None if failed
    """
    if vegetation.geometry is None or vegetation.geometry.is_empty:
        logger.warning(f"Skipping hedge {vegetation.id}: no geometry")
        return None
    
    try:
        coords = list(vegetation.geometry.exterior.coords)
        if len(coords) < 3:
            logger.warning(f"Skipping hedge {vegetation.id}: too few coordinates")
            return None
    except Exception as e:
        logger.warning(f"Skipping hedge {vegetation.id}: invalid geometry - {e}")
        return None
    
    # Determine height to use
    if vegetation.height and vegetation.height > MIN_VEGETATION_HEIGHT:
        height_used = vegetation.height
        height_source = "actual"
    else:
        height_used = DEFAULT_HEDGE_HEIGHT
        height_source = "default"
    
    # Create IfcGeographicElement for hedge
    hedge_name = f"Hedge_{vegetation.id}"
    if vegetation.vegetation_type:
        hedge_name = f"{vegetation.vegetation_type}_{vegetation.id}"
    
    ifc_hedge = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class="IfcGeographicElement",
        name=hedge_name
    )
    ifc_hedge.PredefinedType = "USERDEFINED"
    ifc_hedge.ObjectType = "HEDGE"
    
    if vegetation.vegetation_type:
        ifc_hedge.Description = str(vegetation.vegetation_type)
    
    # Create placement relative to site
    origin = model.createIfcCartesianPoint([0.0, 0.0, 0.0])
    axis = model.createIfcDirection([0.0, 0.0, 1.0])
    ref_direction = model.createIfcDirection([1.0, 0.0, 0.0])
    axis2_placement = model.createIfcAxis2Placement3D(origin, axis, ref_direction)
    hedge_placement = model.createIfcLocalPlacement(
        site.ObjectPlacement,
        axis2_placement
    )
    ifc_hedge.ObjectPlacement = hedge_placement
    
    # Assign hedge to site
    ifcopenshell.api.run(
        "spatial.assign_container",
        model,
        products=[ifc_hedge],
        relating_structure=site
    )
    
    # Create representations
    # Remove duplicate closing point
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    
    if len(coords) < 3:
        logger.warning(f"Hedge {vegetation.id}: too few unique coordinates")
        return None
    
    # Create extruded solid representation
    base_z = float(base_elevation - offset_z)
    
    # Create footprint polygon
    profile_points = [
        model.createIfcCartesianPoint([
            float(x - offset_x),
            float(y - offset_y)
        ])
        for x, y in coords
    ]
    # Close the profile
    profile_points.append(profile_points[0])
    
    polyline = model.createIfcPolyLine(profile_points)
    profile = model.createIfcArbitraryClosedProfileDef(
        "AREA",
        None,
        polyline
    )
    
    # Create extrusion direction
    direction = model.createIfcDirection([0.0, 0.0, 1.0])
    position = model.createIfcAxis2Placement3D(
        model.createIfcCartesianPoint([0.0, 0.0, base_z]),
        direction,
        None
    )
    
    # Create extruded area solid
    extruded_solid = model.createIfcExtrudedAreaSolid(
        profile,
        position,
        direction,
        height_used
    )
    
    # Create representation
    body_rep = model.createIfcShapeRepresentation(
        body_context, "Body", "SweptSolid", [extruded_solid]
    )
    
    ifc_hedge.Representation = model.createIfcProductDefinitionShape(
        None, None, [body_rep]
    )
    
    # Add properties
    _add_vegetation_properties(model, ifc_hedge, vegetation, height_used, height_source)
    
    logger.debug(f"Created IfcGeographicElement (HEDGE): {hedge_name}")
    return ifc_hedge


def vegetation_to_ifc(
    model: ifcopenshell.file,
    vegetation: VegetationFeature,
    site: ifcopenshell.entity_instance,
    body_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    base_elevation: float = 0.0
) -> Optional[ifcopenshell.entity_instance]:
    """
    Convert a VegetationFeature to an IfcGeographicElement with vegetation geometry
    Routes to appropriate converter based on geometry type (tree, hedge, or generic)

    Args:
        model: IFC model
        vegetation: VegetationFeature to convert
        site: Parent IfcSite element
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
        base_elevation: Base elevation for vegetation (ground level)

    Returns:
        IfcGeographicElement for vegetation or None if failed
    """
    if vegetation.geometry is None or vegetation.geometry.is_empty:
        logger.warning(f"Skipping vegetation {vegetation.id}: no geometry")
        return None
    
    # Route to appropriate converter based on original geometry type
    original_type = getattr(vegetation, 'original_geometry_type', None)
    veg_type = vegetation.vegetation_type or ""
    
    # Individual trees (Point geometry)
    if original_type == "Point" or "tree" in veg_type.lower():
        return tree_to_ifc(
            model, vegetation, site, body_context,
            offset_x, offset_y, offset_z, base_elevation
        )
    
    # Hedges (LineString geometry)
    if original_type == "LineString" or "hedge" in veg_type.lower():
        return hedge_to_ifc(
            model, vegetation, site, body_context,
            offset_x, offset_y, offset_z, base_elevation
        )
    
    # Generic vegetation (Polygon geometry) - use original implementation
    try:
        coords = list(vegetation.geometry.exterior.coords)
        if len(coords) < 3:
            logger.warning(f"Skipping vegetation {vegetation.id}: too few coordinates")
            return None
    except Exception as e:
        logger.warning(f"Skipping vegetation {vegetation.id}: invalid geometry - {e}")
        return None

    # Determine height to use
    if vegetation.height and vegetation.height > MIN_VEGETATION_HEIGHT:
        height_used = vegetation.height
        height_source = "actual"
    else:
        height_used = DEFAULT_VEGETATION_HEIGHT
        height_source = "default"

    # Create IfcGeographicElement for vegetation
    veg_name = f"Vegetation_{vegetation.id}"
    if vegetation.vegetation_type:
        veg_name = f"{vegetation.vegetation_type}_{vegetation.id}"

    ifc_vegetation = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class="IfcGeographicElement",
        name=veg_name
    )
    ifc_vegetation.PredefinedType = "USERDEFINED"
    
    # Set ObjectType based on vegetation type
    if vegetation.vegetation_type:
        if "tree" in vegetation.vegetation_type.lower():
            ifc_vegetation.ObjectType = "TREE"
        else:
            ifc_vegetation.ObjectType = "VEGETATION"
    else:
        ifc_vegetation.ObjectType = "VEGETATION"

    if vegetation.vegetation_type:
        ifc_vegetation.Description = str(vegetation.vegetation_type)

    # Create placement relative to site
    origin = model.createIfcCartesianPoint([0.0, 0.0, 0.0])
    axis = model.createIfcDirection([0.0, 0.0, 1.0])
    ref_direction = model.createIfcDirection([1.0, 0.0, 0.0])
    axis2_placement = model.createIfcAxis2Placement3D(origin, axis, ref_direction)
    veg_placement = model.createIfcLocalPlacement(
        site.ObjectPlacement,
        axis2_placement
    )
    ifc_vegetation.ObjectPlacement = veg_placement

    # Assign vegetation to site
    ifcopenshell.api.run(
        "spatial.assign_container",
        model,
        products=[ifc_vegetation],
        relating_structure=site
    )

    # Create representations
    representations = []

    # Remove duplicate closing point
    if coords[0] == coords[-1]:
        coords = coords[:-1]

    if len(coords) < 3:
        logger.warning(f"Vegetation {vegetation.id}: too few unique coordinates")
        return None

    # Create extruded solid representation
    base_z = float(base_elevation - offset_z)
    top_z = base_z + float(height_used)

    # Create footprint polygon
    profile_points = [
        model.createIfcCartesianPoint([
            float(x - offset_x),
            float(y - offset_y)
        ])
        for x, y in coords
    ]
    # Close the profile
    profile_points.append(profile_points[0])

    polyline = model.createIfcPolyLine(profile_points)
    profile = model.createIfcArbitraryClosedProfileDef(
        "AREA",
        None,
        polyline
    )

    # Create extrusion direction
    direction = model.createIfcDirection([0.0, 0.0, 1.0])
    position = model.createIfcAxis2Placement3D(
        model.createIfcCartesianPoint([0.0, 0.0, base_z]),
        direction,
        None
    )

    # Create extruded area solid
    extruded_solid = model.createIfcExtrudedAreaSolid(
        profile,
        position,
        direction,
        height_used
    )

    # Create representation
    body_rep = model.createIfcShapeRepresentation(
        body_context, "Body", "SweptSolid", [extruded_solid]
    )
    representations.append(body_rep)

    # Assign representation
    if representations:
        ifc_vegetation.Representation = model.createIfcProductDefinitionShape(
            None, None, representations
        )
    else:
        logger.warning(f"Vegetation {vegetation.id}: no representations created")
        return None

    # Add properties
    _add_vegetation_properties(model, ifc_vegetation, vegetation, height_used, height_source)

    logger.debug(f"Created IfcGeographicElement ({ifc_vegetation.ObjectType}): {veg_name}")
    return ifc_vegetation


def _add_vegetation_properties(
    model: ifcopenshell.file,
    ifc_vegetation: ifcopenshell.entity_instance,
    vegetation: VegetationFeature,
    height: float,
    height_source: str
):
    """Add property sets to vegetation element"""
    pset = ifcopenshell.api.run(
        "pset.add_pset",
        model,
        product=ifc_vegetation,
        name="Pset_VegetationProperties"
    )

    properties = {
        "Height": height,
        "HeightSource": height_source
    }

    if vegetation.vegetation_type:
        properties["VegetationType"] = vegetation.vegetation_type

    if vegetation.canopy_area:
        properties["CanopyArea"] = vegetation.canopy_area

    if vegetation.tree_species:
        properties["TreeSpecies"] = vegetation.tree_species

    if vegetation.density:
        properties["Density"] = vegetation.density

    ifcopenshell.api.run(
        "pset.edit_pset",
        model,
        pset=pset,
        properties=properties
    )


def vegetation_to_ifc_batch(
    model: ifcopenshell.file,
    vegetation_list: List[VegetationFeature],
    site: ifcopenshell.entity_instance,
    body_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    base_elevation: float = 0.0,
    fetch_elevations_func=None
) -> List[ifcopenshell.entity_instance]:
    """
    Convert multiple vegetation features to IFC with terrain-projected elevations

    Args:
        model: IFC model
        vegetation_list: List of VegetationFeature objects
        site: Parent IfcSite element
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
        base_elevation: Default base elevation for vegetation
        fetch_elevations_func: Function to fetch elevations for coordinates (optional)

    Returns:
        List of created IfcGeographicElement entities
    """
    # Fetch terrain elevations for vegetation centroids
    veg_elevations = []
    if fetch_elevations_func and vegetation_list:
        centroids = []
        for veg in vegetation_list:
            if veg.geometry and not veg.geometry.is_empty:
                centroid = veg.geometry.centroid
                centroids.append((centroid.x, centroid.y))
            else:
                centroids.append(None)
        
        # Get elevations for valid centroids
        valid_centroids = [c for c in centroids if c is not None]
        if valid_centroids:
            logger.info(f"Fetching elevations for {len(valid_centroids)} vegetation centroids...")
            all_elevations = fetch_elevations_func(valid_centroids)
            
            # Map back to original list
            elev_idx = 0
            for c in centroids:
                if c is not None:
                    veg_elevations.append(all_elevations[elev_idx])
                    elev_idx += 1
                else:
                    veg_elevations.append(base_elevation)
        else:
            veg_elevations = [base_elevation] * len(vegetation_list)
    else:
        veg_elevations = [base_elevation] * len(vegetation_list)
    
    ifc_vegetation_list = []
    for vegetation, elev in zip(vegetation_list, veg_elevations):
        ifc_veg = vegetation_to_ifc(
            model, vegetation, site, body_context,
            offset_x, offset_y, offset_z, elev
        )
        if ifc_veg:
            ifc_vegetation_list.append(ifc_veg)

    logger.info(f"Created {len(ifc_vegetation_list)}/{len(vegetation_list)} vegetation elements")
    return ifc_vegetation_list


# ============================================================================
# Forest Tree Generation (from NFI raster data) - Using IFC Type Instancing
# ============================================================================

# Cache for tree type definitions (geometry created once, reused)
_tree_type_cache = {}


def _create_tree_style(
    model: ifcopenshell.file,
    is_deciduous: bool
) -> ifcopenshell.entity_instance:
    """Create a surface style for forest trees."""
    color = DECIDUOUS_COLOR_RGB if is_deciduous else CONIFEROUS_COLOR_RGB
    colour = model.createIfcColourRgb(None, *color)
    
    rendering = model.createIfcSurfaceStyleRendering(
        colour, 0.0, None, None, None, None, None, None, "FLAT"
    )
    
    return model.createIfcSurfaceStyle(
        "DeciduousTree" if is_deciduous else "ConiferousTree",
        "BOTH",
        [rendering]
    )


def _create_tree_type_geometry(
    model: ifcopenshell.file,
    body_context,
    height: float,
    is_deciduous: bool
) -> Optional[ifcopenshell.entity_instance]:
    """
    Create tree geometry at ORIGIN for use as a type definition.
    Instances will be placed using LocalPlacement.
    
    Args:
        model: IFC model
        body_context: IFC Body context
        height: Tree height in meters
        is_deciduous: True for deciduous (sphere canopy), False for coniferous (cone)
    
    Returns:
        IfcShapeRepresentation or None
    """
    import math
    
    # Calculate dimensions
    trunk_height = height * FOREST_TRUNK_HEIGHT_RATIO
    trunk_radius = FOREST_TRUNK_RADIUS
    canopy_height = height - trunk_height
    canopy_radius = height * FOREST_CANOPY_RADIUS_RATIO
    
    # Geometry at origin (0,0,0)
    faces = []
    segments = 6  # Low poly for performance
    
    # === TRUNK ===
    trunk_points_bottom = []
    trunk_points_top = []
    
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        dx = trunk_radius * math.cos(angle)
        dy = trunk_radius * math.sin(angle)
        trunk_points_bottom.append((dx, dy, 0.0))
        trunk_points_top.append((dx, dy, trunk_height))
    
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([
            trunk_points_bottom[i],
            trunk_points_bottom[j],
            trunk_points_top[j],
            trunk_points_top[i]
        ])
    
    # === CANOPY ===
    canopy_base_z = trunk_height
    
    if is_deciduous:
        canopy_center_z = canopy_base_z + canopy_height * 0.5
        top = (0, 0, canopy_base_z + canopy_height)
        bottom = (0, 0, canopy_base_z)
        middle_points = []
        
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            middle_points.append((
                canopy_radius * math.cos(angle),
                canopy_radius * math.sin(angle),
                canopy_center_z
            ))
        
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([middle_points[i], middle_points[j], top])
            faces.append([middle_points[j], middle_points[i], bottom])
    else:
        apex = (0, 0, canopy_base_z + canopy_height)
        base_points = []
        
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            base_points.append((
                canopy_radius * math.cos(angle),
                canopy_radius * math.sin(angle),
                canopy_base_z
            ))
        
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([base_points[i], base_points[j], apex])
        
        center = (0, 0, canopy_base_z)
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([base_points[j], base_points[i], center])
    
    # Create IFC faces
    ifc_faces = []
    for face_points in faces:
        try:
            ifc_points = [
                model.createIfcCartesianPoint([float(p[0]), float(p[1]), float(p[2])])
                for p in face_points
            ]
            polyloop = model.createIfcPolyLoop(ifc_points)
            face_bound = model.createIfcFaceOuterBound(polyloop, True)
            ifc_face = model.createIfcFace([face_bound])
            ifc_faces.append(ifc_face)
        except Exception:
            continue
    
    if not ifc_faces:
        return None
    
    try:
        shell = model.createIfcClosedShell(ifc_faces)
        brep = model.createIfcFacetedBrep(shell)
        
        # Apply style
        style = _create_tree_style(model, is_deciduous)
        model.createIfcStyledItem(brep, [style], None)
        
        return model.createIfcShapeRepresentation(
            body_context, "Body", "Brep", [brep]
        )
    except Exception as e:
        logger.debug(f"Failed to create tree brep: {e}")
        return None


def _get_or_create_tree_type(
    model: ifcopenshell.file,
    body_context,
    is_deciduous: bool,
    height_class: int  # 0=small, 1=medium, 2=tall
) -> Optional[ifcopenshell.entity_instance]:
    """
    Get or create a tree type definition (geometry created once, reused).
    Uses 3 height classes per tree type for variety without explosion.
    """
    global _tree_type_cache
    
    # Create cache key
    model_id = id(model)
    type_key = (model_id, is_deciduous, height_class)
    
    if type_key in _tree_type_cache:
        return _tree_type_cache[type_key]
    
    # Define height for this class
    heights = [
        FOREST_TREE_HEIGHT_MIN,
        (FOREST_TREE_HEIGHT_MIN + FOREST_TREE_HEIGHT_MAX) / 2,
        FOREST_TREE_HEIGHT_MAX
    ]
    height = heights[height_class]
    
    # Create type geometry
    type_rep = _create_tree_type_geometry(model, body_context, height, is_deciduous)
    if type_rep is None:
        return None
    
    # Create type definition
    tree_type_name = f"{'Deciduous' if is_deciduous else 'Coniferous'}Tree_H{height_class}"
    
    tree_type = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class="IfcGeographicElementType",
        name=tree_type_name
    )
    tree_type.PredefinedType = "USERDEFINED"
    tree_type.ElementType = "TREE"
    
    # Assign representation to type
    tree_type.RepresentationMaps = [
        model.createIfcRepresentationMap(
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint([0.0, 0.0, 0.0]),
                None, None
            ),
            type_rep
        )
    ]
    
    _tree_type_cache[type_key] = tree_type
    logger.debug(f"Created tree type: {tree_type_name}")
    return tree_type


def forest_to_ifc(
    model: ifcopenshell.file,
    forest_points: List,  # List of TreeFeature
    site: ifcopenshell.entity_instance,
    body_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0
) -> List[ifcopenshell.entity_instance]:
    """
    Convert tree/hedge features to IFC elements using type instancing.
    Creates tree type definitions once and reuses geometry across instances.
    
    Args:
        model: IFC model
        forest_points: List of TreeFeature objects
        site: Parent IfcSite element
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
    
    Returns:
        List of created IfcGeographicElement entities
    """
    if not forest_points:
        return []
    
    print(f"    Creating tree types (6 variants: 2 species x 3 heights)...")
    
    # Pre-create all tree types
    for is_deciduous in [True, False]:
        for height_class in [0, 1, 2]:
            _get_or_create_tree_type(model, body_context, is_deciduous, height_class)
    
    print(f"    Placing {len(forest_points)} tree/hedge instances...")
    
    # Wrapper class to convert TreeFeature to format expected by hedge_to_ifc
    # Moved outside loop to avoid repeated class creation overhead
    class HedgeWrapper:
        def __init__(self, tree_feature):
            self.id = tree_feature.id
            # Buffer the LineString to create a polygon
            from shapely.geometry import Polygon
            coords = list(tree_feature.geometry.coords)
            if len(coords) >= 3:
                # Create a simple buffered polygon
                buffered = tree_feature.geometry.buffer(0.5)  # 0.5m width hedge
                if buffered.geom_type == 'Polygon':
                    self.geometry = buffered
                else:
                    # Fallback to simple polygon from coords
                    self.geometry = Polygon(coords)
            else:
                self.geometry = Polygon([])
            self.height = None
            self.vegetation_type = "hedge"
            # Add missing attributes expected by _add_vegetation_properties
            self.canopy_area = None
            self.tree_species = None
            self.density = None
            self.is_coniferous = False
    
    ifc_trees = []
    deciduous_count = 0
    coniferous_count = 0
    
    for i, tree_feature in enumerate(forest_points):
        # Handle hedges differently
        if tree_feature.feature_type == "hedge":
            hedge = hedge_to_ifc(
                model, HedgeWrapper(tree_feature), site, body_context,
                offset_x, offset_y, offset_z, base_elevation=tree_feature.z
            )
            if hedge:
                ifc_trees.append(hedge)
            continue
        
        # Handle tree rows - place multiple trees along the line
        if tree_feature.feature_type == "tree_row":
            # Place trees along the line at regular intervals
            tree_spacing = 10.0  # meters between trees
            line = tree_feature.geometry
            
            # Calculate number of trees based on length
            num_trees = max(1, int(tree_feature.length / tree_spacing))
            
            # Interpolate points along the line
            for j in range(num_trees):
                distance_along_line = (j / max(1, num_trees - 1)) * line.length if num_trees > 1 else line.length / 2
                point_on_line = line.interpolate(distance_along_line)
                
                # Determine height class and tree type for this tree
                height_class = int((point_on_line.x + point_on_line.y) * 100) % 3
                is_deciduous = (hash(f"{tree_feature.id}_{j}") % 2 == 0)
                
                # Get tree type
                tree_type = _get_or_create_tree_type(
                    model, body_context, is_deciduous, height_class
                )
                if tree_type is None:
                    continue
                
                # Create instance
                tree_name = f"Tree_{i}_{j}"
                ifc_tree = ifcopenshell.api.run(
                    "root.create_entity",
                    model,
                    ifc_class="IfcGeographicElement",
                    name=tree_name
                )
                ifc_tree.PredefinedType = "USERDEFINED"
                ifc_tree.ObjectType = "TREE"
                
                # Link to type (this reuses geometry!)
                ifcopenshell.api.run(
                    "type.assign_type",
                    model,
                    related_objects=[ifc_tree],
                    relating_type=tree_type
                )
                
                # Get elevation for this point
                tree_z = tree_feature.z  # Use feature elevation (or could interpolate)
                
                # Create placement at tree position
                local_x = point_on_line.x - offset_x
                local_y = point_on_line.y - offset_y
                local_z = tree_z - offset_z
                
                origin = model.createIfcCartesianPoint([local_x, local_y, local_z])
                axis2_placement = model.createIfcAxis2Placement3D(origin, None, None)
                tree_placement = model.createIfcLocalPlacement(
                    site.ObjectPlacement,
                    axis2_placement
                )
                ifc_tree.ObjectPlacement = tree_placement
                
                # Assign to site
                ifcopenshell.api.run(
                    "spatial.assign_container",
                    model,
                    products=[ifc_tree],
                    relating_structure=site
                )
                
                ifc_trees.append(ifc_tree)
                if is_deciduous:
                    deciduous_count += 1
                else:
                    coniferous_count += 1
            continue
        
        # For single trees (shouldn't happen with current loader, but handle it)
        # Determine height class based on position (deterministic variety)
        height_class = int((tree_feature.x + tree_feature.y) * 100) % 3
        
        # Get tree type (deciduous vs coniferous)
        is_deciduous = (tree_feature.tree_type == "deciduous")
        
        # Get tree type
        tree_type = _get_or_create_tree_type(
            model, body_context, is_deciduous, height_class
        )
        if tree_type is None:
            continue
        
        # Create instance
        tree_name = f"Tree_{i}"
        ifc_tree = ifcopenshell.api.run(
            "root.create_entity",
            model,
            ifc_class="IfcGeographicElement",
            name=tree_name
        )
        ifc_tree.PredefinedType = "USERDEFINED"
        ifc_tree.ObjectType = "TREE"
        
        # Link to type (this reuses geometry!)
        ifcopenshell.api.run(
            "type.assign_type",
            model,
            related_objects=[ifc_tree],
            relating_type=tree_type
        )
        
        # Create placement at tree position
        local_x = tree_feature.x - offset_x
        local_y = tree_feature.y - offset_y
        local_z = tree_feature.z - offset_z
        
        origin = model.createIfcCartesianPoint([local_x, local_y, local_z])
        axis2_placement = model.createIfcAxis2Placement3D(origin, None, None)
        tree_placement = model.createIfcLocalPlacement(
            site.ObjectPlacement,
            axis2_placement
        )
        ifc_tree.ObjectPlacement = tree_placement
        
        # Assign to site
        ifcopenshell.api.run(
            "spatial.assign_container",
            model,
            products=[ifc_tree],
            relating_structure=site
        )
        
        ifc_trees.append(ifc_tree)
        if is_deciduous:
            deciduous_count += 1
        else:
            coniferous_count += 1
    
    logger.info(f"Created {len(ifc_trees)} trees using 6 type definitions ({deciduous_count} deciduous, {coniferous_count} coniferous)")
    return ifc_trees


# ============================================================================
# Railway Conversion (with instanced sleeper/rail geometry)
# ============================================================================

# Railway geometry constants
RAIL_HEAD_WIDTH = 0.065  # meters - UIC60 rail head width
RAIL_HEIGHT = 0.172  # meters - UIC60 rail height
STANDARD_GAUGE = 1.435  # meters - standard track gauge
NARROW_GAUGE = 1.000  # meters - narrow gauge
SLEEPER_LENGTH = 2.6  # meters
SLEEPER_WIDTH = 0.26  # meters
SLEEPER_HEIGHT = 0.16  # meters
SLEEPER_SPACING = 0.6  # meters - spacing between sleepers
BALLAST_WIDTH = 3.0  # meters - ballast bed width
BALLAST_DEPTH = 0.3  # meters - ballast depth

# Railway colors
RAIL_COLOR_RGB = (0.3, 0.3, 0.3)  # Dark gray
SLEEPER_CONCRETE_COLOR_RGB = (0.5, 0.5, 0.5)  # Gray concrete
SLEEPER_WOOD_COLOR_RGB = (0.4, 0.3, 0.2)  # Brown wood
BALLAST_COLOR_RGB = (0.5, 0.5, 0.5)  # Light gray gravel

# Cache for railway type definitions
_railway_type_cache = {}


def _create_railway_style(
    model: ifcopenshell.file,
    color: Tuple[float, float, float]
) -> ifcopenshell.entity_instance:
    """Create a surface style for railway components."""
    colour = model.createIfcColourRgb(None, *color)
    rendering = model.createIfcSurfaceStyleRendering(
        colour, 0.0, None, None, None, None, None, None, "FLAT"
    )
    return model.createIfcSurfaceStyle(
        f"Railway_{color[0]:.1f}_{color[1]:.1f}_{color[2]:.1f}",
        "BOTH",
        [rendering]
    )


def _create_railway_type_geometry(
    model: ifcopenshell.file,
    body_context,
    gauge: float,
    sleeper_material: str = "concrete"  # "concrete" or "wood"
) -> Optional[ifcopenshell.entity_instance]:
    """
    Create railway cross-section geometry at ORIGIN for use as a type definition.
    Creates: 2 rails + 1 sleeper + ballast bed.
    
    Args:
        model: IFC model
        body_context: IFC Body context
        gauge: Track gauge in meters (1.435 for standard, 1.0 for narrow)
        sleeper_material: "concrete" or "wood"
    
    Returns:
        IfcShapeRepresentation or None
    """
    import math
    
    half_gauge = gauge / 2.0
    half_sleeper_length = SLEEPER_LENGTH / 2.0
    half_sleeper_width = SLEEPER_WIDTH / 2.0
    
    faces = []
    segments = 8  # Low poly for performance
    
    # === LEFT RAIL ===
    rail_left_x = -half_gauge
    rail_points_bottom = []
    rail_points_top = []
    
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        # Rail head (circular cross-section)
        dx = (RAIL_HEAD_WIDTH / 2) * math.cos(angle)
        dy = (RAIL_HEAD_WIDTH / 2) * math.sin(angle)
        rail_points_bottom.append((rail_left_x + dx, dy, 0.0))
        rail_points_top.append((rail_left_x + dx, dy, RAIL_HEIGHT))
    
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([
            rail_points_bottom[i],
            rail_points_bottom[j],
            rail_points_top[j],
            rail_points_top[i]
        ])
    
    # === RIGHT RAIL ===
    rail_right_x = half_gauge
    rail_points_bottom = []
    rail_points_top = []
    
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        dx = (RAIL_HEAD_WIDTH / 2) * math.cos(angle)
        dy = (RAIL_HEAD_WIDTH / 2) * math.sin(angle)
        rail_points_bottom.append((rail_right_x + dx, dy, 0.0))
        rail_points_top.append((rail_right_x + dx, dy, RAIL_HEIGHT))
    
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([
            rail_points_bottom[i],
            rail_points_bottom[j],
            rail_points_top[j],
            rail_points_top[i]
        ])
    
    # === SLEEPER ===
    sleeper_z = -SLEEPER_HEIGHT
    sleeper_points = [
        (-half_sleeper_length, -half_sleeper_width, sleeper_z),
        (half_sleeper_length, -half_sleeper_width, sleeper_z),
        (half_sleeper_length, half_sleeper_width, sleeper_z),
        (-half_sleeper_length, half_sleeper_width, sleeper_z),
    ]
    sleeper_top = [p[:2] + (0.0,) for p in sleeper_points]
    
    # Sleeper top face
    faces.append(sleeper_points)
    # Sleeper bottom face
    faces.append([(p[0], p[1], p[2] - SLEEPER_HEIGHT) for p in reversed(sleeper_points)])
    # Sleeper sides
    for i in range(4):
        j = (i + 1) % 4
        faces.append([
            sleeper_points[i],
            sleeper_points[j],
            (sleeper_points[j][0], sleeper_points[j][1], sleeper_points[j][2] - SLEEPER_HEIGHT),
            (sleeper_points[i][0], sleeper_points[i][1], sleeper_points[i][2] - SLEEPER_HEIGHT)
        ])
    
    # === BALLAST BED ===
    half_ballast_width = BALLAST_WIDTH / 2.0
    ballast_z = sleeper_z - SLEEPER_HEIGHT
    ballast_points = [
        (-half_ballast_width, -half_ballast_width, ballast_z),
        (half_ballast_width, -half_ballast_width, ballast_z),
        (half_ballast_width, half_ballast_width, ballast_z),
        (-half_ballast_width, half_ballast_width, ballast_z),
    ]
    
    # Ballast top face
    faces.append(ballast_points)
    # Ballast bottom face
    faces.append([(p[0], p[1], p[2] - BALLAST_DEPTH) for p in reversed(ballast_points)])
    # Ballast sides
    for i in range(4):
        j = (i + 1) % 4
        faces.append([
            ballast_points[i],
            ballast_points[j],
            (ballast_points[j][0], ballast_points[j][1], ballast_points[j][2] - BALLAST_DEPTH),
            (ballast_points[i][0], ballast_points[i][1], ballast_points[i][2] - BALLAST_DEPTH)
        ])
    
    # Create IFC faces
    ifc_faces = []
    for face_points in faces:
        try:
            ifc_points = [
                model.createIfcCartesianPoint([float(p[0]), float(p[1]), float(p[2])])
                for p in face_points
            ]
            polyloop = model.createIfcPolyLoop(ifc_points)
            face_bound = model.createIfcFaceOuterBound(polyloop, True)
            ifc_face = model.createIfcFace([face_bound])
            ifc_faces.append(ifc_face)
        except Exception:
            continue
    
    if not ifc_faces:
        return None
    
    try:
        shell = model.createIfcClosedShell(ifc_faces)
        brep = model.createIfcFacetedBrep(shell)
        
        # Apply styles
        rail_style = _create_railway_style(model, RAIL_COLOR_RGB)
        sleeper_color = SLEEPER_CONCRETE_COLOR_RGB if sleeper_material == "concrete" else SLEEPER_WOOD_COLOR_RGB
        sleeper_style = _create_railway_style(model, sleeper_color)
        ballast_style = _create_railway_style(model, BALLAST_COLOR_RGB)
        
        # Style rails (first 2*segments faces)
        for i in range(2 * segments):
            if i < len(ifc_faces):
                model.createIfcStyledItem(ifc_faces[i], [rail_style], None)
        
        # Style sleeper (next 6 faces)
        for i in range(2 * segments, 2 * segments + 6):
            if i < len(ifc_faces):
                model.createIfcStyledItem(ifc_faces[i], [sleeper_style], None)
        
        # Style ballast (remaining faces)
        for i in range(2 * segments + 6, len(ifc_faces)):
            model.createIfcStyledItem(ifc_faces[i], [ballast_style], None)
        
        return model.createIfcShapeRepresentation(
            body_context, "Body", "Brep", [brep]
        )
    except Exception as e:
        logger.debug(f"Failed to create railway brep: {e}")
        return None


def _get_or_create_railway_type(
    model: ifcopenshell.file,
    body_context,
    gauge: float,
    sleeper_material: str = "concrete"
) -> Optional[ifcopenshell.entity_instance]:
    """
    Get or create a railway type definition (geometry created once, reused).
    """
    global _railway_type_cache
    
    model_id = id(model)
    type_key = (model_id, gauge, sleeper_material)
    
    if type_key in _railway_type_cache:
        return _railway_type_cache[type_key]
    
    # Create type geometry
    type_rep = _create_railway_type_geometry(model, body_context, gauge, sleeper_material)
    if type_rep is None:
        return None
    
    # Create type definition
    gauge_name = "Standard" if abs(gauge - STANDARD_GAUGE) < 0.1 else "Narrow"
    railway_type_name = f"RailwayType_{gauge_name}_{sleeper_material.capitalize()}"
    
    railway_type = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class="IfcGeographicElementType",
        name=railway_type_name
    )
    railway_type.PredefinedType = "USERDEFINED"
    railway_type.ElementType = "RAILWAY"
    
    # Assign representation to type
    railway_type.RepresentationMaps = [
        model.createIfcRepresentationMap(
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint([0.0, 0.0, 0.0]),
                None, None
            ),
            type_rep
        )
    ]
    
    _railway_type_cache[type_key] = railway_type
    logger.debug(f"Created railway type: {railway_type_name}")
    return railway_type


def railway_to_ifc(
    model: ifcopenshell.file,
    railway,
    site: ifcopenshell.entity_instance,
    body_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    elevations: Optional[List[float]] = None
) -> Optional[ifcopenshell.entity_instance]:
    """
    Convert a RailwayFeature to an IfcGeographicElement with instanced railway geometry.
    
    Args:
        model: IFC model
        railway: RailwayFeature to convert
        site: Parent IfcSite element
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
        elevations: List of elevations for each railway coordinate
    
    Returns:
        IfcGeographicElement for railway or None if failed
    """
    from src.loaders.railway import RailwayFeature
    
    if railway.geometry is None or railway.geometry.is_empty:
        logger.warning(f"Skipping railway {railway.id}: no geometry")
        return None
    
    try:
        coords = list(railway.geometry.coords)
        if len(coords) < 2:
            logger.warning(f"Skipping railway {railway.id}: too few coordinates")
            return None
    except Exception as e:
        logger.warning(f"Skipping railway {railway.id}: invalid geometry - {e}")
        return None
    
    # Determine gauge
    gauge_str = railway.gauge
    if gauge_str:
        try:
            gauge = float(gauge_str) / 1000.0  # Convert mm to meters
        except (ValueError, TypeError):
            gauge = STANDARD_GAUGE
    else:
        # Default based on railway type
        if railway.railway_type == "narrow_gauge":
            gauge = NARROW_GAUGE
        else:
            gauge = STANDARD_GAUGE
    
    # Determine sleeper material (concrete for main lines, wood for narrow gauge)
    sleeper_material = "wood" if gauge < 1.2 else "concrete"
    
    # Get railway type
    railway_type = _get_or_create_railway_type(model, body_context, gauge, sleeper_material)
    if railway_type is None:
        logger.warning(f"Failed to create railway type for {railway.id}")
        return None
    
    # Create IfcGeographicElement
    railway_name = railway.name or f"Railway_{railway.id}"
    ifc_railway = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class="IfcGeographicElement",
        name=railway_name
    )
    ifc_railway.PredefinedType = "USERDEFINED"
    ifc_railway.ObjectType = "RAILWAY"
    
    if railway.railway_type:
        ifc_railway.Description = railway.railway_type.replace("_", " ").title()
    
    # Create placement - use first coordinate
    first_x, first_y = coords[0]
    z_adjustment = 0.0
    if elevations and len(elevations) > 0:
        z_adjustment = elevations[0]
    
    local_x = float(first_x - offset_x)
    local_y = float(first_y - offset_y)
    local_z = float(z_adjustment - offset_z)
    
    origin = model.createIfcCartesianPoint([local_x, local_y, local_z])
    
    # Calculate direction along track
    if len(coords) >= 2:
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
        length = (dx**2 + dy**2)**0.5
        if length > 0:
            axis_x = dx / length
            axis_y = dy / length
            axis = model.createIfcDirection([axis_x, axis_y, 0.0])
        else:
            axis = model.createIfcDirection([1.0, 0.0, 0.0])
    else:
        axis = model.createIfcDirection([1.0, 0.0, 0.0])
    
    ref_direction = model.createIfcDirection([0.0, 0.0, 1.0])
    axis2_placement = model.createIfcAxis2Placement3D(origin, ref_direction, axis)
    railway_placement = model.createIfcLocalPlacement(
        site.ObjectPlacement,
        axis2_placement
    )
    ifc_railway.ObjectPlacement = railway_placement
    
    # Link to type (this reuses geometry!)
    ifcopenshell.api.run(
        "type.assign_type",
        model,
        related_objects=[ifc_railway],
        relating_type=railway_type
    )
    
    # Assign to site
    ifcopenshell.api.run(
        "spatial.assign_container",
        model,
        products=[ifc_railway],
        relating_structure=site
    )
    
    # Create swept path representation along the railway line
    # This creates the actual track geometry following the path
    representations = []
    
    # Create polyline representation for the track path
    if elevations and len(elevations) == len(coords):
        track_points = [
            model.createIfcCartesianPoint([
                float(x - offset_x),
                float(y - offset_y),
                float(z - offset_z)
            ])
            for (x, y), z in zip(coords, elevations)
        ]
    else:
        default_z = z_adjustment - offset_z
        track_points = [
            model.createIfcCartesianPoint([
                float(x - offset_x),
                float(y - offset_y),
                default_z
            ])
            for x, y in coords
        ]
    
    polyline = model.createIfcPolyLine(track_points)
    path_rep = model.createIfcShapeRepresentation(
        body_context, "Body", "Curve3D", [polyline]
    )
    representations.append(path_rep)
    
    ifc_railway.Representation = model.createIfcProductDefinitionShape(
        None, None, representations
    )
    
    # Add properties
    _add_railway_properties(model, ifc_railway, railway, gauge)
    
    logger.debug(f"Created IfcGeographicElement (RAILWAY): {railway_name}")
    return ifc_railway


def _add_railway_properties(
    model: ifcopenshell.file,
    ifc_railway: ifcopenshell.entity_instance,
    railway,
    gauge: float
):
    """Add property sets to railway element"""
    pset = ifcopenshell.api.run(
        "pset.add_pset",
        model,
        product=ifc_railway,
        name="Pset_RailwayProperties"
    )
    
    properties = {
        "Gauge": gauge
    }
    
    if railway.railway_type:
        properties["RailwayType"] = railway.railway_type
    
    if railway.name:
        properties["Name"] = railway.name
    
    if railway.electrified:
        properties["Electrified"] = railway.electrified
    
    if railway.tracks:
        properties["Tracks"] = railway.tracks
    
    if railway.service:
        properties["Service"] = railway.service
    
    ifcopenshell.api.run(
        "pset.edit_pset",
        model,
        pset=pset,
        properties=properties
    )


def railways_to_ifc(
    model: ifcopenshell.file,
    railways: List,
    site: ifcopenshell.entity_instance,
    body_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    fetch_elevations_func=None
) -> List[ifcopenshell.entity_instance]:
    """
    Convert multiple railways to IFC with terrain-projected elevations.
    
    Args:
        model: IFC model
        railways: List of RailwayFeature objects
        site: Parent IfcSite element
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
        fetch_elevations_func: Function to fetch elevations for coordinates
    
    Returns:
        List of created IfcGeographicElement entities
    """
    if not railways:
        return []
    
    # Pre-create railway types
    print(f"    Creating railway types...")
    gauges_used = set()
    for railway in railways:
        gauge_str = railway.gauge
        if gauge_str:
            try:
                gauge = float(gauge_str) / 1000.0
            except (ValueError, TypeError):
                gauge = STANDARD_GAUGE
        else:
            gauge = STANDARD_GAUGE if railway.railway_type != "narrow_gauge" else NARROW_GAUGE
        
        sleeper_material = "wood" if gauge < 1.2 else "concrete"
        gauges_used.add((gauge, sleeper_material))
    
    for gauge, sleeper_material in gauges_used:
        _get_or_create_railway_type(model, body_context, gauge, sleeper_material)
    
    ifc_railways = []
    
    # Collect coordinates for batch elevation fetching
    if fetch_elevations_func:
        all_coords = []
        coord_counts = []
        for railway in railways:
            if railway.geometry and not railway.geometry.is_empty:
                coords = list(railway.geometry.coords)
                all_coords.extend([(x, y) for x, y in coords])
                coord_counts.append(len(coords))
            else:
                coord_counts.append(0)
        
        # Fetch elevations
        if all_coords:
            logger.info(f"Fetching elevations for {len(all_coords)} railway points...")
            all_elevations = fetch_elevations_func(all_coords)
            
            # Split elevations back to individual railways
            idx = 0
            railway_elevations = []
            for count in coord_counts:
                if count > 0:
                    railway_elevations.append(all_elevations[idx:idx + count])
                    idx += count
                else:
                    railway_elevations.append(None)
        else:
            railway_elevations = [None] * len(railways)
    else:
        railway_elevations = [None] * len(railways)
    
    # Create IFC elements
    for railway, elevations in zip(railways, railway_elevations):
        ifc_railway = railway_to_ifc(
            model, railway, site, body_context,
            offset_x, offset_y, offset_z, elevations
        )
        if ifc_railway:
            ifc_railways.append(ifc_railway)
    
    logger.info(f"Created {len(ifc_railways)}/{len(railways)} railway elements")
    return ifc_railways


# ============================================================================
# Bridge Conversion
# ============================================================================

# Bridge geometry constants
DEFAULT_BRIDGE_DECK_THICKNESS = 0.2  # meters - highway bridge deck
RAILWAY_BRIDGE_DECK_THICKNESS = 0.3  # meters - railway bridge deck
BRIDGE_BARRIER_HEIGHT = 1.0  # meters - barrier height
BRIDGE_BARRIER_WIDTH = 0.2  # meters - barrier width
FOOTWAY_BRIDGE_WIDTH = 2.0  # meters - default footway width

# Bridge colors
BRIDGE_CONCRETE_COLOR_RGB = (0.7, 0.7, 0.7)  # Light gray concrete
BRIDGE_STEEL_COLOR_RGB = (0.4, 0.4, 0.45)  # Dark gray steel
BRIDGE_BARRIER_COLOR_RGB = (0.8, 0.8, 0.8)  # Light gray barriers


def _create_bridge_style(
    model: ifcopenshell.file,
    color: Tuple[float, float, float]
) -> ifcopenshell.entity_instance:
    """Create a surface style for bridge components."""
    colour = model.createIfcColourRgb(None, *color)
    rendering = model.createIfcSurfaceStyleRendering(
        colour, 0.0, None, None, None, None, None, None, "FLAT"
    )
    return model.createIfcSurfaceStyle(
        f"Bridge_{color[0]:.1f}_{color[1]:.1f}_{color[2]:.1f}",
        "BOTH",
        [rendering]
    )


def bridge_to_ifc(
    model: ifcopenshell.file,
    bridge,
    site: ifcopenshell.entity_instance,
    body_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    elevations: Optional[List[float]] = None
) -> Optional[ifcopenshell.entity_instance]:
    """
    Convert a BridgeFeature to an IfcGeographicElement with bridge geometry.
    
    Args:
        model: IFC model
        bridge: BridgeFeature to convert
        site: Parent IfcSite element
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
        elevations: List of elevations for each bridge coordinate
    
    Returns:
        IfcGeographicElement for bridge or None if failed
    """
    from src.loaders.bridge import BridgeFeature
    
    if bridge.geometry is None or bridge.geometry.is_empty:
        logger.warning(f"Skipping bridge {bridge.id}: no geometry")
        return None
    
    try:
        coords = list(bridge.geometry.coords)
        if len(coords) < 2:
            logger.warning(f"Skipping bridge {bridge.id}: too few coordinates")
            return None
    except Exception as e:
        logger.warning(f"Skipping bridge {bridge.id}: invalid geometry - {e}")
        return None
    
    # Determine bridge width
    width = bridge.width if bridge.width and bridge.width > 0 else FOOTWAY_BRIDGE_WIDTH
    if bridge.carries == "highway":
        width = bridge.width if bridge.width and bridge.width > 0 else 8.0
    elif bridge.carries == "railway":
        width = bridge.width if bridge.width and bridge.width > 0 else 5.0
    
    # Determine deck thickness
    if bridge.carries == "railway":
        deck_thickness = RAILWAY_BRIDGE_DECK_THICKNESS
    else:
        deck_thickness = DEFAULT_BRIDGE_DECK_THICKNESS
    
    # Determine material and color
    material = bridge.material or "concrete"
    if "steel" in material.lower():
        bridge_color = BRIDGE_STEEL_COLOR_RGB
    else:
        bridge_color = BRIDGE_CONCRETE_COLOR_RGB
    
    # Create IfcGeographicElement
    bridge_name = bridge.name or f"Bridge_{bridge.id}"
    ifc_bridge = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class="IfcGeographicElement",
        name=bridge_name
    )
    ifc_bridge.PredefinedType = "USERDEFINED"
    ifc_bridge.ObjectType = "BRIDGE"
    
    if bridge.bridge_type:
        ifc_bridge.Description = bridge.bridge_type.replace("_", " ").title()
    
    # Create placement relative to site
    origin = model.createIfcCartesianPoint([0.0, 0.0, 0.0])
    axis = model.createIfcDirection([0.0, 0.0, 1.0])
    ref_direction = model.createIfcDirection([1.0, 0.0, 0.0])
    axis2_placement = model.createIfcAxis2Placement3D(origin, axis, ref_direction)
    bridge_placement = model.createIfcLocalPlacement(
        site.ObjectPlacement,
        axis2_placement
    )
    ifc_bridge.ObjectPlacement = bridge_placement
    
    # Assign bridge to site
    ifcopenshell.api.run(
        "spatial.assign_container",
        model,
        products=[ifc_bridge],
        relating_structure=site
    )
    
    # Create bridge deck representation
    representations = []
    half_width = width / 2.0
    
    # Create 3D coordinates
    if elevations and len(elevations) == len(coords):
        coords_3d = [(x, y, z) for (x, y), z in zip(coords, elevations)]
    else:
        # Use default elevation
        default_z = offset_z
        coords_3d = [(x, y, default_z) for x, y in coords]
    
    # Create bridge deck surface
    all_faces = []
    
    for i in range(len(coords_3d) - 1):
        x1, y1, z1 = coords_3d[i]
        x2, y2, z2 = coords_3d[i + 1]
        
        # Calculate perpendicular direction
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        if length == 0:
            continue
        
        perp_x = -dy / length
        perp_y = dx / length
        
        # Deck surface corners
        d1 = (x1 + perp_x * half_width - offset_x, y1 + perp_y * half_width - offset_y, z1 - offset_z)
        d2 = (x2 + perp_x * half_width - offset_x, y2 + perp_y * half_width - offset_y, z2 - offset_z)
        d3 = (x2 - perp_x * half_width - offset_x, y2 - perp_y * half_width - offset_y, z2 - offset_z)
        d4 = (x1 - perp_x * half_width - offset_x, y1 - perp_y * half_width - offset_y, z1 - offset_z)
        
        # Bottom corners
        b1 = (d1[0], d1[1], d1[2] - deck_thickness)
        b2 = (d2[0], d2[1], d2[2] - deck_thickness)
        b3 = (d3[0], d3[1], d3[2] - deck_thickness)
        b4 = (d4[0], d4[1], d4[2] - deck_thickness)
        
        def make_face(pts):
            points = [model.createIfcCartesianPoint([float(p[0]), float(p[1]), float(p[2])]) for p in pts]
            poly_loop = model.createIfcPolyLoop(points)
            return model.createIfcFace([model.createIfcFaceOuterBound(poly_loop, True)])
        
        # Top face (deck surface)
        all_faces.append(make_face([d1, d2, d3, d4]))
        # Bottom face
        all_faces.append(make_face([b4, b3, b2, b1]))
        # Side faces
        all_faces.append(make_face([d1, d4, b4, b1]))
        all_faces.append(make_face([d2, d1, b1, b2]))
        all_faces.append(make_face([d3, d2, b2, b3]))
        all_faces.append(make_face([d4, d3, b3, b4]))
    
    if all_faces:
        try:
            shell = model.createIfcClosedShell(all_faces)
            brep = model.createIfcFacetedBrep(shell)
            
            # Apply bridge style
            bridge_style = _create_bridge_style(model, bridge_color)
            model.createIfcStyledItem(brep, [bridge_style], None)
            
            deck_rep = model.createIfcShapeRepresentation(
                body_context, "Body", "Brep", [brep]
            )
            representations.append(deck_rep)
        except Exception as e:
            logger.warning(f"Could not create bridge deck solid: {e}")
    
    # Add barriers for highway/footway bridges
    if bridge.carries in ("highway", "footway") and all_faces:
        barrier_faces = []
        barrier_style = _create_bridge_style(model, BRIDGE_BARRIER_COLOR_RGB)
        
        # Create barriers along edges
        for i in range(len(coords_3d) - 1):
            x1, y1, z1 = coords_3d[i]
            x2, y2, z2 = coords_3d[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            length = (dx**2 + dy**2)**0.5
            if length == 0:
                continue
            
            perp_x = -dy / length
            perp_y = dx / length
            
            # Left barrier
            barrier_top_1 = (x1 + perp_x * half_width - offset_x, y1 + perp_y * half_width - offset_y, z1 - offset_z + BRIDGE_BARRIER_HEIGHT)
            barrier_top_2 = (x2 + perp_x * half_width - offset_x, y2 + perp_y * half_width - offset_y, z2 - offset_z + BRIDGE_BARRIER_HEIGHT)
            barrier_bottom_1 = (x1 + perp_x * half_width - offset_x, y1 + perp_y * half_width - offset_y, z1 - offset_z)
            barrier_bottom_2 = (x2 + perp_x * half_width - offset_x, y2 + perp_y * half_width - offset_y, z2 - offset_z)
            
            barrier_side_1 = (barrier_top_1[0] + perp_x * BRIDGE_BARRIER_WIDTH, barrier_top_1[1] + perp_y * BRIDGE_BARRIER_WIDTH, barrier_top_1[2])
            barrier_side_2 = (barrier_top_2[0] + perp_x * BRIDGE_BARRIER_WIDTH, barrier_top_2[1] + perp_y * BRIDGE_BARRIER_WIDTH, barrier_top_2[2])
            barrier_side_3 = (barrier_bottom_2[0] + perp_x * BRIDGE_BARRIER_WIDTH, barrier_bottom_2[1] + perp_y * BRIDGE_BARRIER_WIDTH, barrier_bottom_2[2])
            barrier_side_4 = (barrier_bottom_1[0] + perp_x * BRIDGE_BARRIER_WIDTH, barrier_bottom_1[1] + perp_y * BRIDGE_BARRIER_WIDTH, barrier_bottom_1[2])
            
            def make_barrier_face(pts):
                points = [model.createIfcCartesianPoint([float(p[0]), float(p[1]), float(p[2])]) for p in pts]
                poly_loop = model.createIfcPolyLoop(points)
                return model.createIfcFace([model.createIfcFaceOuterBound(poly_loop, True)])
            
            # Barrier faces
            barrier_faces.append(make_barrier_face([barrier_top_1, barrier_top_2, barrier_side_2, barrier_side_1]))
            barrier_faces.append(make_barrier_face([barrier_bottom_1, barrier_bottom_2, barrier_side_3, barrier_side_4]))
            barrier_faces.append(make_barrier_face([barrier_top_1, barrier_bottom_1, barrier_side_4, barrier_side_1]))
            barrier_faces.append(make_barrier_face([barrier_top_2, barrier_bottom_2, barrier_side_3, barrier_side_2]))
        
        if barrier_faces:
            try:
                barrier_shell = model.createIfcOpenShell(barrier_faces)
                barrier_model = model.createIfcShellBasedSurfaceModel([barrier_shell])
                
                # Apply barrier style
                for face in barrier_faces:
                    model.createIfcStyledItem(face, [barrier_style], None)
                
                barrier_rep = model.createIfcShapeRepresentation(
                    body_context, "Body", "SurfaceModel", [barrier_model]
                )
                representations.append(barrier_rep)
            except Exception as e:
                logger.debug(f"Could not create bridge barriers: {e}")
    
    # Assign representation
    if representations:
        ifc_bridge.Representation = model.createIfcProductDefinitionShape(
            None, None, representations
        )
    else:
        logger.warning(f"Bridge {bridge.id}: no representations created")
        return None
    
    # Add properties
    _add_bridge_properties(model, ifc_bridge, bridge, width)
    
    logger.debug(f"Created IfcGeographicElement (BRIDGE): {bridge_name}")
    return ifc_bridge


def _add_bridge_properties(
    model: ifcopenshell.file,
    ifc_bridge: ifcopenshell.entity_instance,
    bridge,
    width: float
):
    """Add property sets to bridge element"""
    pset = ifcopenshell.api.run(
        "pset.add_pset",
        model,
        product=ifc_bridge,
        name="Pset_BridgeProperties"
    )
    
    properties = {
        "Width": width
    }
    
    if bridge.bridge_type:
        properties["BridgeType"] = bridge.bridge_type
    
    if bridge.name:
        properties["Name"] = bridge.name
    
    if bridge.structure:
        properties["Structure"] = bridge.structure
    
    if bridge.material:
        properties["Material"] = bridge.material
    
    if bridge.carries:
        properties["Carries"] = bridge.carries
    
    if bridge.layer is not None:
        properties["Layer"] = bridge.layer
    
    ifcopenshell.api.run(
        "pset.edit_pset",
        model,
        pset=pset,
        properties=properties
    )


def bridges_to_ifc(
    model: ifcopenshell.file,
    bridges: List,
    site: ifcopenshell.entity_instance,
    body_context,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    fetch_elevations_func=None
) -> List[ifcopenshell.entity_instance]:
    """
    Convert multiple bridges to IFC with terrain-projected elevations.
    
    Args:
        model: IFC model
        bridges: List of BridgeFeature objects
        site: Parent IfcSite element
        body_context: IFC Body context
        offset_x, offset_y, offset_z: Project origin offsets
        fetch_elevations_func: Function to fetch elevations for coordinates
    
    Returns:
        List of created IfcGeographicElement entities
    """
    if not bridges:
        return []
    
    ifc_bridges = []
    
    # Collect coordinates for batch elevation fetching
    if fetch_elevations_func:
        all_coords = []
        coord_counts = []
        for bridge in bridges:
            if bridge.geometry and not bridge.geometry.is_empty:
                coords = list(bridge.geometry.coords)
                all_coords.extend([(x, y) for x, y in coords])
                coord_counts.append(len(coords))
            else:
                coord_counts.append(0)
        
        # Fetch elevations
        if all_coords:
            logger.info(f"Fetching elevations for {len(all_coords)} bridge points...")
            all_elevations = fetch_elevations_func(all_coords)
            
            # Split elevations back to individual bridges
            idx = 0
            bridge_elevations = []
            for count in coord_counts:
                if count > 0:
                    bridge_elevations.append(all_elevations[idx:idx + count])
                    idx += count
                else:
                    bridge_elevations.append(None)
        else:
            bridge_elevations = [None] * len(bridges)
    else:
        bridge_elevations = [None] * len(bridges)
    
    # Create IFC elements
    for bridge, elevations in zip(bridges, bridge_elevations):
        ifc_bridge = bridge_to_ifc(
            model, bridge, site, body_context,
            offset_x, offset_y, offset_z, elevations
        )
        if ifc_bridge:
            ifc_bridges.append(ifc_bridge)
    
    logger.info(f"Created {len(ifc_bridges)}/{len(bridges)} bridge elements")
    return ifc_bridges

