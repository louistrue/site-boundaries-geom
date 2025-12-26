#!/usr/bin/env python3
"""
Site Model Workflow

Main workflow orchestrator for generating IFC site models with terrain, site solid, roads, trees, water, and buildings.
"""

import numpy as np
import logging
from src.elevation import fetch_elevation_batch
from src.loaders.cadastre import fetch_boundary_by_egrid
from src.terrain_mesh import create_circular_terrain_grid, apply_road_recesses_to_terrain, triangulate_terrain_with_cutout
from src.site_geometry import create_site_solid_coords, calculate_height_offset
from src.ifc_builder import create_combined_ifc

logger = logging.getLogger(__name__)


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
    include_roads=False,
    include_forest=False,
    include_water=False,
    include_buildings=False,
    road_buffer_m=100.0,
    forest_spacing=20.0,
    forest_threshold=30.0,
    road_recess_depth=0.15,
    embed_roads_in_terrain=True,
    output_path="combined_terrain.ifc",
    return_model=False,
):
    """
    Run the combined terrain generation workflow.
    
    Args:
        egrid: Swiss EGRID number (required)
        center_x, center_y: Optional center coordinates (defaults to site centroid)
        radius: Radius of circular terrain area (meters)
        resolution: Grid resolution (meters)
        densify: Site boundary densification interval (meters)
        attach_to_solid: Attach terrain to smoothed site solid edges
        include_terrain: Include surrounding terrain mesh
        include_site_solid: Include site solid
        include_roads: Include roads
        include_forest: Include forest trees and hedges
        include_water: Include water features
        include_buildings: Include buildings from CityGML
        road_buffer_m: Buffer distance for road search (meters)
        forest_spacing: Spacing between forest sample points (meters)
        forest_threshold: Minimum forest coverage to place tree (0-100)
        road_recess_depth: Depth to recess roads into terrain (meters)
        embed_roads_in_terrain: Embed roads in terrain mesh vs separate elements
        output_path: Output IFC file path
        return_model: If True, return model object instead of writing to file
    
    Returns:
        If return_model=True: (model, offset_x, offset_y, offset_z)
        If return_model=False: (offset_x, offset_y, offset_z)
    """
    if not egrid:
        raise ValueError("EGRID is required for combined terrain generation.")

    # Fetch site boundary from cadastre
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
    z_offset = None
    roads = None  # Initialize roads variable

    # Get site boundary 3D coordinates
    ring = site_geometry.exterior
    distances = np.arange(0, ring.length, densify)
    if distances[-1] < ring.length:
        distances = np.append(distances, ring.length)

    site_points = [ring.interpolate(d) for d in distances]
    site_coords_2d = [(p.x, p.y) for p in site_points]

    print(f"\nFetching site boundary elevations ({len(site_coords_2d)} points)...")
    site_elevations = fetch_elevation_batch(site_coords_2d)
    site_coords_3d = [(x, y, z) for (x, y), z in zip(site_coords_2d, site_elevations, strict=True)]

    # Create terrain if requested
    if include_terrain:
        terrain_coords, circle_bounds = create_circular_terrain_grid(
            center_x, center_y, radius=radius, resolution=resolution
        )

        if len(terrain_coords) == 0:
            raise ValueError("No points generated in circular area.")

        print("\nFetching terrain elevations...")
        terrain_elevations = fetch_elevation_batch(terrain_coords)

        # Calculate height offset if site solid is also included
        if include_site_solid and z_offset is None:
            print("\nCalculating height offset for site solid...")
            z_offset = calculate_height_offset(
                site_geometry, site_coords_3d, terrain_coords, terrain_elevations
            )

        # Load roads EARLY if requested (needed before triangulation for recesses)
        roads = None
        if include_roads:
            print(f"\nLoading roads (buffer: {road_buffer_m}m)...")
            try:
                from src.loaders.road import get_roads_around_egrid
                roads, road_stats = get_roads_around_egrid(egrid, buffer_m=road_buffer_m)
                print(f"  Found {road_stats['count']} roads")
                print(f"  Total length: {road_stats['total_length_m']:.1f} m")
                if road_stats['road_classes']:
                    print("  Road classes:")
                    for cls, count in road_stats['road_classes'].items():
                        print(f"    - {cls}: {count}")
            except Exception as e:
                import traceback
                print(f"  ERROR loading roads: {e}")
                traceback.print_exc()
                roads = None

        # Apply road recesses if roads are included
        road_polygons = None
        road_edge_coords = None
        road_edge_elevations = None
        if roads and embed_roads_in_terrain:
            print("\nApplying road recesses to terrain...")
            road_polygons, road_edge_coords, road_edge_elevations = apply_road_recesses_to_terrain(
                roads, fetch_elevations_func=fetch_elevation_batch
            )

        # Triangulate terrain with cutouts
        print("\nTriangulating terrain...")
        terrain_triangles = triangulate_terrain_with_cutout(
            terrain_coords, terrain_elevations, site_geometry,
            site_boundary_coords=site_coords_2d,
            site_boundary_elevations=site_elevations,
            road_polygons=road_polygons,
            road_edge_coords=road_edge_coords,
            road_edge_elevations=road_edge_elevations
        )

    # Create site solid if requested
    if include_site_solid:
        print("\nCreating site solid...")
        if z_offset is None and include_terrain:
            z_offset = calculate_height_offset(
                site_geometry, site_coords_3d, terrain_coords, terrain_elevations
            )
        elif z_offset is None:
            z_offset = 0.0

        ext_coords, base_elevation, polygon_2d, smoothed_boundary_2d, smoothed_boundary_z = create_site_solid_coords(
            site_geometry, site_coords_3d, z_offset_adjustment=z_offset
        )

        site_solid_data = {
            'ext_coords': ext_coords,
            'base_elevation': base_elevation,
            'polygon_2d': polygon_2d,
            'smoothed_boundary_2d': smoothed_boundary_2d,
            'smoothed_boundary_z': smoothed_boundary_z
        }

    # Load additional features if requested
    forest_points = None
    if include_forest:
        print(f"\nLoading forest trees...")
        try:
            from src.loaders.forest import SwissTreeLoader
            loader = SwissTreeLoader()
            bounds = circle_bounds if circle_bounds else site_geometry.bounds
            forest_points = loader.get_trees_in_bounds(
                bounds, 
                fetch_elevations_func=fetch_elevation_batch,
                max_features=1000  # Limit to reasonable number
            )
            print(f"  Found {len(forest_points)} trees/hedges")
        except Exception as e:
            import traceback
            print(f"  ERROR loading forest: {e}")
            traceback.print_exc()
            forest_points = None

    waters = None
    if include_water:
        print(f"\nLoading water features...")
        try:
            from src.loaders.water import SwissWaterLoader
            loader = SwissWaterLoader()
            bounds = circle_bounds if circle_bounds else site_geometry.bounds
            waters = loader.get_water_in_bounds(bounds)
            print(f"  Found {len(waters)} water features")
        except Exception as e:
            import traceback
            print(f"  ERROR loading water: {e}")
            traceback.print_exc()
            waters = None

    buildings = None
    if include_buildings:
        print(f"\nLoading buildings...")
        try:
            from src.loaders.building import CityGMLBuildingLoader
            loader = CityGMLBuildingLoader()
            bounds = circle_bounds if circle_bounds else site_geometry.bounds
            buildings = loader.get_buildings_in_bbox(bounds, max_tiles=3)  # Allow multiple tiles for better coverage
            print(f"  Found {len(buildings)} buildings")
            if buildings:
                for i, b in enumerate(buildings[:5]):
                    print(f"    {i+1}. {b.id} - {b.building_type if b.building_type else 'unknown'} - {len(b.faces)} faces")
        except Exception as e:
            import traceback
            print(f"  ERROR loading buildings: {e}")
            traceback.print_exc()
            buildings = None

    # Create IFC file
    print(f"\nCreating IFC file: {output_path}")
    result = create_combined_ifc(
        terrain_triangles, site_solid_data, output_path, bounds,
        center_x, center_y, egrid=egrid, cadastre_metadata=cadastre_metadata,
        roads=roads, forest_points=forest_points, waters=waters, buildings=buildings,
        base_elevation=0.0, road_recess_depth=road_recess_depth, return_model=return_model
    )

    return result

