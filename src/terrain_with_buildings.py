"""
Extended terrain workflow with building integration

This module extends terrain_with_site.py to include building footprints and 3D models.
"""

import logging
from typing import Optional, List, Tuple

import ifcopenshell

from src.terrain_with_site import (
    run_combined_terrain_workflow,
    fetch_boundary_by_egrid
)
from shapely.geometry import Point

from src.citygml_loader import CityGMLBuildingLoader, CityGMLBuilding
from src.citygml_to_ifc import citygml_buildings_to_ifc


logger = logging.getLogger(__name__)


def run_terrain_with_buildings_workflow(
    egrid: Optional[str] = None,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
    radius: float = 500.0,
    resolution: float = 10.0,
    densify: float = 0.5,
    attach_to_solid: bool = False,
    include_terrain: bool = True,
    include_site_solid: bool = True,
    include_buildings: bool = True,
    building_buffer_m: float = 0.0,
    buildings_on_site_only: bool = True,
    output_path: str = "terrain_with_buildings.ifc",
) -> str:
    """
    Run the combined terrain generation workflow with building integration

    Args:
        egrid: Swiss EGRID identifier
        center_x, center_y: Center coordinates (EPSG:2056)
        radius: Radius of circular terrain area (meters)
        resolution: Grid resolution (meters)
        densify: Site boundary densification interval (meters)
        attach_to_solid: Attach terrain to smoothed site solid edges
        include_terrain: Include surrounding terrain mesh
        include_site_solid: Include site boundary solid
        include_buildings: Include complete 3D buildings from CityGML (default: True)
        building_buffer_m: Buffer around parcel to include buildings (meters)
        buildings_on_site_only: If True, only include buildings on/near site. If False, include all buildings in terrain radius
        output_path: Output IFC file path

    Returns:
        Path to generated IFC file
    """
    if not egrid:
        raise ValueError("EGRID is required for this workflow")

    # Step 1: Generate terrain and site (existing workflow)
    print("="*80)
    print("STEP 1: Generating terrain and site")
    print("="*80)

    # Generate terrain and site model in memory (no intermediate file)
    if include_buildings:
        # Return model in memory for building integration
        model, site_boundary, cadastre_metadata, offsets = run_combined_terrain_workflow(
            egrid=egrid,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            resolution=resolution,
            densify=densify,
            attach_to_solid=attach_to_solid,
            include_terrain=include_terrain,
            include_site_solid=include_site_solid,
            output_path=output_path,  # Not used when return_model=True
            return_model=True,
        )
        offset_x, offset_y, offset_z = offsets
    else:
        # No buildings - write directly to output
        run_combined_terrain_workflow(
            egrid=egrid,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            resolution=resolution,
            densify=densify,
            attach_to_solid=attach_to_solid,
            include_terrain=include_terrain,
            include_site_solid=include_site_solid,
            output_path=output_path,
            return_model=False,
        )
        print(f"\n✅ Output saved to: {output_path}")
        return output_path

    # Step 2: Load buildings if requested
    citygml_buildings = []
    if include_buildings:
        print("\n" + "="*80)
        print("STEP 2: Loading complete 3D buildings from CityGML")
        print("="*80)

        try:
            # Use site_boundary and metadata from Step 1 (no duplicate fetch)
            metadata = cadastre_metadata
            
            if site_boundary is None:
                raise ValueError(f"No boundary found for EGRID {egrid}")
            
            # Determine search area based on user preference
            if buildings_on_site_only:
                # Filter to buildings on/near site only
                if building_buffer_m > 0:
                    search_area = site_boundary.buffer(building_buffer_m)
                else:
                    search_area = site_boundary
                bbox = (
                    search_area.bounds[0] - 50,  # Small buffer for tile download
                    search_area.bounds[1] - 50,
                    search_area.bounds[2] + 50,
                    search_area.bounds[3] + 50
                )
                print("   Filtering: Buildings on site only")
            else:
                # Include all buildings in terrain radius
                from shapely.geometry import Point as ShapelyPoint
                # Use site boundary centroid (metadata doesn't have center_x/y)
                center_point = ShapelyPoint(site_boundary.centroid.x, site_boundary.centroid.y)
                search_area = center_point.buffer(radius)
                bbox = (
                    search_area.bounds[0] - 50,
                    search_area.bounds[1] - 50,
                    search_area.bounds[2] + 50,
                    search_area.bounds[3] + 50
                )
                print(f"   Filtering: All buildings within {radius}m radius")
            
            # Use CityGML for complete 3D buildings (default, fastest, most accurate)
            print("   Downloading CityGML tiles...")
            
            citygml_loader = CityGMLBuildingLoader()
            citygml_buildings = citygml_loader.get_buildings_in_bbox(bbox, max_tiles=1)
            
            # Filter to search area
            filtered_citygml = []
            for b in citygml_buildings:
                if b.centroid:
                    centroid_point = Point(b.centroid[0], b.centroid[1])
                    if search_area.intersects(centroid_point):
                        filtered_citygml.append(b)
            
            citygml_buildings = filtered_citygml
            
            print(f"\n✅ Loaded {len(citygml_buildings)} complete 3D buildings:")
            if citygml_buildings:
                heights = [b.z_max - b.z_min for b in citygml_buildings if b.z_max and b.z_min]
                if heights:
                    print(f"   Average height: {sum(heights)/len(heights):.1f}m")
                    print(f"   Max height: {max(heights):.1f}m")
                total_faces = sum(len(b.faces) for b in citygml_buildings)
                print(f"   Total faces: {total_faces}")
                print(f"   ✅ Complete lod2Solid geometry (walls + roofs + ground)")

        except Exception as e:
            logger.error(f"Failed to load buildings: {e}")
            print(f"\n⚠️  Building loading failed: {e}")
            print("   Continuing without buildings...")
            citygml_buildings = []

    # Step 3: Add buildings to IFC if any were loaded
    if citygml_buildings:
        print("\n" + "="*80)
        print("STEP 3: Adding buildings to IFC model")
        print("="*80)

        try:
            add_citygml_buildings_to_ifc_model(
                model=model,
                buildings=citygml_buildings,
                offset_x=offset_x,
                offset_y=offset_y,
                offset_z=offset_z,
            )

            # Write complete model to file (single write)
            print(f"   Saving complete IFC model to: {output_path}")
            model.write(output_path)

            print(f"\n✅ Successfully added {len(citygml_buildings)} complete 3D buildings to IFC")
            print(f"\n🎉 Final output: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Failed to add buildings to IFC: {e}")
            print(f"\n⚠️  Building integration failed: {e}")
            # Save model without buildings as fallback
            fallback_path = output_path.replace(".ifc", "_no_buildings.ifc")
            model.write(fallback_path)
            print(f"   Terrain saved to: {fallback_path}")
            return fallback_path
    else:
        # No buildings - write model directly
        print(f"\n   Saving IFC model to: {output_path}")
        model.write(output_path)
        print(f"\n✅ Output saved to: {output_path}")
        return output_path


def add_citygml_buildings_to_ifc_model(
    model: ifcopenshell.file,
    buildings: List[CityGMLBuilding],
    offset_x: float,
    offset_y: float,
    offset_z: float,
):
    """
    Add CityGML buildings to an existing IFC model (in memory)
    
    Uses complete lod2Solid geometry with walls, roofs, and ground surfaces.
    Follows IFC best practices with proper property sets.
    
    Args:
        model: IFC model object (in memory)
        buildings: List of CityGMLBuilding objects with complete 3D geometry
        offset_x, offset_y, offset_z: Project origin offsets
    """
    # Find contexts
    body_context = None
    footprint_context = None

    for context in model.by_type("IfcGeometricRepresentationContext"):
        if context.ContextIdentifier == "Body":
            body_context = context
        elif context.ContextIdentifier == "FootPrint":
            footprint_context = context

    if not body_context or not footprint_context:
        raise ValueError("Required IFC contexts not found in model")

    # Find site element
    sites = model.by_type("IfcSite")
    if not sites:
        raise ValueError("No IfcSite found in model")

    site = sites[0]  # Use first site

    print(f"   Project origin: E={offset_x}, N={offset_y}, H={offset_z}")

    # Add CityGML buildings to IFC
    print(f"   Converting {len(buildings)} CityGML buildings to IFC...")

    ifc_buildings = citygml_buildings_to_ifc(
        model=model,
        buildings=buildings,
        site=site,
        body_context=body_context,
        footprint_context=footprint_context,
        offset_x=offset_x,
        offset_y=offset_y,
        offset_z=offset_z
    )

    print(f"   ✅ Added {len(ifc_buildings)} complete 3D buildings to IFC model")


def main():
    """CLI for terrain with buildings workflow"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Create terrain with site and buildings"
    )
    parser.add_argument("--egrid", required=True, help="EGRID number")
    parser.add_argument("--center-x", type=float, help="Center easting (EPSG:2056)")
    parser.add_argument("--center-y", type=float, help="Center northing (EPSG:2056)")
    parser.add_argument("--radius", type=float, default=500.0,
                        help="Radius of circular terrain area (meters), default: 500")
    parser.add_argument("--resolution", type=float, default=10.0,
                        help="Grid resolution (meters), default: 10")
    parser.add_argument("--densify", type=float, default=2.0,
                        help="Site boundary densification interval (meters), default: 2.0 (lower=faster, higher=more precise)")
    parser.add_argument("--attach-to-solid", action="store_true",
                        help="Attach terrain to smoothed site solid edges")
    parser.add_argument("--no-terrain", action="store_true",
                        help="Don't include terrain mesh")
    parser.add_argument("--no-site", action="store_true",
                        help="Don't include site solid")
    parser.add_argument("--include-buildings", action="store_true",
                        help="Include complete 3D buildings from CityGML")
    parser.add_argument("--building-buffer", type=float, default=0.0,
                        help="Buffer around parcel to include buildings (meters), default: 0")
    parser.add_argument("--buildings-full-radius", action="store_true",
                        help="Include all buildings in terrain radius (default: site only)")
    parser.add_argument("--output", default="terrain_with_buildings.ifc",
                        help="Output IFC file path")

    args = parser.parse_args()

    try:
        run_terrain_with_buildings_workflow(
            egrid=args.egrid,
            center_x=args.center_x,
            center_y=args.center_y,
            radius=args.radius,
            resolution=args.resolution,
            densify=args.densify,
            attach_to_solid=args.attach_to_solid,
            include_terrain=not args.no_terrain,
            include_site_solid=not args.no_site,
            include_buildings=args.include_buildings,
            building_buffer_m=args.building_buffer,
            buildings_on_site_only=not args.buildings_full_radius,
            output_path=args.output,
        )

        print("\n✅ Workflow completed successfully!")
        sys.exit(0)

    except Exception as exc:
        logger.error(f"Workflow failed: {exc}", exc_info=True)
        print(f"\n❌ Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
