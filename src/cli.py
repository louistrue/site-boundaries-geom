#!/usr/bin/env python3
"""
CLI for Site Boundaries Geometry Tool

Command-line interface for generating IFC site models with terrain, site solid, roads, trees, water, and buildings.
"""

# Minimal imports for fast startup
import argparse
import sys
import os

# IMMEDIATE feedback - print banner before any heavy imports
print("\n" + "=" * 60, flush=True)
print("  Swiss Site Model Generator", flush=True)
print("  IFC export with terrain, roads, water, trees & buildings", flush=True)
print("=" * 60, flush=True)

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Generate IFC site model with terrain, site solid, roads, trees, water, and buildings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full model with ALL features (recommended)
  %(prog)s --address "Bundesplatz 3, Bern" --all --output site.ifc
  
  # Using EGRID with all features
  %(prog)s --egrid CH999979659148 --all --output site.ifc
  
  # Custom selection
  %(prog)s --address "Bahnhofstrasse 1, Zürich" --include-roads --include-buildings --output custom.ifc
  
  # Large detailed area
  %(prog)s --address "Paradeplatz, Zürich" --all --radius 500 --resolution 10 --output detailed.ifc
        """
    )
    
    # Input group - either EGRID or address (mutually exclusive, one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--egrid",
                            help="Swiss EGRID identifier (e.g., CH999979659148)")
    input_group.add_argument("--address",
                            help="Swiss address (e.g., 'Bundesplatz 3, 3003 Bern')")
    
    # Optional location arguments
    parser.add_argument("--center-x", type=float,
                        help="Center easting (EPSG:2056). Default: site centroid")
    parser.add_argument("--center-y", type=float,
                        help="Center northing (EPSG:2056). Default: site centroid")
    
    # Terrain configuration
    terrain_group = parser.add_argument_group("Terrain Options")
    terrain_group.add_argument("--include-terrain", action="store_true", default=True,
                               help="Include surrounding terrain mesh (default: True)")
    terrain_group.add_argument("--no-terrain", dest="include_terrain", action="store_false",
                               help="Exclude surrounding terrain mesh")
    terrain_group.add_argument("--radius", type=float, default=500.0,
                               help="Radius of circular terrain area (meters), default: 500")
    terrain_group.add_argument("--resolution", type=float, default=10.0,
                               help="Grid resolution (meters), default: 10")
    terrain_group.add_argument("--attach-to-solid", action="store_true",
                               help="Attach terrain to smoothed site solid edges (less bumpy)")
    
    # Site solid configuration
    site_group = parser.add_argument_group("Site Solid Options")
    site_group.add_argument("--include-site-solid", action="store_true", default=True,
                            help="Include site solid (default: True)")
    site_group.add_argument("--no-site-solid", dest="include_site_solid", action="store_false",
                            help="Exclude site solid")
    site_group.add_argument("--densify", type=float, default=2.0,
                            help="Site boundary densification interval (meters), default: 2.0")
    
    # Road configuration
    road_group = parser.add_argument_group("Road Options")
    road_group.add_argument("--include-roads", action="store_true",
                            help="Include roads")
    road_group.add_argument("--road-buffer", type=float, default=100.0,
                            help="Buffer distance for road search (meters), default: 100")
    road_group.add_argument("--road-recess", type=float, default=0.15,
                            help="Depth to recess roads into terrain (meters), default: 0.15")
    road_group.add_argument("--roads-as-separate-elements", action="store_true",
                            help="Add roads as separate IFC elements (don't embed in terrain)")
    
    # Forest configuration
    forest_group = parser.add_argument_group("Forest Options")
    forest_group.add_argument("--include-forest", action="store_true",
                              help="Include forest trees and hedges")
    forest_group.add_argument("--forest-spacing", type=float, default=20.0,
                              help="Spacing between forest sample points (meters), default: 20")
    forest_group.add_argument("--forest-threshold", type=float, default=30.0,
                              help="Minimum forest coverage to place tree (0-100), default: 30")
    
    # Water configuration
    water_group = parser.add_argument_group("Water Options")
    water_group.add_argument("--include-water", action="store_true",
                             help="Include water features (creeks, rivers, lakes)")
    
    # Building configuration
    building_group = parser.add_argument_group("Building Options")
    building_group.add_argument("--include-buildings", action="store_true",
                                help="Include buildings from CityGML")
    
    # Railway configuration
    railway_group = parser.add_argument_group("Railway Options")
    railway_group.add_argument("--include-railways", action="store_true",
                               help="Include railways from OpenStreetMap")
    
    # Bridge configuration
    bridge_group = parser.add_argument_group("Bridge Options")
    bridge_group.add_argument("--include-bridges", action="store_true",
                              help="Include bridges from OpenStreetMap")
    
    # Convenience flag for all features
    parser.add_argument("--all", action="store_true",
                        help="Include ALL features (roads, forest, water, buildings, railways)")
    
    # Output
    parser.add_argument("--output", default="combined_terrain.ifc",
                        help="Output IFC file path")
    
    args = parser.parse_args()
    
    # Handle --all flag
    if args.all:
        args.include_roads = True
        args.include_forest = True
        args.include_water = True
        args.include_buildings = True
        args.include_railways = True
        # Note: bridges excluded from --all, use --include-bridges explicitly if needed
    
    # Build features list
    features = []
    if args.include_terrain:
        features.append("terrain")
    if args.include_site_solid:
        features.append("site")
    if args.include_roads:
        features.append("roads")
    if args.include_forest:
        features.append("trees")
    if args.include_water:
        features.append("water")
    if args.include_buildings:
        features.append("buildings")
    if args.include_railways:
        features.append("railways")
    if args.include_bridges:
        features.append("bridges")
    
    # Show configuration summary IMMEDIATELY
    print(f"\nConfiguration:", flush=True)
    print(f"  Radius: {args.radius}m | Resolution: {args.resolution}m", flush=True)
    print(f"  Features: {', '.join(features)}", flush=True)
    print(f"  Output: {args.output}", flush=True)
    
    # If address provided, resolve it EARLY (before heavy imports)
    egrid = args.egrid
    if args.address:
        print(f"\nResolving address: {args.address}", flush=True)
        from src.loaders.address import AddressResolver
        resolver = AddressResolver()
        result = resolver.resolve(args.address)
        if result is None:
            print(f"Error: Could not resolve address to cadastral parcel")
            sys.exit(1)
        egrid, address_metadata = result
        print(f"  EGRID: {egrid} (Canton: {address_metadata.get('canton', 'N/A')})", flush=True)
        if address_metadata.get('map_url'):
            print(f"  Verify: {address_metadata['map_url']}", flush=True)
    else:
        print(f"\nEGRID: {args.egrid}", flush=True)
    
    print("-" * 60, flush=True)
    
    # NOW import heavy modules (user sees progress)
    print("\nLoading modules...", flush=True)
    from src.site_model import run_combined_terrain_workflow
    import requests
    
    try:
        run_combined_terrain_workflow(
            egrid=egrid,
            address=None,  # Already resolved above
            center_x=args.center_x,
            center_y=args.center_y,
            radius=args.radius,
            resolution=args.resolution,
            densify=args.densify,
            attach_to_solid=args.attach_to_solid,
            include_terrain=args.include_terrain,
            include_site_solid=args.include_site_solid,
            include_roads=args.include_roads,
            include_forest=args.include_forest,
            include_water=args.include_water,
            include_buildings=args.include_buildings,
            include_railways=args.include_railways,
            include_bridges=args.include_bridges,
            road_buffer_m=args.road_buffer,
            road_recess_depth=args.road_recess,
            forest_spacing=args.forest_spacing,
            forest_threshold=args.forest_threshold,
            embed_roads_in_terrain=not args.roads_as_separate_elements,
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
