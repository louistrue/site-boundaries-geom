#!/usr/bin/env python3
"""
Generate showcase IFC files for famous Swiss locations.

Usage:
    python generate_showcases.py              # Generate all showcases
    python generate_showcases.py --quick      # Faster generation (smaller radius, lower resolution)
    python generate_showcases.py --list       # List available locations
"""

import subprocess
import sys
import os
from pathlib import Path

# Famous Swiss locations for showcases
# Note: Building data availability varies by format:
# - Some regions have CityGML format (2019 and earlier surveys)
# - Newer regions (2020+) have FileGDB format
# The loader automatically handles both formats for full 3D building support.
LOCATIONS = [
    {
        "name": "bern_bundeshaus",
        "address": "Bundesplatz 3, Bern",
        "description": "Federal Palace & Parliament Square"
    },
    {
        "name": "zurich_paradeplatz",
        "address": "Paradeplatz, Zürich",
        "description": "Financial District & Banking Center"
    },
    {
        "name": "basel_rathausplatz",
        "address": "Rathausplatz, Basel",
        "description": "Town Hall Square & Rhine River"
    },
    {
        "name": "luzern_bahnhofplatz",
        "address": "Bahnhofplatz, Luzern",
        "description": "Train Station & Lake Lucerne Waterfront"
    },
    {
        "name": "geneva_molard",
        "address": "Place du Molard, Genève",
        "description": "Old Town Square & Lake Geneva"
    },
    {
        "name": "lausanne_oldtown",
        "address": "Place de la Palud 2, Lausanne",
        "description": "Historic Old Town & Fountain"
    },
    {
        "name": "interlaken_hoeheweg",
        "address": "Höheweg, Interlaken",
        "description": "Between Two Lakes with Alpine Panorama"
    },
]

# Default settings
DEFAULT_RADIUS = 500
DEFAULT_RESOLUTION = 10

# Quick settings (for faster testing)
QUICK_RADIUS = 250
QUICK_RESOLUTION = 20


def generate_showcase(location: dict, radius: int, resolution: int, output_dir: str = "results"):
    """Generate a single showcase IFC file."""
    name = location["name"]
    address = location["address"]
    output_path = f"{output_dir}/showcase_{name}.ifc"
    
    print(f"\n{'='*60}")
    print(f"  Generating: {name}")
    print(f"  Address: {address}")
    print(f"  Radius: {radius}m | Resolution: {resolution}m")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, "src/cli.py",
        "--address", address,
        "--all",
        "--radius", str(radius),
        "--resolution", str(resolution),
        "--output", output_path
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ SUCCESS: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {name} (exit code {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  INTERRUPTED: {name}")
        raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Swiss site model showcases")
    parser.add_argument("--quick", action="store_true", 
                        help=f"Quick mode: {QUICK_RADIUS}m radius, {QUICK_RESOLUTION}m resolution")
    parser.add_argument("--radius", type=int, default=None,
                        help=f"Custom radius in meters (default: {DEFAULT_RADIUS})")
    parser.add_argument("--resolution", type=int, default=None,
                        help=f"Custom resolution in meters (default: {DEFAULT_RESOLUTION})")
    parser.add_argument("--list", action="store_true",
                        help="List available locations and exit")
    parser.add_argument("--only", type=str, default=None,
                        help="Generate only specific location(s), comma-separated names")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for IFC files (default: results)")
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("\nAvailable showcase locations:")
        print("-" * 50)
        for loc in LOCATIONS:
            print(f"  {loc['name']:25} - {loc['description']}")
        print()
        return
    
    # Determine settings
    if args.quick:
        radius = QUICK_RADIUS
        resolution = QUICK_RESOLUTION
    else:
        radius = args.radius or DEFAULT_RADIUS
        resolution = args.resolution or DEFAULT_RESOLUTION
    
    # Filter locations if --only specified
    if args.only:
        only_names = [n.strip() for n in args.only.split(",")]
        locations = [loc for loc in LOCATIONS if loc["name"] in only_names]
        if not locations:
            print(f"❌ No matching locations found for: {args.only}")
            print("   Use --list to see available locations")
            sys.exit(1)
    else:
        locations = LOCATIONS
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate showcases
    print("\n" + "=" * 60)
    print("  Swiss Site Model - Showcase Generator")
    print("=" * 60)
    print(f"\nLocations: {len(locations)}")
    print(f"Radius: {radius}m")
    print(f"Resolution: {resolution}m")
    print(f"Output: {args.output_dir}/")
    
    success_count = 0
    failed = []
    
    try:
        for i, location in enumerate(locations, 1):
            print(f"\n[{i}/{len(locations)}]", end="")
            if generate_showcase(location, radius, resolution, args.output_dir):
                success_count += 1
            else:
                failed.append(location["name"])
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  ✅ Successful: {success_count}/{len(locations)}")
    if failed:
        print(f"  ❌ Failed: {', '.join(failed)}")
    print(f"  📁 Output: {args.output_dir}/showcase_*.ifc")
    print()


if __name__ == "__main__":
    main()

