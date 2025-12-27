#!/usr/bin/env python3
"""
Test script to generate IFC file with railways and bridges

Tests the new railway and bridge loaders from OpenStreetMap.
"""

import logging
import sys
from src.site_model import run_combined_terrain_workflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Generate test IFC with railways and bridges"""
    
    # Use Zurich HB area (known to have railways and bridges)
    # EGRID for a parcel near Zurich HB
    # Or use address-based lookup
    address = "Bahnhofplatz, 8001 Zürich"
    
    print("=" * 80)
    print("Testing Railways and Bridges Loaders")
    print("=" * 80)
    print(f"Location: {address}")
    print(f"Radius: 500m")
    print()
    
    try:
        result = run_combined_terrain_workflow(
            address=address,
            radius=500.0,
            resolution=10.0,
            include_terrain=True,
            include_site_solid=True,
            include_roads=True,
            include_forest=False,  # Skip forest for faster generation
            include_water=True,
            include_buildings=False,  # Skip buildings for faster generation
            include_railways=True,  # Enable railways
            include_bridges=True,   # Enable bridges
            output_path="test_railways_bridges.ifc",
            return_model=False
        )
        
        print("\n" + "=" * 80)
        print("SUCCESS: Test IFC file created!")
        print("=" * 80)
        print(f"Output file: test_railways_bridges.ifc")
        print(f"Offsets: x={result[0]:.2f}, y={result[1]:.2f}, z={result[2]:.2f}")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR: Failed to create test IFC")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

