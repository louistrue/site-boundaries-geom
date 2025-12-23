#!/usr/bin/env python3
"""
Demonstration of Swiss Building Loader

Shows how to efficiently get building data around a parcel or in an area.
Run this script when network access is available to test the APIs.
"""

import logging
import json
from src.building_loader import (
    SwissBuildingLoader,
    get_buildings_around_egrid,
    get_buildings_in_bbox
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def demo_bbox_query():
    """Demo: Get buildings in a bounding box"""
    print("\n" + "="*80)
    print("DEMO 1: Get buildings in bounding box")
    print("="*80)
    print("\n📍 Location: Zurich HB area (500m × 500m)")

    bbox = (2682500, 1247500, 2683000, 1248000)
    print(f"📦 Bounding box (EPSG:2056): {bbox}")

    try:
        buildings, stats = get_buildings_in_bbox(bbox, method="wfs")

        print(f"\n✅ SUCCESS! Retrieved {stats['count']} buildings")
        print("\n📊 Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key:30s}: {value:10.1f}")
            else:
                print(f"   {key:30s}: {value:10}")

        # Show first 3 buildings
        if buildings:
            print(f"\n🏢 Sample buildings:")
            for i, building in enumerate(buildings[:3], 1):
                print(f"\n   Building {i}:")
                print(f"      ID: {building.id}")
                print(f"      Height: {building.height:.1f}m" if building.height else "      Height: N/A")
                print(f"      Footprint area: {building.geometry.area:.1f}m²")
                if building.building_class:
                    print(f"      Class: {building.building_class}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 This is expected if running without network access.")


def demo_egrid_query():
    """Demo: Get buildings on a cadastral parcel"""
    print("\n" + "="*80)
    print("DEMO 2: Get buildings on cadastral parcel")
    print("="*80)

    egrid = "CH999979659148"
    buffer_m = 10

    print(f"\n📍 EGRID: {egrid}")
    print(f"📦 Buffer: {buffer_m}m around parcel boundary")

    try:
        buildings, stats = get_buildings_around_egrid(egrid, buffer_m=buffer_m)

        print(f"\n✅ SUCCESS! Found {stats['count']} buildings on parcel")
        print("\n📊 Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key:30s}: {value:10.1f}")
            else:
                print(f"   {key:30s}: {value:10}")

        # Show all buildings
        if buildings:
            print(f"\n🏢 Buildings on parcel:")
            for i, building in enumerate(buildings, 1):
                print(f"\n   Building {i}:")
                print(f"      ID: {building.id}")
                print(f"      Height: {building.height:.1f}m" if building.height else "      Height: N/A")
                print(f"      Footprint area: {building.geometry.area:.1f}m²")
                if building.roof_type:
                    print(f"      Roof type: {building.roof_type}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 This is expected if running without network access.")


def demo_filtered_query():
    """Demo: Get tall buildings only"""
    print("\n" + "="*80)
    print("DEMO 3: Get tall buildings (>30m) in area")
    print("="*80)

    bbox = (2682000, 1247000, 2683000, 1248000)  # 1km × 1km
    min_height = 30

    print(f"\n📍 Location: Zurich center (1km × 1km)")
    print(f"📏 Filter: Buildings > {min_height}m tall")

    try:
        loader = SwissBuildingLoader()
        buildings = loader.get_buildings_by_height(
            bbox_2056=bbox,
            min_height=min_height
        )

        print(f"\n✅ SUCCESS! Found {len(buildings)} tall buildings")

        if buildings:
            # Calculate stats
            stats = loader.get_building_statistics(buildings)

            print("\n📊 Statistics:")
            print(f"   Tallest building: {stats['max_height_m']:.1f}m")
            print(f"   Average height: {stats['avg_height_m']:.1f}m")
            print(f"   Total footprint: {stats['total_footprint_area_m2']:.0f}m²")

            # Show tallest 3
            sorted_buildings = sorted(
                buildings,
                key=lambda b: b.height if b.height else 0,
                reverse=True
            )

            print(f"\n🏙️  Top 3 tallest buildings:")
            for i, building in enumerate(sorted_buildings[:3], 1):
                print(f"\n   #{i}: {building.id}")
                print(f"      Height: {building.height:.1f}m" if building.height else "      Height: N/A")
                print(f"      Footprint: {building.geometry.area:.1f}m²")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 This is expected if running without network access.")


def demo_export_geojson():
    """Demo: Export buildings to GeoJSON"""
    print("\n" + "="*80)
    print("DEMO 4: Export buildings to GeoJSON")
    print("="*80)

    bbox = (2682500, 1247500, 2683000, 1248000)

    try:
        buildings, stats = get_buildings_in_bbox(bbox, method="wfs")

        # Convert to GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        for building in buildings[:10]:  # Export first 10
            feature = {
                "type": "Feature",
                "id": building.id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(building.geometry.exterior.coords)]
                },
                "properties": {
                    "height": building.height,
                    "building_class": building.building_class,
                    "roof_type": building.roof_type,
                    "year_built": building.year_built,
                    "area_m2": building.geometry.area
                }
            }
            geojson["features"].append(feature)

        # Save to file
        output_file = "buildings_export.geojson"
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)

        print(f"\n✅ SUCCESS! Exported {len(geojson['features'])} buildings")
        print(f"📁 File saved: {output_file}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 This is expected if running without network access.")


def demo_performance_comparison():
    """Demo: Compare different API methods"""
    print("\n" + "="*80)
    print("DEMO 5: Performance comparison (WFS vs STAC)")
    print("="*80)

    bbox = (2682500, 1247500, 2683000, 1248000)

    print(f"\n📍 Test area: 500m × 500m (Zurich HB)")

    # Test WFS
    print(f"\n🔄 Testing WFS API...")
    try:
        import time
        start = time.time()
        buildings_wfs, stats_wfs = get_buildings_in_bbox(bbox, method="wfs")
        wfs_time = time.time() - start

        print(f"   ✅ WFS: {stats_wfs['count']} buildings in {wfs_time:.2f}s")
        print(f"   ⚡ Efficiency: {stats_wfs['count']/wfs_time:.1f} buildings/sec")

    except Exception as e:
        print(f"   ❌ WFS failed: {e}")

    # Test STAC (tiles only, no parsing yet)
    print(f"\n🔄 Testing STAC API...")
    try:
        start = time.time()
        loader = SwissBuildingLoader()
        tiles = loader.get_buildings_stac(bbox)
        stac_time = time.time() - start

        print(f"   ✅ STAC: {len(tiles)} tiles in {stac_time:.2f}s")
        print(f"   ⚠️  Note: Tiles need additional parsing (not implemented)")

    except Exception as e:
        print(f"   ❌ STAC failed: {e}")


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("🏗️  SWISS BUILDING LOADER - INTERACTIVE DEMO")
    print("="*80)
    print("\nThis demo shows how to efficiently get Swiss building data.")
    print("Network access to geo.admin.ch APIs is required.\n")

    demos = [
        ("Bounding box query", demo_bbox_query),
        ("EGRID parcel query", demo_egrid_query),
        ("Filtered query (tall buildings)", demo_filtered_query),
        ("Export to GeoJSON", demo_export_geojson),
        ("Performance comparison", demo_performance_comparison),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Demo '{name}' failed: {e}")

    print("\n" + "="*80)
    print("✅ Demo complete!")
    print("="*80)
    print("\n📚 For more information, see:")
    print("   - BUILDING_API_GUIDE.md (detailed API documentation)")
    print("   - BUILDING_EFFICIENCY_TEST_RESULTS.md (test results & recommendations)")
    print("   - src/building_loader.py (implementation)")
    print()


if __name__ == "__main__":
    main()
