"""
Test script to compare different approaches for efficiently getting Swiss building data.

This script tests:
1. GeoAdmin REST API (MapServer) - for building footprints
2. STAC API - for swissBUILDINGS3D 3.0 data
3. WFS service - for vector building data

Performance metrics tracked:
- Response time
- Data size
- Number of buildings retrieved
- Data format and quality
"""

import requests
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """Store benchmark results for an API test"""
    api_name: str
    request_url: str
    response_time_ms: float
    status_code: int
    data_size_bytes: int
    num_buildings: int
    success: bool
    error_message: Optional[str] = None
    format: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class BuildingAPITester:
    """Test different Swiss building data APIs"""

    def __init__(self, bbox: Tuple[float, float, float, float] = None, egrid: str = None):
        """
        Initialize tester with either bbox or EGRID

        Args:
            bbox: Bounding box (min_x, min_y, max_x, max_y) in EPSG:2056
            egrid: Swiss EGRID identifier to get buildings around
        """
        self.bbox = bbox
        self.egrid = egrid
        self.results: List[BenchmarkResult] = []

        # Default test area: Zurich city center (small area for testing)
        if not self.bbox and not self.egrid:
            # Small bbox around Zurich HB (500m x 500m)
            self.bbox = (2682500, 1247500, 2683000, 1248000)

    def _time_request(self, func, *args, **kwargs) -> Tuple[any, float]:
        """Time a request function"""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - start) * 1000
        return result, elapsed_ms

    def test_geoadmin_identify(self) -> BenchmarkResult:
        """
        Test GeoAdmin MapServer Identify API
        Fast for point queries but limited for area queries
        """
        print("\n🔍 Testing GeoAdmin Identify API...")

        if not self.bbox:
            return BenchmarkResult(
                api_name="GeoAdmin Identify",
                request_url="N/A",
                response_time_ms=0,
                status_code=0,
                data_size_bytes=0,
                num_buildings=0,
                success=False,
                error_message="Requires bbox"
            )

        # Use center point of bbox
        center_x = (self.bbox[0] + self.bbox[2]) / 2
        center_y = (self.bbox[1] + self.bbox[3]) / 2

        url = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
        params = {
            "geometryType": "esriGeometryPoint",
            "geometry": f"{center_x},{center_y}",
            "layers": "all:ch.swisstopo.swissbuildings3d_3_0-beta",
            "mapExtent": f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}",
            "imageDisplay": "1000,1000,96",
            "tolerance": 50,
            "returnGeometry": "true",
            "geometryFormat": "geojson",
            "sr": "2056"
        }

        try:
            response, elapsed = self._time_request(
                requests.get, url, params=params, timeout=30
            )

            num_buildings = 0
            if response.status_code == 200:
                data = response.json()
                num_buildings = len(data.get("results", []))

            result = BenchmarkResult(
                api_name="GeoAdmin Identify (Point)",
                request_url=response.url,
                response_time_ms=elapsed,
                status_code=response.status_code,
                data_size_bytes=len(response.content),
                num_buildings=num_buildings,
                success=response.status_code == 200,
                format="GeoJSON"
            )

            print(f"  ✓ Found {num_buildings} buildings in {elapsed:.0f}ms")
            return result

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return BenchmarkResult(
                api_name="GeoAdmin Identify",
                request_url=url,
                response_time_ms=0,
                status_code=0,
                data_size_bytes=0,
                num_buildings=0,
                success=False,
                error_message=str(e)
            )

    def test_stac_api(self) -> BenchmarkResult:
        """
        Test STAC API for swissBUILDINGS3D 3.0
        Best for bulk downloads of tiled data
        """
        print("\n📦 Testing STAC API...")

        if not self.bbox:
            return BenchmarkResult(
                api_name="STAC API",
                request_url="N/A",
                response_time_ms=0,
                status_code=0,
                data_size_bytes=0,
                num_buildings=0,
                success=False,
                error_message="Requires bbox"
            )

        # Convert EPSG:2056 to WGS84 (approximate)
        # For testing, use rough conversion
        # min_x, min_y, max_x, max_y in EPSG:2056
        # Need to convert to lon/lat for STAC
        # Rough approximation for Swiss coordinates
        lon_min = (self.bbox[0] - 2600000) / 111320 + 7.44
        lat_min = (self.bbox[1] - 1200000) / 111320 + 46.0
        lon_max = (self.bbox[2] - 2600000) / 111320 + 7.44
        lat_max = (self.bbox[3] - 1200000) / 111320 + 46.0

        url = "https://data.geo.admin.ch/api/stac/v1/collections/ch.swisstopo.swissbuildings3d_3_0/items"
        params = {
            "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "limit": 10
        }

        try:
            response, elapsed = self._time_request(
                requests.get, url, params=params, timeout=30
            )

            num_items = 0
            if response.status_code == 200:
                data = response.json()
                num_items = len(data.get("features", []))

            result = BenchmarkResult(
                api_name="STAC API",
                request_url=response.url,
                response_time_ms=elapsed,
                status_code=response.status_code,
                data_size_bytes=len(response.content),
                num_buildings=num_items,  # These are tiles, not individual buildings
                success=response.status_code == 200,
                format="STAC GeoJSON",
                error_message=None if response.status_code == 200 else f"Status: {response.status_code}"
            )

            print(f"  ✓ Found {num_items} data tiles in {elapsed:.0f}ms")
            return result

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return BenchmarkResult(
                api_name="STAC API",
                request_url=url,
                response_time_ms=0,
                status_code=0,
                data_size_bytes=0,
                num_buildings=0,
                success=False,
                error_message=str(e)
            )

    def test_wfs_service(self) -> BenchmarkResult:
        """
        Test WFS (Web Feature Service) for building footprints
        Good for vector data with attribute queries
        """
        print("\n🗺️  Testing WFS Service...")

        if not self.bbox:
            return BenchmarkResult(
                api_name="WFS Service",
                request_url="N/A",
                response_time_ms=0,
                status_code=0,
                data_size_bytes=0,
                num_buildings=0,
                success=False,
                error_message="Requires bbox"
            )

        # WFS GetFeature request
        url = "https://wms.geo.admin.ch/"
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": "ch.swisstopo.swissbuildings3d_3_0",
            "srsName": "EPSG:2056",
            "bbox": f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]},EPSG:2056",
            "outputFormat": "application/json",
            "count": 100  # Limit results
        }

        try:
            response, elapsed = self._time_request(
                requests.get, url, params=params, timeout=30
            )

            num_buildings = 0
            if response.status_code == 200:
                try:
                    data = response.json()
                    num_buildings = len(data.get("features", []))
                except:
                    # Might be XML/GML
                    pass

            result = BenchmarkResult(
                api_name="WFS Service",
                request_url=response.url,
                response_time_ms=elapsed,
                status_code=response.status_code,
                data_size_bytes=len(response.content),
                num_buildings=num_buildings,
                success=response.status_code == 200,
                format="GeoJSON/GML",
                error_message=None if response.status_code == 200 else f"Status: {response.status_code}"
            )

            print(f"  ✓ Found {num_buildings} buildings in {elapsed:.0f}ms")
            return result

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return BenchmarkResult(
                api_name="WFS Service",
                request_url=url,
                response_time_ms=0,
                status_code=0,
                data_size_bytes=0,
                num_buildings=0,
                success=False,
                error_message=str(e)
            )

    def test_geoadmin_find(self) -> BenchmarkResult:
        """
        Test GeoAdmin Find API
        Good for searching by attributes
        """
        print("\n🔎 Testing GeoAdmin Find API...")

        url = "https://api3.geo.admin.ch/rest/services/ech/MapServer/find"
        params = {
            "layer": "ch.swisstopo.swissbuildings3d_3_0",
            "searchText": "*",
            "searchField": "id",
            "returnGeometry": "true",
            "geometryFormat": "geojson",
            "sr": "2056",
            "limit": 100
        }

        # If we have a bbox, add it as a filter
        if self.bbox:
            params["bbox"] = f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}"

        try:
            response, elapsed = self._time_request(
                requests.get, url, params=params, timeout=30
            )

            num_buildings = 0
            if response.status_code == 200:
                try:
                    data = response.json()
                    num_buildings = len(data.get("results", []))
                except:
                    pass

            result = BenchmarkResult(
                api_name="GeoAdmin Find",
                request_url=response.url,
                response_time_ms=elapsed,
                status_code=response.status_code,
                data_size_bytes=len(response.content),
                num_buildings=num_buildings,
                success=response.status_code == 200,
                format="GeoJSON",
                error_message=None if response.status_code == 200 else f"Status: {response.status_code}"
            )

            print(f"  ✓ Found {num_buildings} buildings in {elapsed:.0f}ms")
            return result

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return BenchmarkResult(
                api_name="GeoAdmin Find",
                request_url=url,
                response_time_ms=0,
                status_code=0,
                data_size_bytes=0,
                num_buildings=0,
                success=False,
                error_message=str(e)
            )

    def run_all_tests(self) -> List[BenchmarkResult]:
        """Run all API tests and return results"""
        print("=" * 80)
        print("🏗️  SWISS BUILDING DATA API BENCHMARK")
        print("=" * 80)

        if self.bbox:
            print(f"\n📍 Test area: BBOX {self.bbox} (EPSG:2056)")
        elif self.egrid:
            print(f"\n📍 Test area: EGRID {self.egrid}")

        # Run all tests
        self.results.append(self.test_geoadmin_identify())
        self.results.append(self.test_stac_api())
        self.results.append(self.test_wfs_service())
        self.results.append(self.test_geoadmin_find())

        return self.results

    def print_summary(self):
        """Print a summary comparison of all results"""
        print("\n" + "=" * 80)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 80)

        # Sort by response time
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        if successful:
            print("\n✅ Successful APIs (sorted by speed):\n")
            successful.sort(key=lambda x: x.response_time_ms)

            for i, result in enumerate(successful, 1):
                print(f"{i}. {result.api_name}")
                print(f"   ⏱️  Response time: {result.response_time_ms:.0f}ms")
                print(f"   📦 Data size: {result.data_size_bytes:,} bytes ({result.data_size_bytes/1024:.1f} KB)")
                print(f"   🏢 Buildings/Items: {result.num_buildings}")
                print(f"   📄 Format: {result.format}")
                print(f"   🔗 Efficiency: {result.num_buildings / (result.response_time_ms/1000):.1f} items/sec")
                print()

        if failed:
            print("❌ Failed APIs:\n")
            for result in failed:
                print(f"   • {result.api_name}: {result.error_message}")
                print()

        # Recommendations
        print("=" * 80)
        print("💡 RECOMMENDATIONS")
        print("=" * 80)
        print()

        if successful:
            fastest = successful[0]
            print(f"🚀 Fastest: {fastest.api_name} ({fastest.response_time_ms:.0f}ms)")

            most_data = max(successful, key=lambda x: x.num_buildings)
            print(f"📊 Most data: {most_data.api_name} ({most_data.num_buildings} items)")

            most_efficient = max(successful, key=lambda x: x.num_buildings / (x.response_time_ms/1000) if x.response_time_ms > 0 else 0)
            print(f"⚡ Most efficient: {most_efficient.api_name} ({most_efficient.num_buildings / (most_efficient.response_time_ms/1000):.1f} items/sec)")

        print("\n" + "=" * 80)

    def export_results(self, filename: str = "building_api_benchmark.json"):
        """Export results to JSON file"""
        with open(filename, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        print(f"\n💾 Results exported to {filename}")


def main():
    """Run benchmark tests"""

    # Test different scenarios
    scenarios = [
        {
            "name": "Small area (Zurich HB - 500m x 500m)",
            "bbox": (2682500, 1247500, 2683000, 1248000),
        },
        {
            "name": "Medium area (Zurich center - 1km x 1km)",
            "bbox": (2682000, 1247000, 2683000, 1248000),
        },
    ]

    all_results = []

    for scenario in scenarios:
        print(f"\n\n{'='*80}")
        print(f"🧪 SCENARIO: {scenario['name']}")
        print(f"{'='*80}")

        tester = BuildingAPITester(bbox=scenario.get('bbox'))
        results = tester.run_all_tests()
        tester.print_summary()

        all_results.extend(results)

    # Export combined results
    with open("all_building_api_benchmarks.json", 'w') as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)

    print("\n" + "="*80)
    print("✅ All benchmarks complete!")
    print("="*80)


if __name__ == "__main__":
    main()
