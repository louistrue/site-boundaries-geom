import numpy as np
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import Polygon
import os
from workflow import drape_cadastral_to_3d, create_cadastral_ifc

def create_mock_data():
    """Create a mock DEM and a mock cadastral polygon for testing."""
    # 1. Create a mock DEM (100x100m, 1m resolution)
    # Origin in LV95: 2600000, 1200000
    res = 1.0
    width = 100
    height = 100
    transform = from_origin(2600000, 1200100, res, res)
    
    # Simple slope: height increases with Y
    data = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        data[y, :] = 400.0 + (height - y) * 0.1 # 400m to 410m slope
    
    dem_path = "mock_dem.tif"
    with rasterio.open(
        dem_path, 'w', driver='GTiff',
        height=height, width=width,
        count=1, dtype=data.dtype,
        crs='EPSG:2056',
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    
    # 2. Create a mock cadastral polygon
    # A rectangle within the DEM bounds
    poly = Polygon([
        (2600010, 1200010),
        (2600090, 1200010),
        (2600090, 1200090),
        (2600010, 1200090),
        (2600010, 1200010)
    ])
    
    gdf = gpd.GeoDataFrame({'parcel_id': ['MOCK_001'], 'geometry': [poly]}, crs="EPSG:2056")
    cadastral_path = "mock_cadastral.gpkg"
    gdf.to_file(cadastral_path, driver="GPKG")
    
    return cadastral_path, dem_path

def verify():
    cad_path, dem_path = create_mock_data()
    out_ifc = "test_output.ifc"
    
    print("--- Running Workflow Verification ---")
    try:
        # Step 1: Drape
        print("Draping geometries...")
        gdf = gpd.read_file(cad_path)
        gdf_3d = drape_cadastral_to_3d(gdf, dem_path, densify_interval=2.0)
        
        # Check Z coordinates
        ext_coords = list(gdf_3d.geometry.iloc[0].exterior.coords)
        print(f"Sample 3D coordinate: {ext_coords[0]}")
        
        if not all(len(c) == 3 for c in ext_coords):
            print("FAILED: Z coordinates missing!")
            return
            
        # Step 2: IFC Export
        print("Exporting to IFC...")
        create_cadastral_ifc(gdf_3d, out_ifc)
        
        if os.path.exists(out_ifc):
            print(f"SUCCESS: IFC created at {out_ifc}")
        else:
            print("FAILED: IFC not created.")
            
    finally:
        # Cleanup
        for p in [cad_path, dem_path]: # Keep IFC for manual inspection if needed
            if os.path.exists(p):
                os.remove(p)

if __name__ == "__main__":
    verify()
