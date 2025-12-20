# Site Boundaries Geometry to IFC Converter

This tool converts Swiss cadastral site boundaries (EGRID parcels) into 3D IFC (Industry Foundation Classes) files with terrain geometry. It fetches cadastral boundaries from the Swiss geo.admin.ch API, samples elevation data, and generates georeferenced IFC files suitable for BIM applications.

## Features

- **Fetch cadastral boundaries** via EGRID from geo.admin.ch API
- **3D terrain generation** by sampling elevation data from DEM or API
- **IFC4 export** with proper georeferencing (EPSG:2056 Swiss LV95)
- **Terrain smoothing** to reduce noise while preserving overall slope
- **Solid geometry** with triangulated top surface and extruded sides

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Setup

1. Clone or navigate to the project directory:
```bash
cd site-boundaries-geom
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Fetch geometry for an EGRID and generate an IFC file:

```bash
python workflow.py --egrid CH999979659148 --output CH999979659148.ifc
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--egrid` | EGRID number to fetch boundary for | Required* |
| `--cadastral` | Path to local cadastral file (GeoPackage/Shapefile) | - |
| `--dem` | Path to local DEM file (GeoTIFF) | Uses API |
| `--output` | Output IFC file path | `output.ifc` |
| `--densify` | Densification interval in meters | `0.5` |
| `--offset-x` | Custom Easting offset (auto-calculated if 0) | `0.0` |
| `--offset-y` | Custom Northing offset (auto-calculated if 0) | `0.0` |
| `--offset-z` | Custom Height offset (auto-calculated if 0) | `0.0` |

*Either `--egrid` or `--cadastral` must be provided.

### Examples

**Using EGRID with API elevation service:**
```bash
python workflow.py --egrid CH999979659148 --output terrain.ifc
```

**Using local cadastral file and DEM:**
```bash
python workflow.py --cadastral parcels.gpkg --dem dem.tif --output terrain.ifc
```

**Reducing API calls with larger densification interval:**
```bash
python workflow.py --egrid CH999979659148 --densify 2.0 --output terrain.ifc
```

**Custom project origin:**
```bash
python workflow.py --egrid CH999979659148 --offset-x 2687100 --offset-y 1246400 --offset-z 600
```

## How It Works

The workflow consists of three main stages:

### 1. Boundary Fetching (`fetch_boundary_by_egrid`)

- Queries the geo.admin.ch API (`api3.geo.admin.ch/rest/services/ech/MapServer/find`)
- Searches for cadastral boundaries by EGRID in the `ch.kantone.cadastralwebmap-farbe` layer
- Returns a GeoDataFrame with the polygon geometry in EPSG:2056 (Swiss LV95)

**API Details:**
- Layer: `ch.kantone.cadastralwebmap-farbe`
- Search field: `egris_egrid`
- Coordinate system: EPSG:2056 (Swiss LV95)

### 2. 3D Draping (`drape_cadastral_to_3d`)

Converts 2D polygons to 3D by sampling elevation data:

1. **Densification**: Adds vertices along the polygon boundary at regular intervals (default: 0.5m)
   - Ensures sufficient detail for terrain representation
   - Smaller intervals = more accurate but slower (more API calls)

2. **Elevation Sampling**:
   - **With DEM**: Samples elevation from a local GeoTIFF using rasterio
   - **Without DEM**: Uses geo.admin.ch height API (`api3.geo.admin.ch/rest/services/height`)
     - Makes individual API calls for each point
     - Progress updates every 20 points

3. **3D Coordinate Creation**: Combines 2D coordinates with sampled elevations
4. **Interior Handling**: Processes any holes/interior rings in the polygon

### 3. IFC Generation (`create_cadastral_ifc`)

Creates a georeferenced IFC4 file with:

#### Project Setup
- Creates `IfcProject` with SI units (meters)
- Sets up model contexts (Body, FootPrint)
- Defines coordinate reference system:
  - **Projected CRS**: EPSG:2056 (Swiss LV95 / CH1903+)
  - **Vertical Datum**: LN02
- Calculates project origin (rounded to nearest 100m) if not specified

#### Site and Terrain Creation

For each cadastral parcel:

1. **IfcSite**: 
   - Created with EGRID as name
   - Contains a 2D footprint polyline representation
   - Aggregated to the project

2. **IfcGeographicElement** (Terrain):
   - Type: `TERRAIN`
   - Contains 3D solid geometry
   - Placed relative to the site

#### Geometry Processing

The terrain geometry undergoes several processing steps:

1. **Smoothing**:
   - **Best-fit plane**: Calculates a plane that best fits the elevation data
   - **Circular mean filter**: Smooths raw elevations and residuals separately
   - **Residual attenuation**: Reduces small bumps (20% of residual) while preserving overall slope
   - Result: Flat top surface with preserved tilt/orientation

2. **Triangulation**:
   - Triangulates the 2D polygon footprint
   - Assigns smoothed Z coordinates to each triangle vertex
   - Creates top faces for the solid

3. **Solid Creation**:
   - **Top faces**: Triangulated surface with smoothed elevations
   - **Side faces** ("skirt"): Extruded down from boundary to base elevation (2m below lowest point)
   - **Bottom face**: Closes the solid at the base elevation
   - Result: Closed manifold solid (FacetedBrep)

#### Coordinate Transformation

All coordinates are transformed relative to the project origin:
- `local_x = easting - offset_x`
- `local_y = northing - offset_y`
- `local_z = elevation - offset_z`

This keeps coordinates manageable while preserving georeferencing through the `IfcMapConversion`.

## Technical Details

### Coordinate Systems

- **Input/Output CRS**: EPSG:2056 (Swiss LV95 / CH1903+)
- **Vertical Datum**: LN02 (Swiss height system)
- **Units**: Meters (SI)

### Elevation Data Sources

1. **geo.admin.ch Height API**:
   - Endpoint: `https://api3.geo.admin.ch/rest/services/height`
   - Parameters: `easting`, `northing`, `sr=2056`
   - Returns: JSON with `height` field
   - Rate limits: Individual requests, may be slow for many points

2. **Local DEM** (GeoTIFF):
   - Must be in EPSG:2056
   - Faster than API for large datasets
   - Recommended for batch processing

### Smoothing Algorithm

The terrain smoothing uses a multi-step process:

1. Calculate best-fit plane using least squares
2. Apply circular mean filter (window size: 9) to raw elevations
3. Calculate residuals (smoothed - plane)
4. Smooth residuals with circular mean filter
5. Attenuate residuals (20% scale factor)
6. Final elevation = plane + attenuated residuals

This preserves the overall slope while removing small terrain variations.

### IFC Structure

```
IfcProject
└── IfcSite (EGRID)
    ├── Representation: FootPrint (2D polyline)
    └── IfcGeographicElement (Terrain)
        ├── PredefinedType: TERRAIN
        └── Representation: Body (FacetedBrep solid)
```

## Troubleshooting

### API Rate Limits

If you encounter rate limiting with the elevation API:
- Use a local DEM file (`--dem`)
- Increase densification interval (`--densify 2.0` or higher)
- Process smaller parcels

### Invalid Geometry

If you see "invalid footprint" warnings:
- The polygon may have self-intersections
- The code attempts to fix with `buffer(0)`
- Check the source cadastral data

### Python Version Issues

Ensure you're using Python 3.9+:
```bash
python --version
```

If using ifcopenshell, ensure compatibility with your Python version.

## Output

The generated IFC file contains:
- Georeferenced site boundaries
- 3D terrain geometry as solid volumes
- Proper IFC4 structure for BIM applications
- Compatible with major IFC viewers (BlenderBIM, Solibri, etc.)

## License

See LICENSE file for details.

## References

- [geo.admin.ch API Documentation](https://api3.geo.admin.ch/)
- [IFC4 Schema](https://www.buildingsmart.org/standards/bsi-standards/industry-foundation-classes-ifc/)
- [Swiss LV95 Coordinate System](https://www.swisstopo.admin.ch/en/knowledge-facts/surveying-geodesy/reference-systems/switzerland.html)

