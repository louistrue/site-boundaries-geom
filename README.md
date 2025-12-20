# Site Boundaries Geometry to IFC Converter

This tool converts Swiss cadastral site boundaries (EGRID parcels) into 3D IFC (Industry Foundation Classes) files with terrain geometry. It fetches cadastral boundaries from the Swiss geo.admin.ch API, samples elevation data, and generates georeferenced IFC files suitable for BIM applications.

## Features

- **Fetch cadastral boundaries** via EGRID from geo.admin.ch API
- **3D terrain generation** by sampling elevation data from DEM or API
- **IFC4 export** with proper georeferencing (EPSG:2056 Swiss LV95)
- **Terrain smoothing** to reduce noise while preserving overall slope
- **Solid geometry** with triangulated top surface and extruded sides
- **Combined terrain workflow** - site solid with surrounding terrain mesh and precise cutout
- **IFC schema compliance** - proper property sets (Pset_LandRegistration, Pset_SiteCommon, Qto_SiteBaseQuantities)
- **Cadastre metadata** - automatic extraction and mapping to IFC properties
- **Optional 3D buildings** - include swissBUILDINGS3D 3.0 Beta meshes around the site

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

### Combined Terrain Workflow (Recommended)

The `combined_terrain.py` script creates a single IFC file containing:
- **Surrounding terrain mesh** in a circular area around the site
- **Site solid** with smoothed surface, height-adjusted to align with terrain
- **Precise cutout** in terrain that follows the exact site boundary shape
- **Full cadastre metadata** mapped to IFC schema property sets

#### Basic Usage

```bash
python combined_terrain.py --egrid CH999979659148 --radius 500 --output combined.ifc
```

#### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--egrid` | EGRID number to fetch boundary for | Required* |
| `--center-x` | Center easting (EPSG:2056) if no EGRID | - |
| `--center-y` | Center northing (EPSG:2056) if no EGRID | - |
| `--radius` | Radius of circular terrain area (meters) | `500` |
| `--resolution` | Grid resolution in meters (lower = more detail) | `10` |
| `--densify` | Site boundary densification interval (meters) | `0.5` |
| `--attach-to-solid` | Attach terrain to smoothed site solid edges (less bumpy) | False |
| `--include-buildings` | Include swissBUILDINGS3D 3.0 Beta buildings | False |
| `--buildings-separate` | Export swissBUILDINGS3D roofs/facades/footprints as separate objects when Z is present | False |
| `--building-radius` | Radius for swissBUILDINGS3D query (meters) | Matches `--radius` |
| `--building-layer` | GeoAdmin layer name for swissBUILDINGS3D | `ch.swisstopo.swissbuildings3d_3_0` |
| `--max-buildings` | Maximum number of nearby buildings to include | `250` |
| `--output` | Output IFC file path | `combined_terrain.ifc` |

*Either `--egrid` or both `--center-x` and `--center-y` must be provided.

#### Examples

**Standard combined terrain (500m radius):**
```bash
python combined_terrain.py --egrid CH999979659148 --output combined.ifc
```

**Smaller area for testing:**
```bash
python combined_terrain.py --egrid CH999979659148 --radius 200 --resolution 30 --output test.ifc
```

**High detail with smooth terrain attachment:**
```bash
python combined_terrain.py --egrid CH999979659148 --radius 300 --resolution 5 --densify 1.0 --attach-to-solid --output detailed.ifc
```

**Faster processing with coarser resolution:**
```bash
python combined_terrain.py --egrid CH999979659148 --radius 500 --resolution 20 --densify 2.0 --output fast.ifc
```

**Include nearby swissBUILDINGS3D buildings (matching terrain radius):**
```bash
python combined_terrain.py --egrid CH999979659148 --include-buildings --output combined_with_buildings.ifc
```

**Separated elements (roofs/facades/footprints when Z is available):**
```bash
python combined_terrain.py --egrid CH999979659148 --include-buildings --buildings-separate --output combined_buildings_separated.ifc
```

### Site Boundary Only (workflow.py)

For generating only the site boundary solid without surrounding terrain:

```bash
python workflow.py --egrid CH999979659148 --output CH999979659148.ifc
```

#### Workflow Options

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

## IFC Property Sets and Metadata

The generated IFC files include comprehensive metadata following IFC schema standards:

### Standard Property Sets

#### Pset_LandRegistration
- `LandID`: Parcel number (e.g., "WI3988")
- `LandTitleID`: EGRID identifier (e.g., "CH999979659148")
- `IsPermanentID`: Boolean indicating EGRID is permanent identifier

#### Pset_SiteCommon
- `Reference`: Local identifier from cadastre
- `TotalArea`: Total site area in m²
- `BuildableArea`: Maximum buildable area in m²

#### Qto_SiteBaseQuantities
- `GrossArea`: Total site area in m²
- `GrossPerimeter`: Site perimeter in meters

### Custom Property Set

#### CPset_SwissCadastre
- `GeoportalURL`: Link to canton geoportal
- `Canton`: Canton abbreviation (e.g., "ZH")
- `ParcelNumber`: Parcel number

### IfcSite Attributes

- `LandTitleNumber`: EGRID (official Swiss land registry ID)
- `LongName`: Canton + Parcel number
- `Description`: Human-readable site description

All property sets include proper `OwnerHistory` for IFC viewer compatibility.

## How It Works

### Combined Terrain Workflow

The `combined_terrain.py` script performs the following steps:

1. **Boundary Fetching**
   - Fetches cadastral boundary polygon via geo.admin.ch API
   - Extracts metadata (parcel number, canton, area, perimeter)
   - Calculates site centroid for terrain center

2. **Terrain Grid Creation**
   - Creates circular grid of points around site centroid
   - Grid resolution determines point density
   - Filters points within specified radius

3. **Elevation Sampling**
   - Fetches elevation for terrain grid points
   - Fetches elevation for site boundary points (densified)
   - Progress updates during API calls

4. **Height Offset Calculation**
   - Samples terrain elevations at site boundary
   - Compares with smoothed site elevations
   - Calculates offset to align site solid with terrain

5. **Site Solid Creation**
   - Applies smoothing algorithm (best-fit plane + circular mean filter)
   - Adjusts height using calculated offset
   - Creates closed solid with triangulated top, sides, and bottom

6. **Terrain Mesh with Cutout**
   - Merges site boundary points into terrain grid
   - Triangulates combined point cloud (Delaunay)
   - Excludes triangles whose centroids are inside site boundary
   - Optionally uses smoothed site elevations for boundary points (`--attach-to-solid`)

7. **IFC Generation**
   - Creates georeferenced IFC4 file
   - Sets project origin at site centroid (rounded to 100m)
   - Adds terrain mesh (IfcGeographicElement with SurfaceModel)
   - Adds site solid (IfcGeographicElement with FacetedBrep)
   - Adds site footprint (2D polyline on IfcSite)
   - Maps cadastre metadata to IFC property sets
   - Sets OwnerHistory on all entities

### Site Boundary Only Workflow

The `workflow.py` script creates a standalone site solid:

1. **Boundary Fetching**: Same as combined workflow
2. **3D Draping**: Samples elevations and creates 3D coordinates
3. **Smoothing**: Applies terrain smoothing algorithm
4. **Solid Creation**: Creates closed solid with triangulated top
5. **IFC Export**: Exports site solid with footprint

### Smoothing Algorithm

The terrain smoothing uses a multi-step process:

1. **Best-fit plane**: Calculates plane that best fits elevation data using least squares
2. **Circular mean filter**: Smooths raw elevations (window size: 9)
3. **Residual calculation**: Computes residuals (smoothed - plane)
4. **Residual smoothing**: Applies circular mean filter to residuals
5. **Residual attenuation**: Scales residuals to 20% to reduce bumps
6. **Final elevation**: `plane + 0.2 * smoothed_residuals`

This preserves overall slope while removing small terrain variations.

### Terrain Cutout Algorithm

The precise cutout follows these steps:

1. **Boundary point merging**: Adds site boundary vertices to terrain grid
2. **Triangulation**: Creates Delaunay triangulation of combined points
3. **Triangle exclusion**: Removes triangles whose centroids are inside site polygon
4. **Boundary attachment** (optional): Uses smoothed site elevations for boundary points

This ensures the terrain mesh edges align exactly with the site boundary.

### swissBUILDINGS3D 3.0 Beta buildings

Use `--include-buildings` to request 3D building geometry from the GeoAdmin swissBUILDINGS3D 3.0 Beta layer. Buildings are fetched within the requested radius and filtered to the circular footprint around the site. If the GeoJSON response contains Z coordinates, the script creates a surface model directly from those meshes. Otherwise, it extrudes the building footprint using the nearest sampled terrain elevation and any available height attribute (falling back to 10m). Each building receives a `CPset_SwissBUILDINGS3D` property set with EGID (when present), the source layer, and the height estimate.

Use `--buildings-separate` to export the swissBUILDINGS3D elements as discrete roofs, facades, and footprints (face-based IfcGeographicElement objects) when Z-enabled meshes are returned. This mirrors the swissBUILDINGS3D “separated elements” model; if Z coordinates are missing, the script falls back to footprint extrusion.

## Technical Details

### Coordinate Systems

- **Input/Output CRS**: EPSG:2056 (Swiss LV95 / CH1903+)
- **Vertical Datum**: LN02 (Swiss height system)
- **Units**: Meters (SI)
- **Project Origin**: Centered on site centroid (rounded to nearest 100m)

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

### IFC Structure

**combined_terrain.py:**
```
IfcProject
└── IfcSite
    ├── Representation: FootPrint (2D polyline)
    ├── Property Sets: Pset_LandRegistration, Pset_SiteCommon, CPset_SwissCadastre
    ├── Quantities: Qto_SiteBaseQuantities
    ├── IfcGeographicElement (Surrounding_Terrain)
    │   ├── PredefinedType: TERRAIN
    │   └── Representation: Body (ShellBasedSurfaceModel with cutout)
    └── IfcGeographicElement (Site_Solid)
        ├── PredefinedType: TERRAIN
        └── Representation: Body (FacetedBrep solid)
```

**workflow.py:**
```
IfcProject
└── IfcSite (EGRID)
    ├── Representation: FootPrint (2D polyline)
    └── IfcGeographicElement (Terrain)
        ├── PredefinedType: TERRAIN
        └── Representation: Body (FacetedBrep solid)
```

### Performance Notes

- Each grid point requires one API call to the Swiss height service
- A 500m radius with 10m resolution creates ~2000 points (~3-4 minutes)
- A 500m radius with 20m resolution creates ~500 points (~1 minute)
- Use larger resolution values (15-25m) for faster processing
- Use smaller resolution values (2-5m) for detailed terrain
- The circular area ensures consistent coverage even when site is at terrain edge

## Troubleshooting

### API Rate Limits

If you encounter rate limiting with the elevation API:
- Use a local DEM file (`--dem` for workflow.py)
- Increase resolution (`--resolution 20` or higher)
- Increase densification interval (`--densify 2.0` or higher)
- Process smaller areas (reduce `--radius`)

### Property Sets Not Visible in Viewer

Ensure OwnerHistory is set (automatically handled in `combined_terrain.py`):
- All property sets include OwnerHistory
- All relationships include OwnerHistory
- All entities include OwnerHistory

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

The generated IFC files contain:
- Georeferenced site boundaries
- 3D terrain geometry (solid volumes or surface meshes)
- Proper IFC4 structure for BIM applications
- Comprehensive metadata in standard property sets
- Compatible with major IFC viewers (BlenderBIM, Solibri, etc.)

## License

See LICENSE file for details.

## References

- [geo.admin.ch API Documentation](https://api3.geo.admin.ch/)
- [IFC4 Schema](https://www.buildingsmart.org/standards/bsi-standards/industry-foundation-classes-ifc/)
- [Swiss LV95 Coordinate System](https://www.swisstopo.admin.ch/en/knowledge-facts/surveying-geodesy/reference-systems/switzerland.html)
- [buildingSMART Property Sets](https://www.buildingsmart.org/standards/bsi-standards/ifc-library/)
