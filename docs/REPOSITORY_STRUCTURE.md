# Repository Structure

This document describes the organization of the Site Boundaries Geometry repository.

## Directory Structure

```
site-boundaries-geom/
├── src/                    # Source code modules
│   ├── __init__.py        # Package initialization
│   ├── rest_api.py        # FastAPI REST API application
│   ├── terrain_with_site.py  # Combined terrain with site solid workflow
│   ├── surrounding_terrain.py  # Surrounding terrain mesh only
│   └── site_solid.py      # Site boundary solid only
│
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── conftest.py        # Pytest configuration and fixtures
│   ├── test_api.py        # API integration tests
│   ├── test_docker_integration.py
│   ├── test_endpoints.py  # Endpoint tests
│   ├── test_error_handling.py
│   ├── test_security.py   # Security tests
│   ├── test_validation.py # Validation tests
│   └── README.md          # Test documentation
│
├── scripts/                # Utility scripts
│   ├── run_tests.sh       # Test runner script
│   └── test_docker_security.sh
│
├── docs/                   # Documentation
│   ├── DEPLOYMENT.md      # Deployment guide
│   └── REPOSITORY_STRUCTURE.md  # This file
│
├── build/                  # Build outputs (gitignored)
│   ├── coverage.xml       # Coverage report (XML)
│   └── htmlcov/           # Coverage report (HTML)
│
├── output/                 # Runtime outputs (gitignored)
│   └── *.ifc              # Generated IFC files
│
├── results/                # Test results (gitignored)
│
├── .coveragerc             # Coverage configuration
├── .dockerignore           # Docker build exclusions
├── .gitignore             # Git exclusions
├── Dockerfile             # Docker build configuration
├── LICENSE                # License file
├── pytest.ini             # Pytest configuration
├── README.md              # Main documentation
└── requirements.txt       # Python dependencies
```

## Key Changes from Previous Structure

### Source Code Organization
- **Before**: Source files (`rest_api.py`, `terrain_with_site.py`, etc.) were in the root directory
- **After**: All source code is now in the `src/` directory as a proper Python package

### Test Organization
- **Before**: `test_api.py` was in the root directory
- **After**: All test files are consolidated in the `tests/` directory

### Scripts
- **Before**: Utility scripts (`run_tests.sh`, `test_docker_security.sh`) were in the root
- **After**: All scripts are organized in the `scripts/` directory

### Documentation
- **Before**: `DEPLOYMENT.md` was in the root directory
- **After**: Documentation is organized in the `docs/` directory

### Build Outputs
- **Before**: Coverage reports (`coverage.xml`, `htmlcov/`) were in the root
- **After**: All build outputs are in the `build/` directory

### Runtime Outputs
- **Before**: Generated IFC files were in the root directory
- **After**: All runtime outputs are in the `output/` directory

## Import Paths

### Source Code Imports
All source code imports now use the `src` package prefix:

```python
# In src/rest_api.py
from . import terrain_with_site

# In tests
from src.rest_api import app, GenerateRequest
from src.terrain_with_site import run_combined_terrain_workflow
```

### Running Scripts

Scripts can be run as modules:

```bash
# Combined terrain workflow
python -m src.terrain_with_site --egrid CH999979659148 --radius 500

# Site boundary solid only
python -m src.site_solid --egrid CH999979659148

# Surrounding terrain only
python -m src.surrounding_terrain --egrid CH999979659148 --radius 500

# API server
uvicorn src.rest_api:app --host 0.0.0.0 --port 8000
```

Or with PYTHONPATH set:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python src/combined_terrain.py --egrid CH999979659148
```

## Configuration Updates

### Dockerfile
- Updated to copy `src/` directory instead of individual files
- Updated CMD to use `src.rest_api:app`
- Added `PYTHONPATH=/app` environment variable

### pytest.ini
- Updated coverage paths to `src`
- Updated coverage report paths to `build/htmlcov` and `build/coverage.xml`

### .coveragerc
- Updated source path to `src`

### .gitignore
- Updated to ignore `build/` and `output/` directories
- Coverage reports now go to `build/` instead of root

## Benefits of This Structure

1. **Clear Separation**: Source code, tests, scripts, and outputs are clearly separated
2. **Scalability**: Easy to add new modules without cluttering the root directory
3. **Professional**: Follows Python packaging best practices
4. **Maintainability**: Easier to find and organize files
5. **Clean Root**: Root directory only contains essential configuration files

## Migration Notes

If you have existing scripts or workflows that reference the old structure:

1. Update import statements to use `src.` prefix
2. Update command-line invocations to use `python -m src.module` syntax
3. Update any hardcoded paths to generated files (now in `build/` or `output/`)
4. Update Docker builds to reference `src/` directory

