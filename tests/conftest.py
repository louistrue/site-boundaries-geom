"""
Pytest configuration and shared fixtures
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Set test environment variables before importing app
os.environ["ENABLE_DOCS"] = "true"
os.environ["ALLOWED_ORIGINS"] = "*"
os.environ["ALLOWED_HOSTS"] = "*"
os.environ["TMPDIR"] = str(Path(tempfile.gettempdir()) / "test_ifc_files")

# Create test temp directory
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

from src.rest_api import app


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def valid_egrid():
    """Valid Swiss EGRID for testing"""
    return "CH999979659148"


@pytest.fixture
def valid_request_payload(valid_egrid):
    """Valid request payload for testing"""
    return {
        "egrid": valid_egrid,
        "radius": 100.0,
        "resolution": 20.0,
        "densify": 1.0,
        "include_terrain": True,
        "include_site_solid": True,
        "output_name": "test.ifc"
    }


@pytest.fixture
def mock_boundary_response():
    """Mock response from geo.admin.ch boundary API"""
    return {
        "results": [{
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [2675000, 1245000],
                    [2675100, 1245000],
                    [2675100, 1245100],
                    [2675000, 1245100],
                    [2675000, 1245000]
                ]]
            },
            "properties": {
                "ak": "ZH",
                "number": "1234",
                "identnd": "LOCAL123",
                "geoportal_url": "https://example.com",
                "realestate_type": "parcel"
            }
        }]
    }


@pytest.fixture
def mock_elevation_response():
    """Mock response from geo.admin.ch elevation API"""
    return {"height": 500.0}


@pytest.fixture
def mock_combined_terrain_success():
    """Mock successful combined terrain generation"""
    def _mock_run(*args, **kwargs):
        # Create a dummy IFC file
        output_path = kwargs.get('output_path') or args[-1]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"Dummy IFC content")
        return output_path
    return _mock_run

