"""
Integration tests for API endpoints
"""
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.rest_api import app, jobs


class TestHealthEndpoint:
    """Test /health endpoint"""
    
    def test_health_check(self, client):
        """Test health endpoint returns OK"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_health_check_headers(self, client):
        """Test health endpoint includes security headers"""
        response = client.get("/health")
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"


class TestGenerateEndpoint:
    """Test /generate endpoint"""
    
    @patch('src.rest_api._run_generation')
    def test_generate_success(self, mock_run, client, valid_request_payload):
        """Test successful generation"""
        # Create a temporary file for the mock
        import tempfile
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        
        # Mock the generation function
        mock_run.return_value = None  # Async function returns None
        
        # Mock the actual terrain generation
        with patch('src.terrain_with_site.run_combined_terrain_workflow') as mock_terrain:
            mock_terrain.return_value = tmp_file.name
            
            response = client.post("/generate", json=valid_request_payload)
            
            # Should return 200 with file content
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/octet-stream"
            assert "test.ifc" in response.headers.get("content-disposition", "")
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    def test_generate_missing_egrid(self, client):
        """Test generate with missing EGRID"""
        payload = {"radius": 100}
        response = client.post("/generate", json=payload)
        assert response.status_code == 422
    
    def test_generate_invalid_egrid(self, client):
        """Test generate with invalid EGRID format"""
        payload = {
            "egrid": "INVALID",
            "radius": 100
        }
        response = client.post("/generate", json=payload)
        assert response.status_code == 422
    
    def test_generate_invalid_radius(self, client, valid_egrid):
        """Test generate with invalid radius"""
        payload = {
            "egrid": valid_egrid,
            "radius": 5000  # Exceeds maximum
        }
        response = client.post("/generate", json=payload)
        assert response.status_code == 422
    
    def test_generate_invalid_resolution(self, client, valid_egrid):
        """Test generate with invalid resolution"""
        payload = {
            "egrid": valid_egrid,
            "resolution": 1.0  # Below minimum
        }
        response = client.post("/generate", json=payload)
        assert response.status_code == 422


class TestJobsEndpoint:
    """Test /jobs endpoints"""
    
    @patch('src.terrain_with_site.run_combined_terrain_workflow')
    def test_create_job(self, mock_terrain, client, valid_request_payload):
        """Test creating a job"""
        import tempfile
        import os
        
        # Mock terrain generation to avoid real processing
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        mock_terrain.return_value = tmp_file.name
        
        response = client.post("/jobs", json=valid_request_payload)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    def test_create_job_invalid_payload(self, client):
        """Test creating job with invalid payload"""
        response = client.post("/jobs", json={"egrid": "INVALID"})
        assert response.status_code == 422
    
    @patch('src.terrain_with_site.run_combined_terrain_workflow')
    def test_get_job_status_pending(self, mock_terrain, client, valid_request_payload):
        """Test getting status of pending job"""
        import tempfile
        import os
        import time
        
        # Mock terrain generation to avoid real processing (but add delay to simulate processing)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        
        def delayed_return(*args, **kwargs):
            time.sleep(0.1)  # Small delay to allow status check
            return tmp_file.name
        
        mock_terrain.side_effect = delayed_return
        
        # Create a job
        create_response = client.post("/jobs", json=valid_request_payload)
        job_id = create_response.json()["job_id"]
        
        # Check status immediately (should be pending or running)
        status_response = client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200
        data = status_response.json()
        # Status may be pending, running, or completed (if mock completes very quickly)
        assert data["status"] in ["pending", "running", "completed"]
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    def test_get_job_status_not_found(self, client):
        """Test getting status of non-existent job"""
        response = client.get("/jobs/non-existent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @patch('src.terrain_with_site.run_combined_terrain_workflow')
    def test_download_job_not_ready(self, mock_terrain, client, valid_request_payload):
        """Test downloading job that's not ready"""
        import tempfile
        import os
        import time
        
        # Mock terrain generation with delay to ensure job is still pending
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        
        def delayed_return(*args, **kwargs):
            time.sleep(0.2)  # Delay to allow status check
            return tmp_file.name
        
        mock_terrain.side_effect = delayed_return
        
        # Create a job
        create_response = client.post("/jobs", json=valid_request_payload)
        job_id = create_response.json()["job_id"]
        
        # Try to download immediately (should fail - job not ready yet)
        response = client.get(f"/jobs/{job_id}/download")
        assert response.status_code == 409
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    def test_download_job_not_found(self, client):
        """Test downloading non-existent job"""
        response = client.get("/jobs/non-existent-id/download")
        assert response.status_code == 404


class TestOpenAPISchema:
    """Test OpenAPI schema generation"""
    
    def test_openapi_schema_exists(self, client):
        """Test that OpenAPI schema is generated"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
    
    def test_openapi_schema_has_endpoints(self, client):
        """Test that all endpoints are in schema"""
        response = client.get("/openapi.json")
        schema = response.json()
        paths = schema["paths"]
        assert "/health" in paths
        assert "/generate" in paths
        assert "/jobs" in paths
    
    def test_docs_endpoint(self, client):
        """Test that docs endpoint exists when enabled"""
        response = client.get("/docs")
        # Should redirect or return HTML
        assert response.status_code in [200, 307]

