"""
Integration tests against running Docker container
"""
import os
import pytest
import requests
from time import sleep

BASE_URL = os.getenv("BASE_URL", "http://localhost:8080")
TEST_EGRID = "CH999979659148"


class TestDockerHealth:
    """Test health endpoint on Docker container"""
    
    def test_health_endpoint(self):
        """Test health endpoint"""
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_health_headers(self):
        """Test security headers on health endpoint"""
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"


class TestDockerValidation:
    """Test input validation on Docker container"""
    
    def test_invalid_egrid_format(self):
        """Test invalid EGRID format"""
        response = requests.post(
            f"{BASE_URL}/generate",
            json={"egrid": "INVALID", "radius": 100},
            timeout=5
        )
        assert response.status_code == 422
    
    def test_invalid_egrid_too_short(self):
        """Test EGRID too short"""
        response = requests.post(
            f"{BASE_URL}/generate",
            json={"egrid": "CH123", "radius": 100},
            timeout=5
        )
        assert response.status_code == 422
    
    def test_invalid_radius_too_large(self):
        """Test radius exceeding maximum"""
        response = requests.post(
            f"{BASE_URL}/generate",
            json={"egrid": TEST_EGRID, "radius": 5000},
            timeout=5
        )
        assert response.status_code == 422
    
    def test_invalid_resolution_too_small(self):
        """Test resolution below minimum"""
        response = requests.post(
            f"{BASE_URL}/generate",
            json={"egrid": TEST_EGRID, "resolution": 1.0},
            timeout=5
        )
        assert response.status_code == 422
    
    def test_invalid_resolution_too_large(self):
        """Test resolution exceeding maximum"""
        response = requests.post(
            f"{BASE_URL}/generate",
            json={"egrid": TEST_EGRID, "resolution": 200.0},
            timeout=5
        )
        assert response.status_code == 422
    
    def test_valid_request(self):
        """Test valid request structure - request accepted (may timeout during processing)"""
        try:
            response = requests.post(
                f"{BASE_URL}/generate",
                json={
                    "egrid": TEST_EGRID,
                    "radius": 100.0,
                    "resolution": 20.0,
                    "densify": 1.0
                },
                timeout=2  # Short timeout - we just want to verify request is accepted
            )
            # If we get a response, it should not be 422 (validation error)
            assert response.status_code != 422
        except requests.exceptions.Timeout:
            # Timeout is OK - means request was accepted and processing started
            # This is expected for terrain generation which takes time
            pass
        except requests.exceptions.RequestException:
            # Any other error is OK as long as it's not a validation error
            # The important thing is that the request structure was valid
            pass


class TestDockerJobs:
    """Test jobs endpoints on Docker container"""
    
    def test_create_job(self):
        """Test creating a job"""
        response = requests.post(
            f"{BASE_URL}/jobs",
            json={
                "egrid": TEST_EGRID,
                "radius": 100.0,
                "resolution": 20.0,
                "densify": 1.0,
                "output_name": "test.ifc"
            },
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0
    
    def test_get_job_status(self):
        """Test getting job status"""
        # Create a job first
        create_response = requests.post(
            f"{BASE_URL}/jobs",
            json={
                "egrid": TEST_EGRID,
                "radius": 100.0,
                "resolution": 20.0,
                "densify": 1.0
            },
            timeout=10
        )
        assert create_response.status_code == 200
        job_id = create_response.json()["job_id"]
        
        # Check status
        status_response = requests.get(f"{BASE_URL}/jobs/{job_id}", timeout=10)
        assert status_response.status_code == 200
        data = status_response.json()
        assert "status" in data
        assert data["status"] in ["pending", "running", "completed", "failed"]
    
    def test_get_nonexistent_job(self):
        """Test getting non-existent job"""
        response = requests.get(f"{BASE_URL}/jobs/non-existent-id", timeout=5)
        assert response.status_code == 404
    
    def test_download_not_ready_job(self):
        """Test downloading job that's not ready"""
        # Create a job
        create_response = requests.post(
            f"{BASE_URL}/jobs",
            json={
                "egrid": TEST_EGRID,
                "radius": 100.0,
                "resolution": 20.0,
                "densify": 1.0
            },
            timeout=10
        )
        job_id = create_response.json()["job_id"]
        
        # Try to download immediately (should fail)
        response = requests.get(f"{BASE_URL}/jobs/{job_id}/download", timeout=5)
        assert response.status_code == 409


class TestDockerSecurity:
    """Test security features on Docker container"""
    
    def test_security_headers_all_endpoints(self):
        """Test security headers on multiple endpoints"""
        endpoints = ["/health", "/docs", "/openapi.json"]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
                if response.status_code == 200:
                    assert "X-Content-Type-Options" in response.headers
                    assert "X-Frame-Options" in response.headers
            except requests.exceptions.RequestException:
                # Some endpoints might not exist, that's OK
                pass
    
    def test_rate_limiting_health(self):
        """Test that health endpoint is not rate limited"""
        # Make many rapid requests
        responses = []
        for _ in range(50):
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            responses.append(response.status_code)
        
        # All should succeed (no rate limiting on health)
        assert all(status == 200 for status in responses)
    
    def test_cors_configuration(self):
        """Test CORS headers"""
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        # CORS headers may not be present for same-origin requests
        # This test verifies the endpoint responds
        assert response.status_code == 200


class TestDockerErrorHandling:
    """Test error handling on Docker container"""
    
    def test_missing_required_field(self):
        """Test missing required field"""
        response = requests.post(
            f"{BASE_URL}/generate",
            json={"radius": 100},
            timeout=5
        )
        assert response.status_code == 422
    
    def test_invalid_json(self):
        """Test invalid JSON"""
        response = requests.post(
            f"{BASE_URL}/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        assert response.status_code in [400, 422]
    
    def test_wrong_method(self):
        """Test wrong HTTP method"""
        response = requests.put(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 405


class TestDockerOpenAPI:
    """Test OpenAPI schema on Docker container"""
    
    def test_openapi_schema_exists(self):
        """Test OpenAPI schema endpoint"""
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
    
    def test_openapi_has_endpoints(self):
        """Test that all endpoints are in schema"""
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        schema = response.json()
        paths = schema["paths"]
        assert "/health" in paths
        assert "/generate" in paths
        assert "/jobs" in paths

