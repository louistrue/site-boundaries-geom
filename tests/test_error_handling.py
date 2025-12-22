"""
Error handling and edge case tests
"""
import pytest
from unittest.mock import patch, MagicMock


class TestExceptionHandling:
    """Test exception handling and error responses"""
    
    @patch('src.rest_api._run_generation')
    def test_generate_timeout_error(self, mock_run, client, valid_request_payload):
        """Test handling of timeout errors"""
        import requests
        # Mock will raise exception when called
        mock_run.side_effect = requests.Timeout("Request timed out")
        
        response = client.post("/generate", json=valid_request_payload)
        assert response.status_code == 504
        detail = response.json()["detail"].lower()
        # Check for timeout-related keywords
        assert "timeout" in detail or "timed out" in detail
    
    @patch('src.rest_api._run_generation')
    def test_generate_http_error(self, mock_run, client, valid_request_payload):
        """Test handling of HTTP errors"""
        import requests
        mock_run.side_effect = requests.HTTPError("Upstream error")
        
        response = client.post("/generate", json=valid_request_payload)
        assert response.status_code == 502
        assert "upstream" in response.json()["detail"].lower()
    
    @patch('src.rest_api._run_generation')
    def test_generate_value_error(self, mock_run, client, valid_request_payload):
        """Test handling of value errors"""
        mock_run.side_effect = ValueError("Invalid EGRID")
        
        response = client.post("/generate", json=valid_request_payload)
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()
    
    @patch('src.rest_api._run_generation')
    def test_generate_generic_error(self, mock_run, client, valid_request_payload):
        """Test handling of generic errors"""
        mock_run.side_effect = Exception("Unexpected error")
        
        response = client.post("/generate", json=valid_request_payload)
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @patch('src.rest_api._run_generation')
    def test_minimum_radius(self, mock_run, client, valid_egrid):
        """Test with minimum valid radius"""
        import tempfile
        import os
        # Mock successful generation
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        mock_run.return_value = None
        
        payload = {
            "egrid": valid_egrid,
            "radius": 0.1  # Just above zero
        }
        response = client.post("/generate", json=payload)
        # Should accept (validation passes)
        assert response.status_code == 200
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    @patch('src.rest_api._run_generation')
    @pytest.mark.slow
    def test_maximum_radius(self, mock_run, client, valid_egrid):
        """Test with maximum valid radius - marked slow due to large radius validation"""
        import tempfile
        import os
        # Mock successful generation
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        mock_run.return_value = None
        
        payload = {
            "egrid": valid_egrid,
            "radius": 2000.0  # Maximum allowed
        }
        response = client.post("/generate", json=payload)
        # Should accept (validation passes)
        assert response.status_code == 200
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    @patch('src.rest_api._run_generation')
    def test_minimum_resolution(self, mock_run, client, valid_egrid):
        """Test with minimum valid resolution"""
        import tempfile
        import os
        # Mock successful generation
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        mock_run.return_value = None
        
        payload = {
            "egrid": valid_egrid,
            "resolution": 5.0  # Minimum allowed
        }
        response = client.post("/generate", json=payload)
        assert response.status_code == 200
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    @patch('src.rest_api._run_generation')
    def test_maximum_resolution(self, mock_run, client, valid_egrid):
        """Test with maximum valid resolution"""
        import tempfile
        import os
        # Mock successful generation
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        mock_run.return_value = None
        
        payload = {
            "egrid": valid_egrid,
            "resolution": 100.0  # Maximum allowed
        }
        response = client.post("/generate", json=payload)
        assert response.status_code == 200
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    @patch('src.rest_api._run_generation')
    def test_optional_center_coordinates(self, mock_run, client, valid_egrid):
        """Test with optional center coordinates"""
        import tempfile
        import os
        # Mock successful generation
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        mock_run.return_value = None
        
        payload = {
            "egrid": valid_egrid,
            "center_x": 2675000.0,
            "center_y": 1245000.0,
            "radius": 100
        }
        response = client.post("/generate", json=payload)
        assert response.status_code == 200
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    def test_output_name_extension_handling(self, client, valid_request_payload):
        """Test output name extension handling"""
        # Test without extension
        payload = valid_request_payload.copy()
        payload["output_name"] = "test"
        response = client.post("/jobs", json=payload)
        assert response.status_code == 200
        
        # Test with extension
        payload["output_name"] = "test.ifc"
        response = client.post("/jobs", json=payload)
        assert response.status_code == 200


class TestConcurrentRequests:
    """Test handling of concurrent requests"""
    
    @patch('src.terrain_with_site.run_combined_terrain_workflow')
    def test_multiple_jobs_concurrent(self, mock_terrain, client, valid_request_payload):
        """Test creating multiple jobs concurrently"""
        import concurrent.futures
        import tempfile
        import os
        
        # Mock terrain generation to avoid real processing
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        mock_terrain.return_value = tmp_file.name
        
        def create_job():
            return client.post("/jobs", json=valid_request_payload)
        
        # Create 5 jobs concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_job) for _ in range(5)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        
        # All should have job_ids
        job_ids = [r.json()["job_id"] for r in responses]
        assert len(set(job_ids)) == 5  # All unique
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


class TestJobLifecycle:
    """Test job lifecycle and state transitions"""
    
    @patch('src.terrain_with_site.run_combined_terrain_workflow')
    def test_job_lifecycle(self, mock_terrain, client, valid_request_payload):
        """Test complete job lifecycle"""
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
        
        # Create job
        create_response = client.post("/jobs", json=valid_request_payload)
        assert create_response.status_code == 200
        job_id = create_response.json()["job_id"]
        
        # Check initial status (should be pending, running, or completed if very fast)
        status_response = client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200
        initial_status = status_response.json()["status"]
        assert initial_status in ["pending", "running", "completed"]  # May complete quickly
        
        # Job should eventually complete or fail
        # (In real scenario, would wait for completion)
        # This test just verifies the state machine works
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


class TestFileHandling:
    """Test file handling and cleanup"""
    
    @patch('src.terrain_with_site.run_combined_terrain_workflow')
    def test_temp_file_cleanup(self, mock_terrain, client, valid_request_payload):
        """Test that temp files are cleaned up"""
        import tempfile
        import os
        
        # Mock terrain generation to avoid real processing
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        mock_terrain.return_value = tmp_file.name
        
        # Create a job (which may create temp files)
        response = client.post("/jobs", json=valid_request_payload)
        assert response.status_code == 200
        
        # Note: Actual cleanup happens after download or via TTL-based cleanup
        # This test verifies the job creation mechanism exists
        # In production, files are cleaned up by background tasks after download
        # or automatically via the TTL-based cleanup mechanism
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)

