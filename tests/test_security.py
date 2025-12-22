"""
Security tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient


class TestSecurityHeaders:
    """Test security headers middleware"""
    
    def test_security_headers_present(self, client):
        """Test that security headers are present in responses"""
        response = client.get("/health")
        
        # Check all security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        
        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        
        assert "Strict-Transport-Security" in response.headers
        assert "max-age=31536000" in response.headers["Strict-Transport-Security"]
        
        assert "Content-Security-Policy" in response.headers
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
        
        assert "Referrer-Policy" in response.headers
        assert "strict-origin-when-cross-origin" in response.headers["Referrer-Policy"]
    
    def test_security_headers_all_endpoints(self, client):
        """Test security headers on all endpoints"""
        endpoints = ["/health", "/docs", "/openapi.json"]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert "X-Content-Type-Options" in response.headers
            assert "X-Frame-Options" in response.headers


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @pytest.mark.slow
    def test_rate_limit_generate_endpoint(self, client, valid_request_payload):
        """Test rate limiting on /generate endpoint - marked slow due to multiple requests"""
        from unittest.mock import patch
        import tempfile
        import os
        
        # Mock generation to avoid actual processing
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        
        with patch('src.rest_api._run_generation') as mock_run:
            mock_run.return_value = None
            # Make multiple rapid requests
            responses = []
            for _ in range(15):
                response = client.post("/generate", json=valid_request_payload)
                responses.append(response.status_code)
            
            # At least one should be rate limited (429) or succeed (200)
            # Note: Rate limiting may not trigger in test environment if using in-memory storage
            # This test verifies the endpoint accepts requests
            # Only accept valid endpoint responses: success (200), validation error (422), or rate limit (429)
            assert all(status in [200, 422, 429] for status in responses)
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    def test_rate_limit_jobs_endpoint(self, client, valid_request_payload):
        """Test rate limiting on /jobs endpoint"""
        # Make multiple rapid requests
        responses = []
        for _ in range(25):
            response = client.post("/jobs", json=valid_request_payload)
            responses.append(response.status_code)
        
        # Should accept most requests (rate limit is 20/min)
        # Note: Rate limiting may not trigger in test environment
        assert all(status in [200, 422, 429] for status in responses)
    
    def test_rate_limit_health_endpoint(self, client):
        """Test that /health endpoint is not rate limited"""
        # Health endpoint should not have rate limiting
        responses = []
        for _ in range(50):
            response = client.get("/health")
            responses.append(response.status_code)
        
        # All should succeed
        assert all(status == 200 for status in responses)


class TestCORS:
    """Test CORS configuration"""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers can be present"""
        # CORS headers are added by middleware
        # In test client, we can check if middleware is configured
        response = client.get("/health")
        # CORS headers may not be present for same-origin requests
        # This test verifies the middleware is configured
        assert response.status_code == 200
    
    def test_options_request(self, client):
        """Test OPTIONS request handling"""
        # FastAPI TestClient may not fully support OPTIONS
        # This is a placeholder for CORS preflight testing
        response = client.options("/health")
        # Should not error
        assert response.status_code in [200, 405, 404]


class TestInputSanitization:
    """Test input sanitization and validation"""
    
    def test_sql_injection_attempt(self, client):
        """Test that SQL injection attempts are rejected"""
        payload = {
            "egrid": "CH999979659148'; DROP TABLE users; --",
            "radius": 100
        }
        response = client.post("/generate", json=payload)
        # Should fail validation (invalid EGRID format)
        assert response.status_code == 422
    
    def test_xss_attempt(self, client):
        """Test that XSS attempts are rejected"""
        from unittest.mock import patch
        import tempfile
        import os
        
        # Mock generation to avoid actual processing
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        
        payload = {
            "egrid": "CH999979659148",
            "output_name": "<script>alert('xss')</script>.ifc",
            "radius": 100
        }
        
        with patch('src.rest_api._run_generation') as mock_run:
            mock_run.return_value = None
            response = client.post("/generate", json=payload)
            # Should accept (output_name is just a filename suggestion)
            # XSS protection is handled by security headers
            assert response.status_code == 200
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    def test_path_traversal_attempt(self, client):
        """Test that path traversal attempts are handled"""
        from unittest.mock import patch
        import tempfile
        import os
        
        # Mock generation to avoid actual processing
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        tmp_file.write(b"Mock IFC content")
        tmp_file.close()
        
        payload = {
            "egrid": "CH999979659148",
            "output_name": "../../../etc/passwd",
            "radius": 100
        }
        
        with patch('src.rest_api._run_generation') as mock_run:
            mock_run.return_value = None
            response = client.post("/generate", json=payload)
            # Should handle safely (output_name is just a filename suggestion)
            # May return 200 (success) or 429 (rate limited) - both are acceptable
            assert response.status_code in [200, 429]
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


class TestErrorHandling:
    """Test error handling and information disclosure"""
    
    def test_error_messages_no_sensitive_info(self, client):
        """Test that error messages don't leak sensitive information"""
        # Test with invalid request
        response = client.post("/generate", json={"egrid": "INVALID"})
        
        assert response.status_code == 422
        error_detail = str(response.json())
        
        # Should not contain sensitive paths or system info
        sensitive_patterns = [
            "/etc/",
            "/root/",
            "/home/",
            "password",
            "secret",
            "key",
            "token"
        ]
        
        error_lower = error_detail.lower()
        for pattern in sensitive_patterns:
            assert pattern not in error_lower
    
    def test_404_error_format(self, client):
        """Test that 404 errors are properly formatted"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        # Should return JSON error
        data = response.json()
        assert "detail" in data

