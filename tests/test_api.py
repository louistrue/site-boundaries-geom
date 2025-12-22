#!/usr/bin/env python3
"""
Comprehensive test script for the Site Boundaries Terrain API
"""
import requests
import json
import time
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_EGRID = "CH999979659148"  # Example EGRID from the project

def test_health():
    """Test the /health endpoint"""
    print("\n=== Testing /health endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        assert data.get("status") == "ok", f"Expected status 'ok', got {data.get('status')}"
        print("✓ Health check passed")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_generate_immediate():
    """Test the /generate endpoint (immediate generation)"""
    print("\n=== Testing /generate endpoint (immediate) ===")
    try:
        payload = {
            "egrid": TEST_EGRID,
            "radius": 200,  # Smaller radius for faster testing
            "resolution": 20,  # Coarser resolution for faster testing
            "densify": 1.0,
            "output_name": "test_immediate.ifc"
        }
        print(f"  Request payload: {json.dumps(payload, indent=2)}")
        print("  Sending request (this may take a while due to API calls)...")
        
        response = requests.post(
            f"{BASE_URL}/generate",
            json=payload,
            timeout=300  # 5 minute timeout for generation
        )
        response.raise_for_status()
        
        # Check content type
        assert response.headers.get("content-type") == "application/octet-stream", \
            f"Expected octet-stream, got {response.headers.get('content-type')}"
        
        # Check content disposition
        content_disp = response.headers.get("content-disposition", "")
        assert "test_immediate.ifc" in content_disp, \
            f"Expected filename in content-disposition, got {content_disp}"
        
        # Check file size
        content_length = len(response.content)
        assert content_length > 0, "File should not be empty"
        print(f"✓ Generate endpoint passed (file size: {content_length} bytes)")
        
        # Save file for verification
        output_path = Path("results/test_immediate.ifc")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"  Saved to {output_path}")
        
        return True
    except requests.Timeout:
        print("✗ Generate endpoint timed out (this is expected for large requests)")
        return False
    except Exception as e:
        print(f"✗ Generate endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_jobs_create():
    """Test creating a background job"""
    print("\n=== Testing /jobs endpoint (create job) ===")
    try:
        payload = {
            "egrid": TEST_EGRID,
            "radius": 200,
            "resolution": 20,
            "densify": 1.0,
            "output_name": "test_job.ifc"
        }
        print(f"  Request payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/jobs",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        assert "job_id" in data, "Response should contain job_id"
        job_id = data["job_id"]
        print(f"✓ Job created successfully: {job_id}")
        return job_id
    except Exception as e:
        print(f"✗ Job creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_job_status(job_id):
    """Test checking job status"""
    print(f"\n=== Testing /jobs/{job_id} endpoint (status) ===")
    if not job_id:
        print("  Skipping (no job_id)")
        return None
    
    max_wait = 300  # 5 minutes max wait
    start_time = time.time()
    check_interval = 5  # Check every 5 seconds
    
    try:
        while True:
            response = requests.get(
                f"{BASE_URL}/jobs/{job_id}",
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            status = data.get("status")
            print(f"  Job status: {status}")
            
            if status == "completed":
                assert "download_url" in data, "Completed job should have download_url"
                assert "output_name" in data, "Completed job should have output_name"
                print(f"✓ Job completed successfully")
                print(f"  Download URL: {data.get('download_url')}")
                print(f"  Output name: {data.get('output_name')}")
                return data
            elif status == "failed":
                error = data.get("error", "Unknown error")
                print(f"✗ Job failed: {error}")
                return None
            elif status == "running":
                elapsed = time.time() - start_time
                if elapsed > max_wait:
                    print(f"✗ Job timed out after {max_wait} seconds")
                    return None
                print(f"  Waiting... (elapsed: {elapsed:.1f}s)")
                time.sleep(check_interval)
            elif status == "pending":
                elapsed = time.time() - start_time
                if elapsed > max_wait:
                    print(f"✗ Job timed out after {max_wait} seconds")
                    return None
                print(f"  Waiting... (elapsed: {elapsed:.1f}s)")
                time.sleep(check_interval)
            else:
                print(f"  Unknown status: {status}")
                time.sleep(check_interval)
                
    except Exception as e:
        print(f"✗ Job status check failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_job_download(job_id, job_data):
    """Test downloading a completed job"""
    print(f"\n=== Testing /jobs/{job_id}/download endpoint ===")
    if not job_id or not job_data:
        print("  Skipping (no job_id or job not completed)")
        return False
    
    try:
        response = requests.get(
            f"{BASE_URL}/jobs/{job_id}/download",
            timeout=60
        )
        response.raise_for_status()
        
        # Check content type
        assert response.headers.get("content-type") == "application/octet-stream", \
            f"Expected octet-stream, got {response.headers.get('content-type')}"
        
        # Check file size
        content_length = len(response.content)
        assert content_length > 0, "File should not be empty"
        print(f"✓ Download endpoint passed (file size: {content_length} bytes)")
        
        # Save file for verification
        output_name = job_data.get("output_name", "test_download.ifc")
        output_path = Path(f"results/{output_name}")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"  Saved to {output_path}")
        
        # Verify job is marked as expired after download
        status_response = requests.get(f"{BASE_URL}/jobs/{job_id}", timeout=10)
        status_response.raise_for_status()
        status_data = status_response.json()
        if status_data.get("status") == "expired":
            print("✓ Job correctly marked as expired after download")
        
        return True
    except Exception as e:
        print(f"✗ Download endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling"""
    print("\n=== Testing error handling ===")
    all_passed = True
    
    # Test missing egrid
    try:
        response = requests.post(
            f"{BASE_URL}/generate",
            json={"radius": 500},
            timeout=10
        )
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"
        print("✓ Missing egrid validation works")
    except Exception as e:
        print(f"✗ Missing egrid test failed: {e}")
        all_passed = False
    
    # Test invalid job_id
    try:
        response = requests.get(
            f"{BASE_URL}/jobs/invalid-job-id",
            timeout=10
        )
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        print("✓ Invalid job_id returns 404")
    except Exception as e:
        print(f"✗ Invalid job_id test failed: {e}")
        all_passed = False
    
    # Test download before completion
    job_id = None
    try:
        job_id = test_jobs_create()
        if job_id:
            # Try to download immediately (should fail)
            response = requests.get(
                f"{BASE_URL}/jobs/{job_id}/download",
                timeout=10
            )
            assert response.status_code == 409, f"Expected 409, got {response.status_code}"
            print("✓ Download before completion returns 409")
    except Exception as e:
        print(f"✗ Download before completion test failed: {e}")
        all_passed = False
    finally:
        # Cleanup: Wait briefly and try to download if job completes
        # This triggers cleanup via the download endpoint's background task
        # If job doesn't complete quickly, it will be cleaned up by TTL mechanism
        if job_id:
            try:
                # Wait a short time to see if job completes
                time.sleep(2)
                status_response = requests.get(f"{BASE_URL}/jobs/{job_id}", timeout=5)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") == "completed":
                        # Trigger cleanup by downloading
                        requests.get(f"{BASE_URL}/jobs/{job_id}/download", timeout=5)
            except Exception:
                # Ignore cleanup errors - job will be cleaned up by TTL
                pass
    
    return all_passed

def main():
    """Run all tests"""
    print("=" * 60)
    print("Site Boundaries Terrain API - Comprehensive Test Suite")
    print("=" * 60)
    
    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.ConnectionError:
        print(f"\n✗ Cannot connect to server at {BASE_URL}")
        print("  Make sure the server is running: uvicorn src.rest_api:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    results = []
    
    # Test 1: Health check
    results.append(("Health Check", test_health()))
    
    # Test 2: Immediate generation (may take a while)
    results.append(("Immediate Generation", test_generate_immediate()))
    
    # Test 3: Job creation
    job_id = test_jobs_create()
    results.append(("Job Creation", job_id is not None))
    
    # Test 4: Job status
    job_data = test_job_status(job_id)
    results.append(("Job Status", job_data is not None))
    
    # Test 5: Job download
    if job_data:
        results.append(("Job Download", test_job_download(job_id, job_data)))
    
    # Test 6: Error handling
    results.append(("Error Handling", test_error_handling()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()


