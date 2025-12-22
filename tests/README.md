# Test Suite Documentation

This directory contains the comprehensive test suite for the Site Boundaries Terrain API.

## Test Structure

```
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Pytest fixtures and configuration
├── test_validation.py        # Unit tests for input validation
├── test_endpoints.py        # Integration tests for API endpoints
├── test_security.py         # Security tests (headers, rate limiting, CORS)
└── test_error_handling.py   # Error handling and edge case tests
```

## Running Tests

### Quick Start

```bash
# Run all tests with coverage
./run_tests.sh

# Run fast tests only (excludes slow tests)
./run_tests.sh fast

# Run specific test categories
./run_tests.sh unit          # Unit tests only
./run_tests.sh integration   # Integration tests only
./run_tests.sh security      # Security tests only
./run_tests.sh error         # Error handling tests only
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_validation.py

# Run specific test class
pytest tests/test_validation.py::TestEGRIDValidation

# Run specific test function
pytest tests/test_validation.py::TestEGRIDValidation::test_valid_egrid

# Run with coverage
pytest --cov=api --cov=combined_terrain --cov-report=html

# Run only fast tests (exclude slow)
pytest -m "not slow"
```

## Test Categories

### Unit Tests (`test_validation.py`)

Tests for input validation and data models:
- EGRID format validation
- Radius bounds validation
- Resolution bounds validation
- Densify bounds validation
- Component inclusion validation
- Output name handling

### Integration Tests (`test_endpoints.py`)

Tests for API endpoints using FastAPI TestClient:
- Health check endpoint
- Generate endpoint (synchronous)
- Jobs endpoints (asynchronous)
- OpenAPI schema generation

### Security Tests (`test_security.py`)

Tests for security features:
- Security headers (X-Content-Type-Options, X-Frame-Options, etc.)
- Rate limiting functionality
- CORS configuration
- Input sanitization
- Error message security (no sensitive info leakage)

### Error Handling Tests (`test_error_handling.py`)

Tests for error scenarios:
- Exception handling and error responses
- Edge cases and boundary conditions
- Concurrent request handling
- Job lifecycle and state transitions
- File handling and cleanup

## Test Fixtures

Common fixtures defined in `conftest.py`:

- `client`: FastAPI TestClient instance
- `valid_egrid`: Valid Swiss EGRID for testing
- `valid_request_payload`: Valid request payload
- `mock_boundary_response`: Mock geo.admin.ch boundary API response
- `mock_elevation_response`: Mock geo.admin.ch elevation API response
- `mock_combined_terrain_success`: Mock successful terrain generation

## Mocking External APIs

Tests use mocks to avoid hitting real external APIs:
- `geo.admin.ch` boundary API calls are mocked
- `geo.admin.ch` elevation API calls are mocked
- Terrain generation is mocked for fast test execution

## Coverage

The test suite aims for:
- **Minimum coverage**: 70% (configured in `pytest.ini`)
- **Target coverage**: 80%+

View coverage report:
```bash
pytest --cov=api --cov=combined_terrain --cov-report=html
open htmlcov/index.html
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=api --cov-report=xml
```

## Writing New Tests

When adding new features:

1. **Add unit tests** for validation logic
2. **Add integration tests** for new endpoints
3. **Add security tests** for security-sensitive features
4. **Add error handling tests** for edge cases

Example test structure:

```python
class TestNewFeature:
    """Test new feature"""
    
    def test_feature_success(self, client):
        """Test successful feature usage"""
        response = client.post("/new-endpoint", json={"param": "value"})
        assert response.status_code == 200
    
    def test_feature_validation(self, client):
        """Test feature validation"""
        response = client.post("/new-endpoint", json={"param": "invalid"})
        assert response.status_code == 422
```

## Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.slow
def test_slow_operation():
    """This test takes a long time"""
    pass

@pytest.mark.integration
def test_api_integration():
    """Integration test"""
    pass
```

Run tests by marker:
```bash
pytest -m "not slow"        # Exclude slow tests
pytest -m "integration"     # Only integration tests
```

## Troubleshooting

### Tests fail with import errors

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### Tests fail with connection errors

Some tests mock external APIs. If you see connection errors, check that mocks are properly configured.

### Coverage is low

Add tests for uncovered code paths. Use `--cov-report=html` to see which lines are not covered.

