#!/bin/bash
# Test runner script for the Site Boundaries Terrain API

set -e

echo "=========================================="
echo "Site Boundaries Terrain API - Test Suite"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install dependencies: pip install -r requirements.txt"
    exit 1
fi

# Parse command line arguments
TEST_TYPE="${1:-all}"
COVERAGE="${2:-true}"

# Initialize exit code
EXIT_CODE=0

case "$TEST_TYPE" in
    unit)
        echo -e "${YELLOW}Running unit tests...${NC}"
        pytest tests/test_validation.py -v -m "not slow" || EXIT_CODE=$?
        ;;
    integration)
        echo -e "${YELLOW}Running integration tests...${NC}"
        pytest tests/test_endpoints.py -v -m "not slow" || EXIT_CODE=$?
        ;;
    security)
        echo -e "${YELLOW}Running security tests...${NC}"
        pytest tests/test_security.py -v || EXIT_CODE=$?
        ;;
    error)
        echo -e "${YELLOW}Running error handling tests...${NC}"
        pytest tests/test_error_handling.py -v -m "not slow" || EXIT_CODE=$?
        ;;
    fast)
        echo -e "${YELLOW}Running fast tests (excluding slow tests)...${NC}"
        pytest tests/ -v -m "not slow" || EXIT_CODE=$?
        ;;
    all)
        echo -e "${YELLOW}Running all tests...${NC}"
        if [ "$COVERAGE" = "true" ]; then
            pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html:build/htmlcov || EXIT_CODE=$?
            echo ""
            echo -e "${GREEN}Coverage report generated in build/htmlcov/index.html${NC}"
        else
            pytest tests/ -v || EXIT_CODE=$?
        fi
        ;;
    *)
        echo "Usage: $0 [unit|integration|security|error|fast|all] [coverage=true|false]"
        echo ""
        echo "Examples:"
        echo "  $0 all              # Run all tests with coverage"
        echo "  $0 fast             # Run fast tests only"
        echo "  $0 unit             # Run unit tests only"
        echo "  $0 security         # Run security tests only"
        echo "  $0 all false        # Run all tests without coverage"
        exit 1
        ;;
esac

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi

exit $EXIT_CODE

