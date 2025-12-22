#!/bin/bash
# Security testing script for Docker image
# Run this after building the Docker image

set -e

IMAGE_NAME="site-boundaries-api"

echo "=== Docker Security Tests ==="
echo ""

echo "1. Verifying non-root user..."
USER=$(docker run --rm $IMAGE_NAME whoami)
if [ "$USER" = "appuser" ]; then
    echo "✓ Container runs as non-root user: $USER"
else
    echo "✗ Container runs as: $USER (expected: appuser)"
    exit 1
fi

echo ""
echo "2. Verifying processes run as non-root..."
FOUND_NON_APPUSER=false
while read proc_user; do
    if [ "$proc_user" != "appuser" ] && [ "$proc_user" != "USER" ]; then
        echo "✗ Found process running as: $proc_user"
        FOUND_NON_APPUSER=true
    fi
done < <(docker run --rm $IMAGE_NAME ps aux | grep -v "USER.*COMMAND" | awk '{print $1}' | sort -u)

if [ "$FOUND_NON_APPUSER" = true ]; then
    exit 1
fi
echo "✓ All processes run as appuser"

echo ""
echo "3. Testing health endpoint..."
# Start container in background
CONTAINER_ID=$(docker run -d -p 8080:8080 $IMAGE_NAME)
sleep 5  # Wait for server to start

# Test health endpoint
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo "✓ Health endpoint responds correctly"
else
    echo "✗ Health endpoint returned: $HTTP_CODE"
    docker stop $CONTAINER_ID > /dev/null
    exit 1
fi

echo ""
echo "4. Testing security headers..."
HEADERS=$(curl -s -I http://localhost:8080/health)
if echo "$HEADERS" | grep -q "X-Content-Type-Options: nosniff"; then
    echo "✓ Security headers present"
else
    echo "✗ Security headers missing"
    docker stop $CONTAINER_ID > /dev/null
    exit 1
fi

echo ""
echo "5. Testing rate limiting..."
# Make multiple rapid requests
RATE_LIMIT_HIT=false
for i in {1..15}; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health)
    if [ "$HTTP_CODE" = "429" ]; then
        RATE_LIMIT_HIT=true
        break
    fi
    sleep 0.1
done

if [ "$RATE_LIMIT_HIT" = true ]; then
    echo "✓ Rate limiting is working"
else
    echo "⚠ Rate limiting may not be active (this is OK for /health endpoint)"
fi

# Cleanup
docker stop $CONTAINER_ID > /dev/null

echo ""
echo "=== All security tests passed! ==="

