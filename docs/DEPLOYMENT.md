# Secure Deployment Guide for Sevalla

This guide covers deploying the Site Boundaries Terrain API to Sevalla with security best practices.

## Prerequisites

- Git repository with the application code
- Sevalla account
- Docker installed locally (for testing)

## Local Testing

Before deploying to Sevalla, test the Docker image locally:

```bash
# Build the Docker image
docker build -t site-boundaries-api .

# Run security tests
./test_docker_security.sh

# Test the API manually
docker run -d -p 8080:8080 site-boundaries-api

# Test health endpoint
curl http://localhost:8080/health

# Test generation endpoint (small params for speed)
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"egrid":"CH999979659148","radius":100,"resolution":25}' \
  -o test.ifc

# Stop container
docker stop $(docker ps -q --filter ancestor=site-boundaries-api)
```

## Sevalla Deployment Steps

### 1. Push Code to Git Repository

Ensure all files are committed and pushed:

```bash
git add Dockerfile .dockerignore requirements.txt src/
git commit -m "Add secure Docker deployment configuration"
git push
```

### 2. Create Application in Sevalla

1. Log in to your Sevalla account
2. Navigate to **Applications** → **Create an app**
3. Connect your Git repository (GitHub, GitLab, or Bitbucket)
4. Select the repository and branch containing your code

### 3. Configure Build Settings

In your application's **Settings** → **Build**:

- **Build Environment**: `Dockerfile`
- **Dockerfile Path**: `Dockerfile`
- **Context**: `.`

### 4. Configure Environment Variables

In **Settings** → **Environment Variables**, add:

| Variable | Value | Description |
|----------|-------|-------------|
| `ALLOWED_ORIGINS` | `https://yourdomain.com` | CORS allowed origins (comma-separated) |
| `ALLOWED_HOSTS` | `yourdomain.com,*.sevalla.app` | Trusted hosts (comma-separated) |
| `ENABLE_DOCS` | `false` | Disable API docs in production (optional) |

**Note**: `PORT` is automatically set by Sevalla - do not override it.

### 5. Configure Health Check

In **Settings** → **Health Check**:

- **Health Check Path**: `/health`
- This enables zero-downtime deployments

### 6. Select Pod Size

- **Recommended**: Medium (2GB RAM, 1 vCPU)
- IFC generation is memory-intensive; upgrade if needed

### 7. Deploy

1. Navigate to **Deployments**
2. Click **Deploy now**
3. Select your branch
4. Monitor deployment logs

## Security Features Implemented

✅ **Container Security**
- Non-root user (`appuser`)
- Multi-stage build (minimal runtime)
- No build tools in production image

✅ **Network Security**
- CORS middleware (configurable)
- Security headers (X-Content-Type-Options, X-Frame-Options, etc.)
- Trusted Host middleware
- HTTPS via Sevalla TLS termination

✅ **Application Security**
- Rate limiting (10/min for `/generate`, 20/min for `/jobs`)
- Input validation (EGRID pattern, parameter bounds)
- Request size limits (via FastAPI)

✅ **Dependency Security**
- Pinned package versions
- Regular security updates recommended

## Monitoring

After deployment:

1. Check application logs in Sevalla dashboard
2. Test the health endpoint: `https://your-app.sevalla.app/health`
3. Monitor resource usage and adjust pod size if needed

## Troubleshooting

### Build Fails

- Check Dockerfile syntax
- Verify all dependencies are in `requirements.txt`
- Check build logs for specific errors

### Container Won't Start

- Verify `PORT` environment variable is being read correctly
- Check health check endpoint is accessible
- Review application logs

### Rate Limiting Too Strict

- Adjust limits in `src/rest_api.py` (search for `@limiter.limit`)
- Redeploy after changes

### Memory Issues

- Upgrade to larger pod size
- Reduce `radius` and `resolution` parameters in requests
- Use `/jobs` endpoint for large requests instead of `/generate`

## Maintenance

- **Regular Updates**: Update pinned versions in `requirements.txt` periodically
- **Security Scanning**: Use `docker scout cves site-boundaries-api` to check for vulnerabilities
- **Monitoring**: Set up alerts for health check failures in Sevalla

