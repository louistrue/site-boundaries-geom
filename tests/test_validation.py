"""
Unit tests for input validation
"""
import pytest
from pydantic import ValidationError

from src.rest_api import GenerateRequest


class TestEGRIDValidation:
    """Test EGRID field validation"""
    
    def test_valid_egrid(self):
        """Test valid EGRID formats"""
        valid_egrids = [
            "CH999979659148",
            "CH123456789",
            "CH123456789012345678"
        ]
        for egrid in valid_egrids:
            req = GenerateRequest(egrid=egrid, radius=100)
            assert req.egrid == egrid
    
    def test_invalid_egrid_too_short(self):
        """Test EGRID that's too short"""
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(egrid="CH123", radius=100)
        assert "egrid" in str(exc_info.value).lower()
    
    def test_invalid_egrid_too_long(self):
        """Test EGRID that's too long"""
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(egrid="CH" + "1" * 20, radius=100)
        assert "egrid" in str(exc_info.value).lower()
    
    def test_invalid_egrid_wrong_prefix(self):
        """Test EGRID with wrong prefix"""
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(egrid="DE1234567890", radius=100)
        assert "egrid" in str(exc_info.value).lower()
    
    def test_invalid_egrid_non_numeric(self):
        """Test EGRID with non-numeric characters"""
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(egrid="CHABCDEFGHIJ", radius=100)
        assert "egrid" in str(exc_info.value).lower()


class TestRadiusValidation:
    """Test radius field validation"""
    
    def test_valid_radius(self):
        """Test valid radius values"""
        req = GenerateRequest(egrid="CH999979659148", radius=500.0)
        assert req.radius == 500.0
    
    def test_radius_zero(self):
        """Test radius of zero (should fail)"""
        with pytest.raises(ValidationError):
            GenerateRequest(egrid="CH999979659148", radius=0.0)
    
    def test_radius_negative(self):
        """Test negative radius (should fail)"""
        with pytest.raises(ValidationError):
            GenerateRequest(egrid="CH999979659148", radius=-100.0)
    
    def test_radius_max_limit(self):
        """Test maximum radius limit (2000m)"""
        req = GenerateRequest(egrid="CH999979659148", radius=2000.0)
        assert req.radius == 2000.0
    
    def test_radius_exceeds_max(self):
        """Test radius exceeding maximum (should fail)"""
        with pytest.raises(ValidationError):
            GenerateRequest(egrid="CH999979659148", radius=3000.0)


class TestResolutionValidation:
    """Test resolution field validation"""
    
    def test_valid_resolution(self):
        """Test valid resolution values"""
        req = GenerateRequest(egrid="CH999979659148", resolution=10.0)
        assert req.resolution == 10.0
    
    def test_resolution_minimum(self):
        """Test minimum resolution (5m)"""
        req = GenerateRequest(egrid="CH999979659148", resolution=5.0)
        assert req.resolution == 5.0
    
    def test_resolution_below_minimum(self):
        """Test resolution below minimum (should fail)"""
        with pytest.raises(ValidationError):
            GenerateRequest(egrid="CH999979659148", resolution=1.0)
    
    def test_resolution_maximum(self):
        """Test maximum resolution (100m)"""
        req = GenerateRequest(egrid="CH999979659148", resolution=100.0)
        assert req.resolution == 100.0
    
    def test_resolution_exceeds_maximum(self):
        """Test resolution exceeding maximum (should fail)"""
        with pytest.raises(ValidationError):
            GenerateRequest(egrid="CH999979659148", resolution=200.0)


class TestDensifyValidation:
    """Test densify field validation"""
    
    def test_valid_densify(self):
        """Test valid densify values"""
        req = GenerateRequest(egrid="CH999979659148", densify=0.5)
        assert req.densify == 0.5
    
    def test_densify_minimum(self):
        """Test minimum densify (0.1m)"""
        req = GenerateRequest(egrid="CH999979659148", densify=0.1)
        assert req.densify == 0.1
    
    def test_densify_below_minimum(self):
        """Test densify below minimum (should fail)"""
        with pytest.raises(ValidationError):
            GenerateRequest(egrid="CH999979659148", densify=0.05)
    
    def test_densify_maximum(self):
        """Test maximum densify (10.0m)"""
        req = GenerateRequest(egrid="CH999979659148", densify=10.0)
        assert req.densify == 10.0
    
    def test_densify_exceeds_maximum(self):
        """Test densify exceeding maximum (should fail)"""
        with pytest.raises(ValidationError):
            GenerateRequest(egrid="CH999979659148", densify=20.0)


class TestComponentValidation:
    """Test component inclusion validation"""
    
    def test_both_components(self):
        """Test with both terrain and site solid"""
        req = GenerateRequest(
            egrid="CH999979659148",
            include_terrain=True,
            include_site_solid=True
        )
        assert req.include_terrain is True
        assert req.include_site_solid is True
    
    def test_only_terrain(self):
        """Test with only terrain"""
        req = GenerateRequest(
            egrid="CH999979659148",
            include_terrain=True,
            include_site_solid=False
        )
        assert req.include_terrain is True
        assert req.include_site_solid is False
    
    def test_only_site_solid(self):
        """Test with only site solid"""
        req = GenerateRequest(
            egrid="CH999979659148",
            include_terrain=False,
            include_site_solid=True
        )
        assert req.include_terrain is False
        assert req.include_site_solid is True
    
    def test_neither_component(self):
        """Test with neither component (should fail)"""
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(
                egrid="CH999979659148",
                include_terrain=False,
                include_site_solid=False
            )
        assert "include_terrain" in str(exc_info.value).lower() or \
               "include_site_solid" in str(exc_info.value).lower()


class TestOutputNameValidation:
    """Test output_name field validation"""
    
    def test_default_output_name(self):
        """Test default output name"""
        req = GenerateRequest(egrid="CH999979659148")
        assert req.output_name == "combined_terrain.ifc"
    
    def test_custom_output_name(self):
        """Test custom output name"""
        req = GenerateRequest(
            egrid="CH999979659148",
            output_name="custom.ifc"
        )
        assert req.output_name == "custom.ifc"
    
    def test_output_name_without_extension(self):
        """Test output name without .ifc extension (should be added)"""
        # Note: The extension is added in _ensure_ifc_extension function
        # This test verifies the field accepts any string
        req = GenerateRequest(
            egrid="CH999979659148",
            output_name="custom"
        )
        assert req.output_name == "custom"

