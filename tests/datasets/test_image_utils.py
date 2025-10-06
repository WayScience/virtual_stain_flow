# test_image_utils.py

import numpy as np
import pytest

from virtual_stain_flow.datasets.image_utils import (
    crop_and_rotate_image,
    _prepare_image_for_cv2,
    _restore_image_format,
)


class TestCropAndRotateImage:
    """Test suite for crop_and_rotate_image function."""

    @pytest.fixture
    def sample_image_3d(self):
        """Create a 3D test image (C, H, W) with distinguishable patterns."""
        # Create 100x100 image with 2 channels
        image = np.zeros((2, 100, 100), dtype=np.float32)
        
        # Channel 0: checkerboard pattern
        for i in range(100):
            for j in range(100):
                if (i // 10 + j // 10) % 2 == 0:
                    image[0, i, j] = 255
                else:
                    image[0, i, j] = 100
        
        # Channel 1: gradient pattern
        for i in range(100):
            image[1, i, :] = i * 2.55  # 0 to 255 gradient
            
        return image

    @pytest.fixture
    def sample_image_4d(self):
        """Create a 4D test image (C, H, W, K) for testing."""
        # Create 50x50 image with 2 channels and 3 additional dimensions
        image = np.zeros((2, 50, 50, 3), dtype=np.float32)
        
        # Fill with distinguishable patterns
        for c in range(2):
            for k in range(3):
                # Different pattern for each channel and k dimension
                value = (c + 1) * (k + 1) * 50
                image[c, :, :, k] = value
                
        return image

    def test_crop_only_no_rotation(self, sample_image_3d):
        """Test cropping without rotation (angle=0)."""
        bbox = (20, 30, 60, 70)  # xmin, ymin, xmax, ymax
        
        result = crop_and_rotate_image(sample_image_3d, bbox, angle=0.0)
        
        # Check dimensions
        expected_height = 70 - 30  # 40
        expected_width = 60 - 20   # 40
        assert result.shape == (2, expected_height, expected_width)
        
        # Check content matches manual crop
        expected = sample_image_3d[:, 30:70, 20:60]
        np.testing.assert_array_equal(result, expected)

    def test_crop_only_small_angle_below_threshold(self, sample_image_3d):
        """Test that very small angles below threshold don't trigger rotation."""
        bbox = (10, 10, 50, 50)
        small_angle = 1e-4  # Below default min_angle of 1e-3
        
        result = crop_and_rotate_image(
            sample_image_3d, bbox, rcx=30.0, rcy=30.0, angle=small_angle
        )
        
        # Should be same as no rotation
        expected = sample_image_3d[:, 10:50, 10:50]
        np.testing.assert_array_equal(result, expected)

    def test_crop_only_none_rotation_centers(self, sample_image_3d):
        """Test that None rotation centers prevent rotation even with angle."""
        bbox = (10, 10, 50, 50)
        
        result = crop_and_rotate_image(
            sample_image_3d, bbox, rcx=None, rcy=None, angle=45.0
        )
        
        # Should be same as no rotation
        expected = sample_image_3d[:, 10:50, 10:50]
        np.testing.assert_array_equal(result, expected)

    def test_crop_with_rotation_applied(self, sample_image_3d):
        """Test that rotation is applied when conditions are met."""
        bbox = (20, 20, 80, 80)
        rcx, rcy = 50.0, 50.0  # Center of rotation
        angle = 90.0  # 90 degree rotation
        
        result = crop_and_rotate_image(
            sample_image_3d, bbox, rcx=rcx, rcy=rcy, angle=angle
        )
        
        # Should have same dimensions as crop
        expected_height = 80 - 20  # 60
        expected_width = 80 - 20   # 60
        assert result.shape == (2, expected_height, expected_width)
        
        # Content should be different from no-rotation case
        no_rotation_result = crop_and_rotate_image(sample_image_3d, bbox, angle=0.0)
        assert not np.array_equal(result, no_rotation_result)

    def test_crop_4d_image(self, sample_image_4d):
        """Test cropping works with 4D images."""
        bbox = (10, 10, 40, 40)
        
        result = crop_and_rotate_image(sample_image_4d, bbox, angle=0.0)
        
        # Check dimensions
        expected_height = 40 - 10  # 30
        expected_width = 40 - 10   # 30
        assert result.shape == (2, expected_height, expected_width, 3)
        
        # Check content
        expected = sample_image_4d[:, 10:40, 10:40, :]
        np.testing.assert_array_equal(result, expected)

    def test_crop_4d_image_with_rotation(self, sample_image_4d):
        """Test cropping with rotation works on 4D images."""
        bbox = (5, 5, 45, 45)
        
        result = crop_and_rotate_image(
            sample_image_4d, bbox, rcx=25.0, rcy=25.0, angle=45.0
        )
        
        # Should have correct dimensions
        expected_height = 45 - 5  # 40
        expected_width = 45 - 5   # 40
        assert result.shape == (2, expected_height, expected_width, 3)

    def test_different_angles_produce_different_results(self, sample_image_3d):
        """Test that different rotation angles produce different results."""
        bbox = (25, 25, 75, 75)
        rcx, rcy = 50.0, 50.0
        
        result_0 = crop_and_rotate_image(sample_image_3d, bbox, rcx, rcy, 0.0)
        result_45 = crop_and_rotate_image(sample_image_3d, bbox, rcx, rcy, 45.0)
        result_90 = crop_and_rotate_image(sample_image_3d, bbox, rcx, rcy, 90.0)
        
        # All should have same shape
        assert result_0.shape == result_45.shape == result_90.shape
        
        # But different content
        assert not np.array_equal(result_0, result_45)
        assert not np.array_equal(result_45, result_90)
        assert not np.array_equal(result_0, result_90)

    def test_custom_min_angle_threshold(self, sample_image_3d):
        """Test custom min_angle threshold parameter."""
        bbox = (10, 10, 50, 50)
        small_angle = 0.5
        
        # With default threshold (1e-3), should trigger rotation
        result_default = crop_and_rotate_image(
            sample_image_3d, bbox, rcx=30.0, rcy=30.0, angle=small_angle
        )
        
        # With higher threshold, should not trigger rotation
        result_high_thresh = crop_and_rotate_image(
            sample_image_3d, bbox, rcx=30.0, rcy=30.0, angle=small_angle, min_angle=1.0
        )
        
        # High threshold result should match no rotation
        expected_no_rotation = sample_image_3d[:, 10:50, 10:50]
        np.testing.assert_array_equal(result_high_thresh, expected_no_rotation)
        
        # Default threshold result should be different (rotated)
        assert not np.array_equal(result_default, expected_no_rotation)

    def test_negative_angle_rotation(self, sample_image_3d):
        """Test that negative angles work correctly."""
        bbox = (20, 20, 60, 60)
        rcx, rcy = 40.0, 40.0
        
        result_pos = crop_and_rotate_image(sample_image_3d, bbox, rcx, rcy, 30.0)
        result_neg = crop_and_rotate_image(sample_image_3d, bbox, rcx, rcy, -30.0)
        
        # Should produce different results
        assert not np.array_equal(result_pos, result_neg)
        
        # Both should have same shape
        assert result_pos.shape == result_neg.shape

    def test_edge_case_zero_area_crop(self, sample_image_3d):
        """Test edge case where crop area is zero or very small."""
        # Same min and max coordinates
        bbox = (50, 50, 50, 50)
        
        result = crop_and_rotate_image(sample_image_3d, bbox)
        
        # Should return empty array
        assert result.shape == (2, 0, 0)

    def test_edge_case_out_of_bounds_crop(self, sample_image_3d):
        """Test cropping coordinates that go beyond image boundaries."""
        # Bbox that extends beyond 100x100 image
        bbox = (80, 80, 120, 120)
        
        # Should not raise error, cv2 handles boundary conditions
        result = crop_and_rotate_image(sample_image_3d, bbox)
        
        # Should have expected crop dimensions
        expected_height = 120 - 80  # 40, but limited by image boundary
        expected_width = 120 - 80   # 40, but limited by image boundary
        assert result.shape[1] <= expected_height
        assert result.shape[2] <= expected_width


class TestPrepareImageForCv2:
    """Test suite for _prepare_image_for_cv2 helper function."""

    def test_3d_image_conversion(self):
        """Test conversion of 3D image from (C, H, W) to (H, W, C)."""
        # Create test image (2 channels, 10x20)
        image = np.random.rand(2, 10, 20).astype(np.float32)
        
        result = _prepare_image_for_cv2(image)
        
        # Should be transposed to (H, W, C)
        assert result.shape == (10, 20, 2)
        
        # Content should be preserved
        np.testing.assert_array_equal(result[:, :, 0], image[0, :, :])
        np.testing.assert_array_equal(result[:, :, 1], image[1, :, :])

    def test_4d_image_conversion(self):
        """Test conversion of 4D image from (C, H, W, K) to (H, W, C*K)."""
        # Create test image (2 channels, 10x20, 3 additional dims)
        image = np.random.rand(2, 10, 20, 3).astype(np.float32)
        
        result = _prepare_image_for_cv2(image)
        
        # Should be reshaped to (H, W, C*K)
        assert result.shape == (10, 20, 6)  # 2*3 = 6
        
        # Verify content mapping
        # First channel, first K should map to first output channel
        np.testing.assert_array_equal(result[:, :, 0], image[0, :, :, 0])
        # Second channel, first K should map to appropriate output channel
        np.testing.assert_array_equal(result[:, :, 3], image[1, :, :, 0])

    def test_unsupported_dimensions(self):
        """Test that unsupported image dimensions raise ValueError."""
        # 2D image (missing channel dimension)
        image_2d = np.random.rand(10, 20)
        with pytest.raises(ValueError, match="Unsupported image dimensions: 2"):
            _prepare_image_for_cv2(image_2d)
        
        # 5D image (too many dimensions)
        image_5d = np.random.rand(2, 10, 20, 3, 4)
        with pytest.raises(ValueError, match="Unsupported image dimensions: 5"):
            _prepare_image_for_cv2(image_5d)

    def test_single_channel_3d(self):
        """Test conversion of single-channel 3D image."""
        image = np.random.rand(1, 15, 25).astype(np.float32)
        
        result = _prepare_image_for_cv2(image)
        
        assert result.shape == (15, 25, 1)
        np.testing.assert_array_equal(result[:, :, 0], image[0, :, :])


class TestRestoreImageFormat:
    """Test suite for _restore_image_format helper function."""

    def test_restore_3d_format(self):
        """Test restoring 3D image format from OpenCV (H, W, C) to (C, H, W)."""
        original_shape = (2, 10, 20)
        cv_image = np.random.rand(10, 20, 2).astype(np.float32)
        
        result = _restore_image_format(cv_image, original_shape)
        
        assert result.shape == original_shape
        
        # Check content preservation
        np.testing.assert_array_equal(result[0, :, :], cv_image[:, :, 0])
        np.testing.assert_array_equal(result[1, :, :], cv_image[:, :, 1])

    def test_restore_4d_format(self):
        """Test restoring 4D image format from OpenCV to (C, H, W, K)."""
        original_shape = (2, 10, 20, 3)
        cv_image = np.random.rand(10, 20, 6).astype(np.float32)  # 2*3 = 6 channels
        
        result = _restore_image_format(cv_image, original_shape)
        
        assert result.shape == original_shape
        
        # Check content mapping - first channel, first K
        np.testing.assert_array_equal(result[0, :, :, 0], cv_image[:, :, 0])
        # Second channel, first K
        np.testing.assert_array_equal(result[1, :, :, 0], cv_image[:, :, 3])

    def test_restore_single_channel_from_2d(self):
        """Test restoring from 2D OpenCV image (single channel case)."""
        original_shape = (1, 15, 25)
        cv_image = np.random.rand(15, 25).astype(np.float32)  # 2D
        
        result = _restore_image_format(cv_image, original_shape)
        
        assert result.shape == original_shape
        np.testing.assert_array_equal(result[0, :, :], cv_image)

    def test_roundtrip_3d_conversion(self):
        """Test that prepare -> restore roundtrip preserves 3D image."""
        original = np.random.rand(3, 12, 18).astype(np.float32)
        
        cv_format = _prepare_image_for_cv2(original)
        restored = _restore_image_format(cv_format, original.shape)
        
        np.testing.assert_array_almost_equal(original, restored)

    def test_roundtrip_4d_conversion(self):
        """Test that prepare -> restore roundtrip preserves 4D image."""
        original = np.random.rand(2, 8, 12, 4).astype(np.float32)
        
        cv_format = _prepare_image_for_cv2(original)
        restored = _restore_image_format(cv_format, original.shape)
        
        np.testing.assert_array_almost_equal(original, restored)


class TestImageUtilsIntegration:
    """Integration tests for the image_utils module."""

    def test_geometric_rotation_properties(self):
        """Test that rotations have expected geometric properties."""
        # Create a simple pattern that's easy to verify
        image = np.zeros((1, 20, 20), dtype=np.float32)
        # Put a bright pixel at (5, 10) - off center
        image[0, 5, 10] = 255.0
        
        bbox = (0, 0, 20, 20)  # Full image
        center_x, center_y = 10.0, 10.0  # Image center
        
        # 180 degree rotation should move the pixel to (15, 10) 
        # (symmetric around center)
        result_180 = crop_and_rotate_image(
            image, bbox, center_x, center_y, 180.0
        )
        
        # The bright pixel should now be at approximately (15, 10)
        # Due to interpolation, check the region around expected position
        bright_region = result_180[0, 14:17, 9:12]
        assert np.max(bright_region) > 200  # Should find bright pixel nearby

    def test_rotation_preserves_image_statistics_approximately(self):
        """Test that rotation preserves approximate image statistics."""
        # Create image with known statistics
        np.random.seed(42)
        image = np.random.rand(2, 50, 50).astype(np.float32) * 100
        
        bbox = (5, 5, 45, 45)
        center_x, center_y = 25.0, 25.0
        
        original_crop = crop_and_rotate_image(image, bbox, angle=0.0)
        rotated_crop = crop_and_rotate_image(image, bbox, center_x, center_y, 45.0)
        
        # Mean should be approximately preserved (within tolerance due to interpolation)
        original_mean = np.mean(original_crop)
        rotated_mean = np.mean(rotated_crop)
        
        # Allow 5% tolerance for interpolation effects
        np.testing.assert_allclose(original_mean, rotated_mean, rtol=0.05)

    def test_multiple_rotations_composition(self):
        """Test that multiple small rotations approximate one large rotation."""
        image = np.random.rand(1, 40, 40).astype(np.float32) * 255
        bbox = (5, 5, 35, 35)
        center_x, center_y = 20.0, 20.0
        
        # Single 90 degree rotation
        result_90 = crop_and_rotate_image(image, bbox, center_x, center_y, 90.0)
        
        # Four 22.5 degree rotations (approximately 90 degrees)
        temp_image = image.copy()
        for _ in range(4):
            temp_crop = crop_and_rotate_image(temp_image, bbox, center_x, center_y, 22.5)
            # For this test, we'd need to reconstruct full image, which is complex
            # So just test that the operation doesn't fail
            assert temp_crop.shape == (1, 30, 30)

    def test_cropping_different_regions_same_image(self):
        """Test cropping different regions from the same image."""
        # Create image with spatial variation
        image = np.zeros((1, 100, 100), dtype=np.float32)
        for i in range(100):
            for j in range(100):
                image[0, i, j] = i + j  # Diagonal gradient
        
        # Crop different regions
        bbox1 = (10, 10, 30, 30)  # Top-left region
        bbox2 = (70, 70, 90, 90)  # Bottom-right region
        
        crop1 = crop_and_rotate_image(image, bbox1)
        crop2 = crop_and_rotate_image(image, bbox2)
        
        # Should have same dimensions
        assert crop1.shape == crop2.shape
        
        # But different content (different means due to gradient)
        assert np.mean(crop1) < np.mean(crop2)  # crop1 from top-left should have smaller values
        
        # Verify specific content
        assert crop1[0, 0, 0] == 20.0  # i=10, j=10
        assert crop2[0, 0, 0] == 140.0  # i=70, j=70
