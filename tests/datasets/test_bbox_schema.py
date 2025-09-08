# test_bbox_schema.py
import numpy as np
import pandas as pd
import pytest

from virtual_stain_flow.datasets.bbox_schema import BBoxSchema, BBoxAccessor, BBoxRowView


class TestBBoxSchema:
    """Test suite for BBoxSchema class."""

    def test_init_default(self):
        """Test BBoxSchema initialization with default values."""
        schema = BBoxSchema()
        assert schema.prefix == ""

    def test_init_with_prefix(self):
        """Test BBoxSchema initialization with custom prefix."""
        schema = BBoxSchema(prefix="bbox_")
        assert schema.prefix == "bbox_"

    def test_dynamic_attribute_access(self):
        """Test dynamic attribute access for column names."""
        schema = BBoxSchema()
        
        # Test standard column names
        assert schema.xmin == "xmin"
        assert schema.ymin == "ymin"
        assert schema.xmax == "xmax"
        assert schema.ymax == "ymax"
        assert schema.cx == "cx"
        assert schema.cy == "cy"
        assert schema.rcx == "rcx"
        assert schema.rcy == "rcy"
        assert schema.angle == "angle"

    def test_dynamic_attribute_access_with_prefix(self):
        """Test dynamic attribute access with prefix."""
        schema = BBoxSchema(prefix="bbox_")
        
        assert schema.xmin == "bbox_xmin"
        assert schema.ymin == "bbox_ymin"
        assert schema.xmax == "bbox_xmax"
        assert schema.ymax == "bbox_ymax"
        assert schema.cx == "bbox_cx"
        assert schema.cy == "bbox_cy"
        assert schema.rcx == "bbox_rcx"
        assert schema.rcy == "bbox_rcy"
        assert schema.angle == "bbox_angle"

    def test_invalid_attribute_access(self):
        """Test that accessing invalid attributes raises AttributeError."""
        schema = BBoxSchema()
        
        with pytest.raises(AttributeError, match="has no attribute 'invalid_attr'"):
            _ = schema.invalid_attr

    def test_bbox_cols_property(self):
        """Test bbox_cols property returns correct tuple."""
        schema = BBoxSchema()
        assert schema.bbox_cols == ("xmin", "ymin", "xmax", "ymax")

    def test_bbox_cols_property_with_prefix(self):
        """Test bbox_cols property with prefix."""
        schema = BBoxSchema(prefix="crop_")
        assert schema.bbox_cols == ("crop_xmin", "crop_ymin", "crop_xmax", "crop_ymax")

    def test_find_column_exact_match(self):
        """Test find_column with exact column name match."""
        df = pd.DataFrame({"xmin": [1, 2], "ymin": [3, 4]})
        schema = BBoxSchema()
        
        assert schema.find_column(df, "xmin") == "xmin"
        assert schema.find_column(df, "ymin") == "ymin"

    def test_find_column_alias_match(self):
        """Test find_column with alias matches."""
        df = pd.DataFrame({"x_min": [1, 2], "top": [3, 4], "right": [5, 6]})
        schema = BBoxSchema()
        
        assert schema.find_column(df, "xmin") == "x_min"
        assert schema.find_column(df, "ymin") == "top"
        assert schema.find_column(df, "xmax") == "right"

    def test_find_column_case_variations(self):
        """Test find_column with case variations."""
        df = pd.DataFrame({"XMIN": [1, 2], "xmax": [3, 4]})
        schema = BBoxSchema()
        
        assert schema.find_column(df, "xmin") == "XMIN"
        assert schema.find_column(df, "xmax") == "xmax"

    def test_find_column_with_prefix(self):
        """Test find_column with prefix in schema."""
        df = pd.DataFrame({"bbox_xmin": [1, 2], "bbox_ymin": [3, 4]})
        schema = BBoxSchema(prefix="bbox_")
        
        assert schema.find_column(df, "xmin") == "bbox_xmin"
        assert schema.find_column(df, "ymin") == "bbox_ymin"

    def test_find_column_not_found(self):
        """Test find_column raises ValueError when column not found."""
        df = pd.DataFrame({"other_col": [1, 2]})
        schema = BBoxSchema()
        
        with pytest.raises(ValueError, match="No column found for key 'xmin'"):
            schema.find_column(df, "xmin")

    def test_frozen_dataclass(self):
        """Test that BBoxSchema is frozen (immutable)."""
        schema = BBoxSchema(prefix="test_")
        
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            schema.prefix = "new_prefix"


class TestBBoxAccessor:
    """Test suite for BBoxAccessor class."""

    @pytest.fixture
    def basic_bbox_df(self):
        """Basic DataFrame with bbox columns."""
        return pd.DataFrame({
            "xmin": [10, 20],
            "ymin": [15, 25],
            "xmax": [50, 60],
            "ymax": [55, 65],
            "angle": [0.0, 45.0],
            "rcx": [30.0, 40.0],
            "rcy": [35.0, 45.0],
        })

    @pytest.fixture
    def minimal_bbox_df(self):
        """Minimal DataFrame with only required bbox columns."""
        return pd.DataFrame({
            "xmin": [5, 15],
            "ymin": [10, 20],
            "xmax": [45, 55],
            "ymax": [50, 60],
        })

    def test_accessor_registration(self, basic_bbox_df):
        """Test that the bbox accessor is properly registered with pandas."""
        assert hasattr(basic_bbox_df, "bbox")
        assert callable(basic_bbox_df.bbox)

    def test_accessor_initialization(self, basic_bbox_df):
        """Test accessor initialization."""
        accessor = basic_bbox_df.bbox
        
        assert isinstance(accessor, BBoxAccessor)
        assert accessor._df is basic_bbox_df

    def test_accessor_with_custom_schema(self, basic_bbox_df):
        """Test accessor with custom schema."""
        custom_schema = BBoxSchema(prefix="bbox_")
        accessor = basic_bbox_df.bbox(custom_schema)
        
        assert accessor._schema is custom_schema
        assert accessor._schema.prefix == "bbox_"

    def test_ensure_columns_with_complete_df(self, basic_bbox_df):
        """Test ensure_columns with DataFrame that has all columns."""
        accessor = basic_bbox_df.bbox
        result_df = accessor.ensure_columns()
        
        # Should return DataFrame with all required columns
        required_cols = ["xmin", "ymin", "xmax", "ymax", "cx", "cy", "rcx", "rcy", "angle"]
        for col in required_cols:
            assert col in result_df.columns
        
        # Original bbox columns should be preserved
        np.testing.assert_array_equal(result_df["xmin"], [10, 20])
        np.testing.assert_array_equal(result_df["ymin"], [15, 25])

    def test_ensure_columns_creates_missing_centers(self, minimal_bbox_df):
        """Test ensure_columns creates missing center columns."""
        accessor = minimal_bbox_df.bbox
        result_df = accessor.ensure_columns()
        
        # Should create cx and cy columns
        assert "cx" in result_df.columns
        assert "cy" in result_df.columns
        
        # Check calculated values
        expected_cx = [(5 + 45) / 2, (15 + 55) / 2]  # [25.0, 35.0]
        expected_cy = [(10 + 50) / 2, (20 + 60) / 2]  # [30.0, 40.0]
        
        np.testing.assert_array_equal(result_df["cx"], expected_cx)
        np.testing.assert_array_equal(result_df["cy"], expected_cy)

    def test_ensure_columns_creates_missing_rotation_centers(self, minimal_bbox_df):
        """Test ensure_columns creates missing rotation center columns."""
        accessor = minimal_bbox_df.bbox
        result_df = accessor.ensure_columns()
        
        # Should create rcx and rcy columns defaulting to cx and cy
        assert "rcx" in result_df.columns
        assert "rcy" in result_df.columns
        
        # Should default to center values
        np.testing.assert_array_equal(result_df["rcx"], result_df["cx"])
        np.testing.assert_array_equal(result_df["rcy"], result_df["cy"])

    def test_ensure_columns_creates_missing_angle(self, minimal_bbox_df):
        """Test ensure_columns creates missing angle column."""
        accessor = minimal_bbox_df.bbox
        result_df = accessor.ensure_columns()
        
        # Should create angle column with default 0.0
        assert "angle" in result_df.columns
        np.testing.assert_array_equal(result_df["angle"], [0.0, 0.0])

    def test_ensure_columns_preserves_existing_values(self, basic_bbox_df):
        """Test ensure_columns preserves existing column values."""
        accessor = basic_bbox_df.bbox
        result_df = accessor.ensure_columns()
        
        # Existing values should be preserved
        np.testing.assert_array_equal(result_df["angle"], [0.0, 45.0])
        np.testing.assert_array_equal(result_df["rcx"], [30.0, 40.0])
        np.testing.assert_array_equal(result_df["rcy"], [35.0, 45.0])

    def test_ensure_columns_handles_alternative_names(self):
        """Test ensure_columns works with alternative column names."""
        df = pd.DataFrame({
            "x_min": [10, 20],
            "y_min": [15, 25],
            "x_max": [50, 60],
            "y_max": [55, 65],
            "rotation": [30.0, 60.0],  # Alternative to "angle"
        })
        
        accessor = df.bbox
        result_df = accessor.ensure_columns()
        
        # Should find and use alternative names
        assert accessor._cols["xmin"] == "x_min"
        assert accessor._cols["angle"] == "rotation"
        
        # Values should be preserved
        np.testing.assert_array_equal(result_df["rotation"], [30.0, 60.0])

    def test_coords_method(self, basic_bbox_df):
        """Test coords method returns correct bbox coordinates."""
        accessor = basic_bbox_df.bbox
        accessor.ensure_columns()
        
        coords_0 = accessor.coords(0)
        coords_1 = accessor.coords(1)
        
        assert coords_0 == (10, 15, 50, 55)
        assert coords_1 == (20, 25, 60, 65)

    def test_centers_method(self, basic_bbox_df):
        """Test centers method returns correct center coordinates."""
        accessor = basic_bbox_df.bbox
        accessor.ensure_columns()
        
        centers_0 = accessor.centers(0)
        centers_1 = accessor.centers(1)
        
        # Calculated from bbox coordinates
        expected_cx_0 = (10 + 50) / 2  # 30.0
        expected_cy_0 = (15 + 55) / 2  # 35.0
        expected_cx_1 = (20 + 60) / 2  # 40.0
        expected_cy_1 = (25 + 65) / 2  # 45.0
        
        assert centers_0 == (expected_cx_0, expected_cy_0)
        assert centers_1 == (expected_cx_1, expected_cy_1)

    def test_rot_centers_method(self, basic_bbox_df):
        """Test rot_centers method returns correct rotation centers."""
        accessor = basic_bbox_df.bbox
        accessor.ensure_columns()
        
        rot_centers_0 = accessor.rot_centers(0)
        rot_centers_1 = accessor.rot_centers(1)
        
        assert rot_centers_0 == (30.0, 35.0)
        assert rot_centers_1 == (40.0, 45.0)

    def test_angle_of_method(self, basic_bbox_df):
        """Test angle_of method returns correct angles."""
        accessor = basic_bbox_df.bbox
        accessor.ensure_columns()
        
        angle_0 = accessor.angle_of(0)
        angle_1 = accessor.angle_of(1)
        
        assert angle_0 == 0.0
        assert angle_1 == 45.0

    def test_row_method_returns_bbox_row_view(self, basic_bbox_df):
        """Test row method returns BBoxRowView instance."""
        accessor = basic_bbox_df.bbox
        accessor.ensure_columns()
        
        row_view = accessor.row(0)
        
        assert isinstance(row_view, BBoxRowView)
        assert row_view._acc is accessor


class TestBBoxRowView:
    """Test suite for BBoxRowView class."""

    @pytest.fixture
    def setup_row_view(self):
        """Set up a BBoxRowView for testing."""
        df = pd.DataFrame({
            "xmin": [10, 20],
            "ymin": [15, 25],
            "xmax": [50, 60],
            "ymax": [55, 65],
            "angle": [0.0, 45.0],
            "rcx": [30.0, 40.0],
            "rcy": [35.0, 45.0],
        })
        
        accessor = df.bbox
        accessor.ensure_columns()
        
        return accessor.row(0), accessor.row(1)

    def test_bbox_property(self, setup_row_view):
        """Test bbox property returns correct coordinates."""
        row_view_0, row_view_1 = setup_row_view
        
        assert row_view_0.bbox == (10, 15, 50, 55)
        assert row_view_1.bbox == (20, 25, 60, 65)

    def test_center_property(self, setup_row_view):
        """Test center property returns correct center coordinates."""
        row_view_0, row_view_1 = setup_row_view
        
        # Calculated from bbox coordinates
        expected_center_0 = (30.0, 35.0)  # (10+50)/2, (15+55)/2
        expected_center_1 = (40.0, 45.0)  # (20+60)/2, (25+65)/2
        
        assert row_view_0.center == expected_center_0
        assert row_view_1.center == expected_center_1

    def test_rot_center_property(self, setup_row_view):
        """Test rot_center property returns correct rotation centers."""
        row_view_0, row_view_1 = setup_row_view
        
        assert row_view_0.rot_center == (30.0, 35.0)
        assert row_view_1.rot_center == (40.0, 45.0)

    def test_angle_property(self, setup_row_view):
        """Test angle property returns correct angles."""
        row_view_0, row_view_1 = setup_row_view
        
        assert row_view_0.angle == 0.0
        assert row_view_1.angle == 45.0


class TestBBoxSchemaIntegration:
    """Integration tests for bbox schema components."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from DataFrame to accessing bbox data."""
        # Start with DataFrame using alternative column names
        df = pd.DataFrame({
            "left": [5, 15, 25],
            "top": [10, 20, 30],
            "right": [35, 45, 55],
            "bottom": [40, 50, 60],
            "rotation": [0.0, 30.0, 60.0],
        })
        
        # Use custom schema with prefix
        schema = BBoxSchema(prefix="bbox_")
        accessor = df.bbox(schema)
        
        # Ensure columns creates missing ones
        result_df = accessor.ensure_columns()
        
        # Should have all required columns
        assert "bbox_cx" in result_df.columns
        assert "bbox_cy" in result_df.columns
        assert "bbox_rcx" in result_df.columns
        assert "bbox_rcy" in result_df.columns
        
        # Should preserve original angle column
        assert accessor._cols["angle"] == "rotation"
        
        # Access data through accessor methods
        coords = accessor.coords(1)
        centers = accessor.centers(1)
        rot_centers = accessor.rot_centers(1)
        angle = accessor.angle_of(1)
        
        assert coords == (15, 20, 45, 50)
        assert centers == (30.0, 35.0)  # (15+45)/2, (20+50)/2
        assert rot_centers == (30.0, 35.0)  # Should default to centers
        assert angle == 30.0

    def test_missing_required_columns_error(self):
        """Test that missing required columns raise appropriate errors."""
        # DataFrame missing required bbox columns
        df = pd.DataFrame({
            "some_col": [1, 2, 3],
            "other_col": [4, 5, 6],
        })
        
        accessor = df.bbox
        
        with pytest.raises(ValueError, match="No column found for key 'xmin'"):
            accessor.ensure_columns()

    def test_schema_column_mapping_priority(self):
        """Test column mapping priority follows the order in _column_map."""
        # DataFrame with multiple possible column names
        df = pd.DataFrame({
            "xmin": [10, 20],
            "x_min": [100, 200],  # This should be used since it's first in _column_map
            "ymin": [15, 25],
            "ymax": [55, 65],
            "xmax": [50, 60],
        })
        
        accessor = df.bbox
        accessor.ensure_columns()
        
        # Should use "x_min" since it appears first in the _column_map for 'xmin'
        assert accessor._cols["xmin"] == "x_min"
        
        # Verify values come from the correct column
        coords = accessor.coords(0)
        assert coords[0] == 100  # Should be from "x_min" column, not "xmin"

    def test_data_type_enforcement(self):
        """Test that ensure_columns enforces appropriate data types."""
        df = pd.DataFrame({
            "xmin": ["10", "20"],  # String values
            "ymin": ["15", "25"],
            "xmax": ["50", "60"],
            "ymax": ["55", "65"],
        })
        
        accessor = df.bbox
        result_df = accessor.ensure_columns()
        
        # Integer columns should be converted to int
        assert result_df["xmin"].dtype in [np.int32, np.int64]
        assert result_df["ymin"].dtype in [np.int32, np.int64]
        
        # Float columns should be float
        assert result_df["cx"].dtype in [np.float32, np.float64]
        assert result_df["angle"].dtype in [np.float32, np.float64]

    def test_accessor_state_preservation(self):
        """Test that accessor state is properly preserved across operations."""
        df = pd.DataFrame({
            "xmin": [10, 20],
            "ymin": [15, 25],
            "xmax": [50, 60],
            "ymax": [55, 65],
        })
        
        # Create accessor with custom schema
        custom_schema = BBoxSchema(prefix="test_")
        accessor = df.bbox(custom_schema)
        
        # Ensure columns
        accessor.ensure_columns()
        
        # Verify schema is preserved
        assert accessor._schema.prefix == "test_"
        
        # Verify column mappings are preserved
        assert "xmin" in accessor._cols
        assert "cx" in accessor._cols
        
        # Multiple method calls should work consistently
        coords_1 = accessor.coords(0)
        coords_2 = accessor.coords(0)
        assert coords_1 == coords_2
