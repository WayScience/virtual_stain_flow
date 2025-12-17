"""
Tests for input validation
"""

import pytest
import pandas as pd
from pathlib import Path
import pandera as pa

from virtual_stain_flow.datasets.ds_engine.input_validation import (
    make_file_index_schema,
)


class TestMakeFileIndexSchema:
    """Tests for make_file_index_schema function"""
    
    def test_valid_dataframe(self):
        df = pd.DataFrame({
            "col1": ["path1.txt", "path2.txt"],
            "col2": ["path3.txt", "path4.txt"],
        })
        schema = make_file_index_schema(check_exists=False)
        schema.validate(df)  # Should not raise
    
    def test_empty_dataframe_fails(self):
        df = pd.DataFrame()
        schema = make_file_index_schema(check_exists=False)
        with pytest.raises(pa.errors.SchemaError, match="file_index must have at least one column"):
            schema.validate(df)
    
    def test_na_values_fail(self):
        df = pd.DataFrame({
            "col1": ["path1.txt", None],
            "col2": ["path3.txt", "path4.txt"],
        })
        schema = make_file_index_schema(check_exists=False)
        with pytest.raises(pa.errors.SchemaError, match="file_index may not contain NA values"):
            schema.validate(df)
    
    def test_non_pathlike_values_fail(self):
        df = pd.DataFrame({
            "col1": ["path1.txt", 123],
            "col2": ["path3.txt", "path4.txt"],
        })
        schema = make_file_index_schema(check_exists=False)
        with pytest.raises(
            pa.errors.SchemaError, 
            match="All file_index cells must be string/pathlib.Path objects"):
            schema.validate(df)

    def test_whitespace_only_values_fail(self):
        df = pd.DataFrame({
            "col1": ["path1.txt", "   "],
            "col2": ["path3.txt", "path4.txt"],
        })
        schema = make_file_index_schema(check_exists=False)
        with pytest.raises(
            pa.errors.SchemaError, 
            match="All file_index cells must be string/pathlib.Path objects"):
            schema.validate(df)
    
    def test_path_objects_valid(self):
        df = pd.DataFrame({
            "col1": [Path("path1.txt"), Path("path2.txt")],
            "col2": [Path("path3.txt"), Path("path4.txt")],
        })
        schema = make_file_index_schema(check_exists=False)
        schema.validate(df)  # Should not raise
    
    def test_check_exists_with_real_files(self, tmp_path):
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("test")
        file2.write_text("test")
        
        df = pd.DataFrame({
            "col1": [str(file1), str(file2)],
        })
        schema = make_file_index_schema(check_exists=True)
        schema.validate(df)  # Should not raise
    
    def test_check_exists_with_nonexistent_files(self):
        df = pd.DataFrame({
            "col1": ["nonexistent1.txt", "nonexistent2.txt"],
        })
        schema = make_file_index_schema(check_exists=True)
        with pytest.raises(pa.errors.SchemaError, match="that exist on disk"):
            schema.validate(df)
