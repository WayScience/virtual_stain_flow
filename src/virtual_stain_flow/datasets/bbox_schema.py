"""
bbox_schema.py

This module defines a schema and accessor for bounding box (bbox) metadata
defining crops in raw images to be extracted and returned by a dataset.
For the purpose of extensibility, the schema additionally defines a rotation
center and angle. Intend to be used by a dataset class as the source of truth
for bbox metadata, column definition and accessor.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class BBoxSchema:
    """
    Centralized bbox column name definitions with flexible aliasing.
    This class defines standard names for bounding box columns and allows
    for flexible aliasing and prefixing to accommodate different dataframe
    naming conventions.
    """
    prefix: str = ""
    
    # mapping canonical keys used by the accessor to possible column names
    # in the dataframe
    _column_map = {
        'xmin': ['x_min', 'xmin', 'left', 'x1'],
        'ymin': ['y_min', 'ymin', 'top', 'y1'], 
        'xmax': ['x_max', 'xmax', 'right', 'x2'],
        'ymax': ['y_max', 'ymax', 'bottom', 'y2'],
        'cx': ['box_x_center', 'cx', 'center_x'],
        'cy': ['box_y_center', 'cy', 'center_y'],
        'rcx': ['rot_x_center', 'rot_cx', 'rcx'],
        'rcy': ['rot_y_center', 'rot_cy', 'rcy'],
        'angle': ['angle', 'rotation', 'theta']
    }
    
    def __getattr__(self, name: str) -> str:
        """
        Dynamic access: schema.xmin, schema.cx, etc.
        This is for making easier access to prefixed column names. 
        Alternatively this could have been implemented with properties 
        but since we have a lot of fields we wish to access this is the
        more compact approach.
        """
        if name in self._column_map:
            return f"{self.prefix}{name}"
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def find_column(self, df: pd.DataFrame, key: str) -> str:
        """Find actual column name in DataFrame for given key."""
        for alias in self._column_map.get(key, []):
            for variant in [alias, f"{self.prefix}{alias}", alias.upper(), alias.lower()]:
                if variant in df.columns:
                    return variant
        raise ValueError(f"No column found for key '{key}' in {list(df.columns)}")
    
    @property
    def bbox_cols(self) -> Tuple[str, str, str, str]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

class BBoxRowView:
    """
    Row accessor for bbox data.
    Useful for dataset class to access a single bbox defined crop selection.
    """
    def __init__(self, row: pd.Series, accessor: BBoxAccessor):
        self._row, self._acc = row, accessor
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return tuple(int(self._row[self._acc._cols[k]]) for k in ['xmin', 'ymin', 'xmax', 'ymax'])
    
    @property
    def center(self) -> Tuple[float, float]:
        return (float(self._row[self._acc._cols['cx']]), 
                float(self._row[self._acc._cols['cy']]))
    
    @property
    def rot_center(self) -> Tuple[float, float]:
        return (float(self._row[self._acc._cols['rcx']]), 
                float(self._row[self._acc._cols['rcy']]))
    
    @property
    def angle(self) -> float:
        return float(self._row[self._acc._cols['angle']])

@pd.api.extensions.register_dataframe_accessor("bbox")
class BBoxAccessor:
    """
    Pandas accessor for bbox operations.
    This accessor provides methods to ensure required bbox columns exist,
    create missing ones, and access bbox data in a structured way.
    1. ensure_columns(): Ensures required columns exist, creates missing ones.
    2. row(i): Returns a BBoxRowView for the i-th row.
    3. coords(i): Returns bbox coordinates for the i-th row.
    4. centers(i): Returns bbox center for the i-th row.
    5. rot_centers(i): Returns rotation center for the i-th row.
    6. angle_of(i): Returns rotation angle for the i-th row.
    """
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._schema = BBoxSchema()
        self._cols = {}
    
    def __call__(self, schema: BBoxSchema) -> BBoxAccessor:
        acc = BBoxAccessor(self._df)
        acc._schema = schema
        acc._cols = self._cols.copy()  # Preserve column mapping
        return acc
    
    def ensure_columns(self) -> pd.DataFrame:
        """Ensure required columns exist, create missing ones."""
        df, s = self._df, self._schema
        
        # Find required bbox columns
        required = ['xmin', 'ymin', 'xmax', 'ymax']
        for key in required:
            self._cols[key] = s.find_column(df, key)
            df[self._cols[key]] = df[self._cols[key]].astype(int)
        
        # Create/find centers
        for key, calc in [('cx', lambda: (df[self._cols['xmin']] + df[self._cols['xmax']]) / 2),
                         ('cy', lambda: (df[self._cols['ymin']] + df[self._cols['ymax']]) / 2)]:
            try:
                self._cols[key] = s.find_column(df, key)
            except ValueError:
                # Create column with proper name
                col_name = getattr(s, key)
                df[col_name] = calc()
                self._cols[key] = col_name
        
        # Create/find rotation centers and angle
        for key, default_key in [('rcx', 'cx'),
                                ('rcy', 'cy')]:
            try:
                self._cols[key] = s.find_column(df, key)
            except ValueError:
                # Create column with proper name
                col_name = getattr(s, key)
                df[col_name] = df[self._cols[default_key]]
                self._cols[key] = col_name
            df[self._cols[key]] = df[self._cols[key]].astype(float)
        
        # Handle angle
        try:
            self._cols['angle'] = s.find_column(df, 'angle')
        except ValueError:
            col_name = s.angle
            df[col_name] = 0.0
            self._cols['angle'] = col_name
        df[self._cols['angle']] = df[self._cols['angle']].astype(float)
        
        self._ensure_cols_mapped()
        
        return df
    
    def _ensure_cols_mapped(self):
        """Ensure column mapping is established."""
        if not self._cols:
            # Direct mapping for columns that exist in the dataframe
            for key in ['xmin', 'ymin', 'xmax', 'ymax', 'cx', 'cy', 
                       'rcx', 'rcy', 'angle']:
                if key in self._df.columns:
                    self._cols[key] = key
                else:
                    try:
                        self._cols[key] = self._schema.find_column(self._df, key)
                    except ValueError:
                        pass
    
    def row(self, i: int) -> BBoxRowView:
        return BBoxRowView(self._df.iloc[i], self)
    
    def coords(self, i: int) -> Tuple[int, int, int, int]:
        return self.row(i).bbox
    
    def centers(self, i: int) -> Tuple[float, float]:
        return self.row(i).center
    
    def rot_centers(self, i: int) -> Tuple[float, float]:
        return self.row(i).rot_center
    
    def angle_of(self, i: int) -> float:
        return self.row(i).angle
