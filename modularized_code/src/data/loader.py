"""
Data Loading Module
==================

Handles loading and initial processing of industrial vibration data from CSV files.
Provides robust error handling, validation, and data quality reporting.
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from config.settings import DATA_CONFIG
from utils.helpers import validate_data_structure, log_dataframe_info, Timer


class DataLoader:
    """
    Load and combine industrial sensor data from multiple CSV files.
    
    Features:
    - Automatic file discovery and sorting
    - Timestamp parsing and validation
    - Data quality reporting
    - Memory-efficient loading for large datasets
    """
    
    def __init__(self, 
                 data_path: str = None,
                 timestamp_column: str = None,
                 timestamp_format: str = None):
        """
        Initialize DataLoader with configuration.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to data directory (uses config default if None)
        timestamp_column : str, optional
            Name of timestamp column (uses config default if None)
        timestamp_format : str, optional
            Timestamp format string (uses config default if None)
        """
        self.data_path = data_path or DATA_CONFIG["data_path"]
        self.timestamp_column = timestamp_column or DATA_CONFIG["timestamp_column"]
        self.timestamp_format = timestamp_format or DATA_CONFIG["timestamp_format"]
        self.file_pattern = DATA_CONFIG["file_pattern"]
        
        # Initialize state
        self.loaded_files = []
        self.data_info = {}
        
    def discover_files(self) -> List[str]:
        """
        Discover and sort CSV files in the data directory.
        
        Returns:
        --------
        List[str]
            Sorted list of CSV file paths
            
        Raises:
        -------
        FileNotFoundError
            If no CSV files found in directory
        """
        csv_files = sorted(glob.glob(os.path.join(self.data_path, self.file_pattern)))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        self.loaded_files = csv_files
        return csv_files
    
    def validate_file_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Validate structure and contents of a single CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to CSV file
            
        Returns:
        --------
        Dict[str, Any]
            Validation results and file info
        """
        try:
            # Read first few rows to check structure
            sample_df = pd.read_csv(file_path, nrows=5)
            
            validation_info = {
                "file_path": file_path,
                "readable": True,
                "columns": list(sample_df.columns),
                "n_columns": len(sample_df.columns),
                "has_timestamp": self.timestamp_column in sample_df.columns,
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                "issues": []
            }
            
            # Check for timestamp column
            if not validation_info["has_timestamp"]:
                validation_info["issues"].append(f"Missing timestamp column: {self.timestamp_column}")
            
            # Check for minimum columns
            if validation_info["n_columns"] < 2:
                validation_info["issues"].append("File has less than 2 columns")
            
            return validation_info
            
        except Exception as e:
            return {
                "file_path": file_path,
                "readable": False,
                "error": str(e),
                "issues": [f"Cannot read file: {str(e)}"]
            }
    
    def load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a single CSV file with error handling.
        
        Parameters:
        -----------
        file_path : str
            Path to CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            
            # Basic validation
            if df.empty:
                raise ValueError(f"File {file_path} is empty")
            
            if self.timestamp_column not in df.columns:
                raise ValueError(f"Timestamp column '{self.timestamp_column}' not found in {file_path}")
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {str(e)}")
    
    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse timestamp column and set as index.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with timestamp column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with parsed timestamp index
        """
        try:
            # Parse timestamps
            df[self.timestamp_column] = pd.to_datetime(
                df[self.timestamp_column], 
                format=self.timestamp_format,
                errors='coerce'
            )
            
            # Check for parsing failures
            failed_parses = df[self.timestamp_column].isnull().sum()
            if failed_parses > 0:
                print(f"⚠️ Warning: {failed_parses} timestamps could not be parsed")
            
            # Set as index and sort
            df = df.set_index(self.timestamp_column).sort_index()
            
            # Remove rows with invalid timestamps
            df = df.dropna(how='all')  # Remove completely empty rows
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to parse timestamps: {str(e)}")
    
    def load_data(self, validate_files: bool = True) -> pd.DataFrame:
        """
        Load and combine all CSV files in the data directory.
        
        Parameters:
        -----------
        validate_files : bool
            Whether to validate file structure before loading
            
        Returns:
        --------
        pd.DataFrame
            Combined DataFrame with timestamp index
        """
        with Timer("Data loading", verbose=True):
            # Discover files
            csv_files = self.discover_files()
            print(f"📁 Found {len(csv_files)} CSV files")
            
            # Validate files if requested
            if validate_files:
                print("🔍 Validating file structure...")
                validation_results = []
                for file in csv_files:
                    result = self.validate_file_structure(file)
                    validation_results.append(result)
                    
                    if result.get("issues"):
                        print(f"⚠️ Issues in {os.path.basename(file)}: {result['issues']}")
                
                # Check if any files failed validation
                failed_files = [r for r in validation_results if r.get("issues")]
                if failed_files:
                    print(f"⚠️ {len(failed_files)} files have validation issues")
            
            # Load all files
            print(f"📊 Loading data from {len(csv_files)} files...")
            df_list = []
            
            for i, file in enumerate(csv_files):
                try:
                    df_temp = self.load_single_file(file)
                    df_list.append(df_temp)
                    
                    if i == 0 or (i + 1) % 10 == 0:
                        print(f"  • Loaded {i + 1}/{len(csv_files)} files...")
                        
                except Exception as e:
                    print(f"❌ Failed to load {file}: {e}")
                    continue
            
            if not df_list:
                raise RuntimeError("No files could be loaded successfully")
            
            # Combine all DataFrames
            print("🔄 Combining data...")
            df_combined = pd.concat(df_list, ignore_index=True)
            
            # Parse timestamps and set index
            df_final = self.parse_timestamps(df_combined)
            
            # Remove problematic columns if they exist
            columns_to_remove = ['CM2_PV_VRM01_VIBRATION1']  # From original notebook
            for col in columns_to_remove:
                if col in df_final.columns:
                    df_final = df_final.drop(columns=[col])
                    print(f"🗑️ Removed problematic column: {col}")
            
            # Store data info for reporting
            self.data_info = {
                'total_files': len(csv_files),
                'loaded_files': len(df_list),
                'total_rows': len(df_final),
                'total_columns': len(df_final.columns),
                'time_range': (df_final.index.min(), df_final.index.max()),
                'duration_days': (df_final.index.max() - df_final.index.min()).days,
                'memory_usage_mb': df_final.memory_usage(deep=True).sum() / (1024 * 1024)
            }
            
            # Log results
            self._log_loading_results(df_final)
            
            return df_final
    
    def _log_loading_results(self, df: pd.DataFrame) -> None:
        """Log comprehensive loading results."""
        info = self.data_info
        
        print(f"\n✅ Data loading completed successfully!")
        print(f"📊 Dataset Overview:")
        print(f"  • Files processed: {info['loaded_files']}/{info['total_files']}")
        print(f"  • Total rows: {info['total_rows']:,}")
        print(f"  • Total columns: {info['total_columns']}")
        print(f"  • Time range: {info['time_range'][0]} to {info['time_range'][1]}")
        print(f"  • Duration: {info['duration_days']} days")
        print(f"  • Memory usage: {info['memory_usage_mb']:.2f} MB")
        
        # Identify vibration columns
        vibration_cols = [col for col in df.columns if 'VIBRATION' in col.upper()]
        if vibration_cols:
            print(f"\n🎯 Vibration columns found: {len(vibration_cols)}")
            for col in vibration_cols:
                col_stats = df[col].describe()
                print(f"  • {col}:")
                print(f"    - Range: {col_stats['min']:.3f} to {col_stats['max']:.3f}")
                print(f"    - Mean: {col_stats['mean']:.3f} ± {col_stats['std']:.3f}")
        
        # Data quality check
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        columns_with_missing = missing_percentage[missing_percentage > 0]
        
        print(f"\n🔍 Data Quality:")
        print(f"  • Columns with missing data: {len(columns_with_missing)}/{len(df.columns)}")
        if len(columns_with_missing) > 0:
            print(f"  • Average missing: {missing_percentage.mean():.1f}%")
            worst_missing = columns_with_missing.nlargest(3)
            for col, pct in worst_missing.items():
                print(f"    - {col}: {pct:.1f}%")
        else:
            print(f"  • No missing data detected")
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get comprehensive data information."""
        return self.data_info.copy()
    
    def get_column_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed information about columns in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to analyze
            
        Returns:
        --------
        pd.DataFrame
            Column information summary
        """
        column_info = []
        
        for col in df.columns:
            col_data = df[col]
            info = {
                'column': col,
                'dtype': str(col_data.dtype),
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'null_percentage': col_data.isnull().sum() / len(col_data) * 100,
                'unique_count': col_data.nunique(),
                'memory_usage_kb': col_data.memory_usage(deep=True) / 1024
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                stats = col_data.describe()
                info.update({
                    'mean': stats['mean'],
                    'std': stats['std'], 
                    'min': stats['min'],
                    'max': stats['max'],
                    'median': stats['50%']
                })
            
            column_info.append(info)
        
        return pd.DataFrame(column_info)

    def load_from_csv(self, data_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV files - compatibility method for pipeline.
        
        Parameters:
        -----------
        data_path : str
            Path to data directory or specific CSV file
        **kwargs
            Additional parameters (ignored for compatibility)
            
        Returns:
        --------
        pd.DataFrame
            Loaded and processed data
        """
        # Update data path if provided
        if data_path:
            self.data_path = data_path
        
        # Load data using the main load_data method
        return self.load_data()


# Convenience function for direct usage
def load_data(data_path: str = None, **kwargs) -> pd.DataFrame:
    """
    Convenience function to load data directly.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to data directory
    **kwargs
        Additional parameters for DataLoader
        
    Returns:
    --------
    pd.DataFrame
        Loaded and combined data
    """
    loader = DataLoader(data_path=data_path, **kwargs)
    return loader.load_data()