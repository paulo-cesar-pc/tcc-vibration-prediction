"""
General Helper Functions
=======================

Common utility functions used throughout the vibration prediction system.
"""

import warnings
import time
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
from datetime import datetime

from ..config.settings import SYSTEM_CONFIG, VISUALIZATION_CONFIG


def setup_warnings(suppress: bool = True, categories: Optional[list] = None) -> None:
    """
    Setup warning filters for cleaner output.
    
    Parameters:
    -----------
    suppress : bool
        Whether to suppress warnings
    categories : Optional[list]
        List of warning categories to suppress
    """
    if suppress:
        if categories is None:
            categories = SYSTEM_CONFIG.get("warning_categories", ["sklearn", "pandas", "matplotlib"])
        
        for category in categories:
            warnings.filterwarnings('ignore', category=UserWarning, module=category)
        
        # Suppress common warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', message='.*divide by zero.*')
        warnings.filterwarnings('ignore', message='.*invalid value.*')


def setup_plotting(style: str = "default", palette: str = "husl") -> None:
    """
    Setup matplotlib and seaborn styling.
    
    Parameters:
    -----------
    style : str
        Matplotlib style to use
    palette : str
        Seaborn color palette
    """
    plt.style.use(style)
    sns.set_palette(palette)
    
    # Set pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print("✅ Plotting environment configured")


def validate_data_structure(
    df: pd.DataFrame,
    required_columns: Optional[list] = None,
    min_rows: int = 10,
    max_missing_ratio: float = 0.5
) -> Tuple[bool, list]:
    """
    Validate DataFrame structure and data quality.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : Optional[list]
        List of required column names
    min_rows : int
        Minimum number of rows required
    max_missing_ratio : float
        Maximum ratio of missing values allowed
        
    Returns:
    --------
    Tuple[bool, list]
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check basic structure
    if df.empty:
        issues.append("DataFrame is empty")
        return False, issues
    
    if len(df) < min_rows:
        issues.append(f"DataFrame has only {len(df)} rows (minimum {min_rows} required)")
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
    
    # Check missing data
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_ratio > max_missing_ratio:
        issues.append(f"Too much missing data: {missing_ratio:.2%} (max allowed: {max_missing_ratio:.2%})")
    
    # Check for duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        issues.append(f"Duplicate columns found: {duplicate_cols}")
    
    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        issues.append(f"Columns with all null values: {null_cols}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def print_section_header(title: str, symbol: str = "=", width: int = 50) -> None:
    """
    Print a formatted section header.
    
    Parameters:
    -----------
    title : str
        Title of the section
    symbol : str
        Symbol to use for the header line
    width : int
        Width of the header line
    """
    print(f"\n{title}")
    print(symbol * width)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Parameters:
    -----------
    seconds : float
        Duration in seconds
        
    Returns:
    --------
    str
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, Union[float, str]]:
    """
    Calculate memory usage of DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    Dict[str, Union[float, str]]
        Memory usage statistics
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)
    memory_gb = memory_mb / 1024
    
    if memory_gb > 1:
        memory_str = f"{memory_gb:.2f} GB"
    elif memory_mb > 1:
        memory_str = f"{memory_mb:.2f} MB"
    else:
        memory_str = f"{memory_bytes / 1024:.2f} KB"
    
    return {
        "bytes": memory_bytes,
        "mb": memory_mb,
        "gb": memory_gb,
        "formatted": memory_str
    }


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and optimization.
    
    Returns:
    --------
    Dict[str, Any]
        System information
    """
    return {
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "memory_percent": psutil.virtual_memory().percent,
        "python_version": f"{psutil.PYTHON}",  # Will show Python major version
    }


def create_results_directory(base_path: Optional[Path] = None, 
                           timestamp: bool = True) -> Path:
    """
    Create a results directory with optional timestamp.
    
    Parameters:
    -----------
    base_path : Optional[Path]
        Base path for results directory
    timestamp : bool
        Whether to include timestamp in directory name
        
    Returns:
    --------
    Path
        Path to created directory
    """
    if base_path is None:
        base_path = Path("results")
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = base_path / f"run_{timestamp_str}"
    else:
        results_dir = base_path
    
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def safe_divide(numerator: Union[float, np.ndarray], 
                denominator: Union[float, np.ndarray], 
                default: float = 0.0) -> Union[float, np.ndarray]:
    """
    Perform safe division avoiding division by zero.
    
    Parameters:
    -----------
    numerator : Union[float, np.ndarray]
        Numerator values
    denominator : Union[float, np.ndarray]
        Denominator values
    default : float
        Default value when denominator is zero
        
    Returns:
    --------
    Union[float, np.ndarray]
        Division results with safe handling of zero division
    """
    if isinstance(denominator, np.ndarray):
        result = np.full_like(numerator, default, dtype=float)
        mask = denominator != 0
        result[mask] = numerator[mask] / denominator[mask]
        return result
    else:
        return numerator / denominator if denominator != 0 else default


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, description: str = "Operation", verbose: bool = True):
        self.description = description
        self.verbose = verbose
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            print(f"🕐 Starting {self.description}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        if self.verbose:
            print(f"✅ {self.description} completed in {format_duration(self.duration)}")


def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Log comprehensive information about a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    name : str
        Name of the DataFrame for logging
    """
    print(f"\n📊 {name} Information:")
    print(f"  • Shape: {df.shape}")
    print(f"  • Memory usage: {calculate_memory_usage(df)['formatted']}")
    
    # Data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    
    print(f"  • Numeric columns: {len(numeric_cols)}")
    print(f"  • Categorical columns: {len(categorical_cols)}")
    print(f"  • Datetime columns: {len(datetime_cols)}")
    
    # Missing data
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    if len(missing_cols) > 0:
        missing_percent = (missing_cols / len(df) * 100).round(2)
        print(f"  • Columns with missing data: {len(missing_cols)}")
        print(f"  • Worst missing: {missing_percent.index[0]} ({missing_percent.iloc[0]}%)")
    else:
        print(f"  • No missing data")
    
    # Index information
    if isinstance(df.index, pd.DatetimeIndex):
        print(f"  • Time range: {df.index.min()} to {df.index.max()}")
        print(f"  • Duration: {(df.index.max() - df.index.min()).days} days")