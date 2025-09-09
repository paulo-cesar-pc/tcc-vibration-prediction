"""
Data Processing Module
=====================

Comprehensive data handling for industrial vibration prediction.

This module provides:
- Data loading from CSV files with validation
- Data cleaning and quality assessment
- Data preprocessing and resampling
- Data transformation utilities

Key Components:
--------------
- DataLoader: Load and combine CSV files
- DataCleaner: Clean and validate industrial data
- DataPreprocessor: Resample and prepare data for ML
"""

from .loader import DataLoader, load_data
from .cleaner import DataCleaner, clean_data
from .preprocessor import DataPreprocessor, resample_aggregate, prepare_model_data

__all__ = [
    # Classes
    "DataLoader",
    "DataCleaner", 
    "DataPreprocessor",
    
    # Convenience functions
    "load_data",
    "clean_data",
    "resample_aggregate",
    "prepare_model_data",
]