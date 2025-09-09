"""
Feature Engineering Module
==========================

Advanced feature engineering and selection for industrial vibration prediction.

This module provides:
- Feature engineering with rolling statistics and temporal features
- Feature importance analysis using multiple methods
- Feature selection algorithms (statistical, tree-based, correlation-based)
- Data leakage prevention mechanisms

Key Components:
--------------
- FeatureEngineer: Create engineered features from raw data
- FeatureImportanceAnalyzer: Analyze feature importance using ML models
- FeatureSelector: Select optimal features using various criteria
"""

from .engineer import FeatureEngineer, engineer_features
from .importance import FeatureImportanceAnalyzer, analyze_feature_importance
from .selector import FeatureSelector, select_features_for_modeling

__all__ = [
    # Classes
    "FeatureEngineer",
    "FeatureImportanceAnalyzer", 
    "FeatureSelector",
    
    # Convenience functions
    "engineer_features",
    "analyze_feature_importance",
    "select_features_for_modeling",
]