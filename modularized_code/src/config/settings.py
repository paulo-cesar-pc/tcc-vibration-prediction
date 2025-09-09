"""
Application Settings and Configuration
====================================

Centralized configuration for the industrial vibration prediction system.
All configuration parameters are defined here for easy maintenance and modification.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Data Configuration
DATA_CONFIG = {
    # Data loading
    "data_path": "full_data/",
    "file_pattern": "*.csv",
    "timestamp_column": "Timestamps",
    "timestamp_format": "%d/%m/%Y %H:%M:%S",
    
    # Data cleaning
    "vibration_min_value": 0.0,
    "vibration_max_value": 12.0,
    "missing_threshold": 0.5,  # Drop columns with >50% missing
    "outlier_percentile_low": 0.05,
    "outlier_percentile_high": 0.95,
    
    # Data resampling
    "original_frequency": "30S",
    "target_frequency": "5T",  # 5 minutes
    "aggregation_method": "mean",
    
    # Data splits
    "test_size": 0.2,
    "validation_size": 0.15,
    "random_state": 42,
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    # Rolling window features
    "rolling_windows": [3, 6, 12],  # 15min, 30min, 1hr for 5-minute data
    "rolling_statistics": ["mean", "std"],
    
    # Feature patterns for key variables
    "key_patterns": ["POWER", "PRESSURE", "CURRENT", "FLOW", "TEMPERATURE"],
    "max_features_per_pattern": 2,
    
    # Temporal features
    "temporal_features": ["hour", "day_of_week", "month"],
    
    # Feature filtering
    "correlation_threshold": 0.01,
    "variance_threshold": 0.001,
    "feature_selection_methods": ["importance", "statistical", "correlation"],
}

# Model Configuration
MODEL_CONFIG = {
    # Training settings
    "cv_folds": 5,
    "scoring_metric": "r2",
    "n_jobs": -1,
    "verbose": 1,
    
    # Feature selection
    "max_features": 50,
    "importance_threshold": 0.8,  # Cumulative importance
    "statistical_k": 20,
    
    # Model validation
    "validation_metrics": ["r2_score", "rmse", "mae", "mape"],
    "overfitting_threshold": 0.1,  # Train R² - Test R²
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    # Metrics
    "primary_metric": "r2_score",
    "regression_metrics": ["r2_score", "rmse", "mae", "mape", "explained_variance"],
    
    # Residual analysis
    "residual_bins": 50,
    "residual_threshold_percentiles": [0.05, 0.95],
    
    # Business metrics
    "accuracy_thresholds": [0.001, 0.002, 0.005, 0.01],  # mm/s
    "performance_categories": {
        "excellent": 0.9,
        "good": 0.8,
        "fair": 0.6,
    },
    
    # Time series analysis
    "chunk_size": 2000,
    "trend_analysis_window": 50,
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    # Plot settings
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "default",
    "color_palette": "husl",
    
    # Time series plots
    "max_points_plot": 10000,
    "line_width": 1.0,
    "alpha": 0.7,
    
    # Statistical plots
    "hist_bins": 50,
    "scatter_alpha": 0.6,
    "scatter_size": 10,
    
    # Feature importance plots
    "top_features_display": 20,
    "importance_threshold_display": 0.01,
    
    # Output settings
    "save_format": "png",
    "save_dpi": 300,
    "bbox_inches": "tight",
}

# System Configuration
SYSTEM_CONFIG = {
    "random_seed": 42,
    "parallel_jobs": -1,
    "memory_limit": "8GB",
    "cache_size": 1000,
    
    # Logging
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    
    # Warnings
    "suppress_warnings": True,
    "warning_categories": ["sklearn", "pandas", "matplotlib"],
}


def get_config(config_name: str) -> Dict[str, Any]:
    """
    Get configuration dictionary by name.
    
    Parameters:
    -----------
    config_name : str
        Name of the configuration to retrieve
        
    Returns:
    --------
    Dict[str, Any]
        Configuration dictionary
        
    Raises:
    -------
    ValueError
        If config_name is not recognized
    """
    configs = {
        "data": DATA_CONFIG,
        "features": FEATURE_CONFIG,
        "models": MODEL_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "system": SYSTEM_CONFIG,
    }
    
    if config_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return configs[config_name].copy()


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that required keys exist in configuration.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary to validate
    required_keys : list
        List of required keys
        
    Returns:
    --------
    bool
        True if all required keys exist
        
    Raises:
    -------
    KeyError
        If required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise KeyError(f"Missing required configuration keys: {missing_keys}")
    
    return True


def update_config(config_name: str, updates: Dict[str, Any]) -> None:
    """
    Update configuration with new values.
    
    Parameters:
    -----------
    config_name : str
        Name of configuration to update
    updates : Dict[str, Any]
        Dictionary of updates to apply
    """
    config_map = {
        "data": DATA_CONFIG,
        "features": FEATURE_CONFIG,
        "models": MODEL_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "system": SYSTEM_CONFIG,
    }
    
    if config_name in config_map:
        config_map[config_name].update(updates)
    else:
        available = ", ".join(config_map.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")


# Environment-specific overrides
def load_environment_config(env: str = "development") -> None:
    """
    Load environment-specific configuration overrides.
    
    Parameters:
    -----------
    env : str
        Environment name (development, testing, production)
    """
    env_configs = {
        "development": {
            "data": {"random_state": 42},
            "models": {"n_jobs": 1, "verbose": 2},
            "system": {"log_level": "DEBUG"},
        },
        "testing": {
            "data": {"test_size": 0.1},
            "models": {"cv_folds": 3},
            "system": {"suppress_warnings": True},
        },
        "production": {
            "models": {"verbose": 0},
            "system": {"log_level": "WARNING", "suppress_warnings": True},
        }
    }
    
    if env in env_configs:
        for config_name, updates in env_configs[env].items():
            update_config(config_name, updates)


# Initialize default environment
ENV = os.getenv("VIBRATION_ENV", "development")
load_environment_config(ENV)


class Settings:
    """Settings object providing access to all configurations."""
    
    def __init__(self):
        self.data = DATA_CONFIG
        self.features = FEATURE_CONFIG
        self.models = MODEL_CONFIG
        self.evaluation = EVALUATION_CONFIG
        self.visualization = VISUALIZATION_CONFIG
        self.system = SYSTEM_CONFIG
        self.env = ENV
        
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get configuration by name."""
        return get_config(config_name)
        
    def update(self, config_name: str, updates: Dict[str, Any]) -> None:
        """Update configuration."""
        update_config(config_name, updates)


_settings_instance = None

def get_settings() -> Settings:
    """Get singleton settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance