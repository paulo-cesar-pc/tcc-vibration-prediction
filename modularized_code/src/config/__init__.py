"""
Configuration Module
==================

Centralized configuration management for the vibration prediction system.

This module provides:
- Application settings and constants
- Model configurations and hyperparameters
- Environment-specific configurations
- Validation of configuration parameters
"""

from .settings import (
    DATA_CONFIG,
    FEATURE_CONFIG,
    MODEL_CONFIG,
    EVALUATION_CONFIG,
    VISUALIZATION_CONFIG,
    get_config,
    validate_config
)

from .model_config import (
    DEFAULT_MODEL_CONFIGS,
    get_model_config,
    create_model_from_config
)

__all__ = [
    "DATA_CONFIG",
    "FEATURE_CONFIG", 
    "MODEL_CONFIG",
    "EVALUATION_CONFIG",
    "VISUALIZATION_CONFIG",
    "get_config",
    "validate_config",
    "DEFAULT_MODEL_CONFIGS",
    "get_model_config",
    "create_model_from_config",
]