"""
Industrial Vibration Prediction Package
=====================================

A modular package for predicting industrial vibration in roller mills using machine learning.

Modules:
--------
- config: Configuration and settings management
- data: Data loading, cleaning, and preprocessing
- features: Feature engineering and selection
- models: Machine learning model implementations
- evaluation: Model evaluation and metrics
- utils: Utility functions and helpers
- pipeline: End-to-end pipeline orchestration

Author: Paulo Cesar da Silva Junior
Institution: Universidade Federal do Ceará (UFC)
"""

__version__ = "1.0.0"
__author__ = "Paulo Cesar da Silva Junior"
__email__ = "paulo.cesar@example.com"

# Package-level imports for convenient access
from .config import settings
from .utils import helpers

__all__ = [
    "settings",
    "helpers",
]