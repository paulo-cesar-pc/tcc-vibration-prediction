"""
Machine Learning Models Module
==============================

This module provides comprehensive model training, evaluation, and management
for industrial vibration prediction tasks.

Key Components:
--------------
- BaseModel: Base class for all machine learning models
- ModelTrainer: Orchestrates model training and comparison
- ModelRegistry: Manages model configurations and instances
- ModelEvaluator: Evaluates model performance with detailed metrics

Features:
---------
- Multiple model types (Linear, Tree-based, Ensemble, Gradient Boosting)
- Hyperparameter optimization with grid/random search
- Cross-validation framework
- Model persistence and loading
- Comprehensive performance evaluation
- Overfitting detection and analysis
"""

from .base import BaseModel
from .trainer import ModelTrainer, train_multiple_models
from .registry import ModelRegistry

__all__ = [
    # Classes
    "BaseModel",
    "ModelTrainer", 
    "ModelRegistry",
    
    # Convenience functions
    "train_multiple_models",
]