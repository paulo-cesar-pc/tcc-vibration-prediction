"""
Model Evaluation Module
=======================

Comprehensive evaluation system for vibration prediction models.
Provides detailed metrics, visualizations, and business insights.

Key Components:
--------------
- ModelMetrics: Calculate comprehensive regression metrics
- ModelValidator: Validate model performance and robustness
- ModelVisualizer: Create detailed evaluation visualizations
- BusinessAnalyzer: Generate business insights and recommendations

Features:
---------
- Comprehensive regression metrics (R², RMSE, MAE, MAPE)
- Residual analysis and distribution testing
- Time series evaluation and drift detection
- Prediction accuracy assessment
- Business impact analysis
- Deployment readiness evaluation
"""

from .metrics import ModelMetrics, calculate_regression_metrics
from .validator import ModelValidator, validate_model_performance
from .visualizer import ModelVisualizer, create_evaluation_plots
from .analyzer import BusinessAnalyzer, generate_business_insights

__all__ = [
    # Classes
    "ModelMetrics",
    "ModelValidator", 
    "ModelVisualizer",
    "BusinessAnalyzer",
    
    # Convenience functions
    "calculate_regression_metrics",
    "validate_model_performance",
    "create_evaluation_plots",
    "generate_business_insights",
]