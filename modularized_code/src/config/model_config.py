"""
Model Configuration and Registry
===============================

Configuration settings for different machine learning models used in vibration prediction.
This module provides standardized configurations and factory methods for model creation.
"""

from typing import Dict, Any, Type, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Default model configurations optimized for industrial vibration prediction
DEFAULT_MODEL_CONFIGS = {
    "linear_regression": {
        "class": LinearRegression,
        "params": {},
        "requires_scaling": True,
        "description": "Simple linear regression baseline",
        "hyperparameter_space": {},
    },
    
    "ridge_regression": {
        "class": Ridge,
        "params": {
            "alpha": 1.0,
            "random_state": 42,
        },
        "requires_scaling": True,
        "description": "Ridge regression with L2 regularization",
        "hyperparameter_space": {
            "alpha": [0.1, 1.0, 10.0, 100.0],
        },
    },
    
    "lasso_regression": {
        "class": Lasso,
        "params": {
            "alpha": 1.0,
            "random_state": 42,
            "max_iter": 2000,
        },
        "requires_scaling": True,
        "description": "Lasso regression with L1 regularization and feature selection",
        "hyperparameter_space": {
            "alpha": [0.1, 1.0, 10.0],
        },
    },
    
    "huber_regressor": {
        "class": HuberRegressor,
        "params": {
            "epsilon": 1.35,
            "alpha": 0.001,
            "max_iter": 200,
        },
        "requires_scaling": True,
        "description": "Robust regression using Huber loss (handles outliers)",
        "hyperparameter_space": {
            "epsilon": [1.1, 1.35, 1.5],
            "alpha": [0.0001, 0.001, 0.01],
        },
    },
    
    "decision_tree": {
        "class": DecisionTreeRegressor,
        "params": {
            "max_depth": 8,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
        },
        "requires_scaling": False,
        "description": "Decision tree regressor with controlled complexity",
        "hyperparameter_space": {
            "max_depth": [6, 8, 10, 12],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [2, 5, 10],
        },
    },
    
    "random_forest": {
        "class": RandomForestRegressor,
        "params": {
            "n_estimators": 50,
            "max_depth": 8,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1,
        },
        "requires_scaling": False,
        "description": "Random Forest optimized for industrial vibration data",
        "hyperparameter_space": {
            "n_estimators": [30, 50, 100],
            "max_depth": [6, 8, 10],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [2, 5, 10],
        },
    },
    
    "gradient_boosting": {
        "class": GradientBoostingRegressor,
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 4,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
        },
        "requires_scaling": False,
        "description": "Gradient boosting with controlled overfitting",
        "hyperparameter_space": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.05, 0.1, 0.15],
            "max_depth": [3, 4, 5],
        },
    },
    
    "knn_regressor": {
        "class": KNeighborsRegressor,
        "params": {
            "n_neighbors": 5,
            "weights": "distance",
            "metric": "euclidean",
        },
        "requires_scaling": True,
        "description": "K-Nearest Neighbors with distance weighting",
        "hyperparameter_space": {
            "n_neighbors": [3, 5, 7, 10],
            "weights": ["uniform", "distance"],
        },
    },
}

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    DEFAULT_MODEL_CONFIGS["xgboost"] = {
        "class": xgb.XGBRegressor,
        "params": {
            "n_estimators": 50,
            "learning_rate": 0.05,
            "max_depth": 4,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbosity": 0,
        },
        "requires_scaling": False,
        "description": "XGBoost regressor tuned for vibration prediction",
        "hyperparameter_space": {
            "n_estimators": [30, 50, 75],
            "learning_rate": [0.03, 0.05, 0.08],
            "max_depth": [3, 4, 5],
            "reg_alpha": [0.0, 0.1, 0.2],
            "reg_lambda": [0.0, 0.1, 0.2],
        },
    }

# Add CatBoost if available
if CATBOOST_AVAILABLE:
    DEFAULT_MODEL_CONFIGS["catboost"] = {
        "class": cb.CatBoostRegressor,
        "params": {
            "iterations": 75,
            "learning_rate": 0.08,
            "depth": 4,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "verbose": False,
        },
        "requires_scaling": False,
        "description": "CatBoost regressor for industrial data",
        "hyperparameter_space": {
            "iterations": [50, 75, 100],
            "learning_rate": [0.05, 0.08, 0.1],
            "depth": [3, 4, 5],
            "l2_leaf_reg": [1, 3, 5],
        },
    }

# Model categories for organized selection
MODEL_CATEGORIES = {
    "linear": ["linear_regression", "ridge_regression", "lasso_regression", "huber_regressor"],
    "tree_based": ["decision_tree", "random_forest", "gradient_boosting"],
    "distance_based": ["knn_regressor"],
    "gradient_boosting": [],  # Filled dynamically based on availability
}

# Add advanced models to categories if available
if XGBOOST_AVAILABLE:
    MODEL_CATEGORIES["gradient_boosting"].append("xgboost")
if CATBOOST_AVAILABLE:
    MODEL_CATEGORIES["gradient_boosting"].append("catboost")

# If no advanced boosting available, add gradient_boosting to tree_based
if not MODEL_CATEGORIES["gradient_boosting"]:
    MODEL_CATEGORIES["tree_based"].append("gradient_boosting")
    del MODEL_CATEGORIES["gradient_boosting"]


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
        
    Returns:
    --------
    Dict[str, Any]
        Model configuration dictionary
        
    Raises:
    -------
    ValueError
        If model_name is not recognized
    """
    if model_name not in DEFAULT_MODEL_CONFIGS:
        available = ", ".join(DEFAULT_MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    return DEFAULT_MODEL_CONFIGS[model_name].copy()


def get_models_by_category(category: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all models in a specific category.
    
    Parameters:
    -----------
    category : str
        Model category name
        
    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary of model configurations in the category
    """
    if category not in MODEL_CATEGORIES:
        available = ", ".join(MODEL_CATEGORIES.keys())
        raise ValueError(f"Unknown category '{category}'. Available: {available}")
    
    return {
        model_name: get_model_config(model_name)
        for model_name in MODEL_CATEGORIES[category]
    }


def create_model_from_config(model_name: str, custom_params: Optional[Dict[str, Any]] = None):
    """
    Create a model instance from configuration.
    
    Parameters:
    -----------
    model_name : str
        Name of the model to create
    custom_params : Optional[Dict[str, Any]]
        Custom parameters to override defaults
        
    Returns:
    --------
    sklearn estimator or Pipeline
        Configured model instance
    """
    config = get_model_config(model_name)
    
    # Merge custom parameters
    params = config["params"].copy()
    if custom_params:
        params.update(custom_params)
    
    # Create model instance
    model = config["class"](**params)
    
    # Wrap in pipeline with scaler if required
    if config["requires_scaling"]:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
    
    return model


def list_available_models() -> Dict[str, str]:
    """
    List all available models with descriptions.
    
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping model names to descriptions
    """
    return {
        name: config["description"] 
        for name, config in DEFAULT_MODEL_CONFIGS.items()
    }


def validate_model_config(config: Dict[str, Any]) -> bool:
    """
    Validate model configuration structure.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Model configuration to validate
        
    Returns:
    --------
    bool
        True if configuration is valid
        
    Raises:
    -------
    ValueError
        If configuration is invalid
    """
    required_keys = ["class", "params", "requires_scaling", "description"]
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required keys in model config: {missing_keys}")
    
    if not hasattr(config["class"], "fit"):
        raise ValueError("Model class must have a 'fit' method")
    
    if not hasattr(config["class"], "predict"):
        raise ValueError("Model class must have a 'predict' method")
    
    return True


def get_hyperparameter_space(model_name: str) -> Dict[str, list]:
    """
    Get hyperparameter search space for a model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
        
    Returns:
    --------
    Dict[str, list]
        Hyperparameter search space
    """
    config = get_model_config(model_name)
    return config.get("hyperparameter_space", {})