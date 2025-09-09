"""
Model Registry
==============

Centralized registry for managing model configurations, instances, and metadata.
Provides factory methods for creating models and organizing them by categories.
"""

import logging
from typing import Dict, List, Any, Optional, Type, Union
from pathlib import Path
import json

from config.model_config import (
    DEFAULT_MODEL_CONFIGS, 
    MODEL_CATEGORIES,
    get_model_config,
    create_model_from_config,
    list_available_models
)
from .base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralized registry for managing machine learning models.
    
    This class provides a unified interface for creating, configuring,
    and managing different types of models for vibration prediction.
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self.registered_models: Dict[str, BaseModel] = {}
        self.model_results: Dict[str, Dict[str, Any]] = {}
        logger.debug("Model registry initialized")
    
    def create_model(self, 
                     model_name: str, 
                     custom_params: Optional[Dict[str, Any]] = None,
                     model_alias: Optional[str] = None) -> BaseModel:
        """
        Create a model instance from the configuration registry.
        
        Parameters:
        -----------
        model_name : str
            Name of the model configuration to use
        custom_params : Optional[Dict[str, Any]], default=None
            Custom parameters to override defaults
        model_alias : Optional[str], default=None
            Custom alias for the model instance
            
        Returns:
        --------
        BaseModel
            Configured model instance
        """
        if model_name not in DEFAULT_MODEL_CONFIGS:
            available = ", ".join(DEFAULT_MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
        
        # Get configuration
        config = get_model_config(model_name)
        
        # Create estimator with custom parameters
        estimator = create_model_from_config(model_name, custom_params)
        
        # Create BaseModel wrapper
        alias = model_alias or model_name
        model = BaseModel(
            model_name=alias,
            estimator=estimator,
            requires_scaling=config["requires_scaling"],
            hyperparameters=custom_params or config["params"]
        )
        
        logger.info(f"Created model: {alias} (type: {model_name})")
        return model
    
    def create_models_by_category(self, 
                                 category: str,
                                 custom_params: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, BaseModel]:
        """
        Create all models in a specific category.
        
        Parameters:
        -----------
        category : str
            Model category name ('linear', 'tree_based', etc.)
        custom_params : Optional[Dict[str, Dict[str, Any]]], default=None
            Custom parameters for specific models
            
        Returns:
        --------
        Dict[str, BaseModel]
            Dictionary mapping model names to instances
        """
        if category not in MODEL_CATEGORIES:
            available = ", ".join(MODEL_CATEGORIES.keys())
            raise ValueError(f"Unknown category '{category}'. Available: {available}")
        
        models = {}
        custom_params = custom_params or {}
        
        for model_name in MODEL_CATEGORIES[category]:
            params = custom_params.get(model_name, None)
            try:
                models[model_name] = self.create_model(model_name, params)
                logger.info(f"Created {category} model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to create model {model_name}: {e}")
        
        logger.info(f"Created {len(models)} models in category '{category}'")
        return models
    
    def create_all_models(self, 
                         custom_params: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, BaseModel]:
        """
        Create all available models.
        
        Parameters:
        -----------
        custom_params : Optional[Dict[str, Dict[str, Any]]], default=None
            Custom parameters for specific models
            
        Returns:
        --------
        Dict[str, BaseModel]
            Dictionary mapping model names to instances
        """
        logger.info("Creating all available models")
        
        models = {}
        custom_params = custom_params or {}
        
        for model_name in DEFAULT_MODEL_CONFIGS.keys():
            params = custom_params.get(model_name, None)
            try:
                models[model_name] = self.create_model(model_name, params)
            except Exception as e:
                logger.warning(f"Failed to create model {model_name}: {e}")
        
        logger.info(f"Created {len(models)} total models")
        return models
    
    def register_model(self, model: BaseModel, replace: bool = False) -> None:
        """
        Register a model instance in the registry.
        
        Parameters:
        -----------
        model : BaseModel
            Model instance to register
        replace : bool, default=False
            Whether to replace existing model with same name
        """
        if model.model_name in self.registered_models and not replace:
            raise ValueError(f"Model '{model.model_name}' already registered. Use replace=True to override.")
        
        self.registered_models[model.model_name] = model
        logger.info(f"Registered model: {model.model_name}")
    
    def get_model(self, model_name: str) -> BaseModel:
        """
        Get a registered model by name.
        
        Parameters:
        -----------
        model_name : str
            Name of the registered model
            
        Returns:
        --------
        BaseModel
            The registered model instance
        """
        if model_name not in self.registered_models:
            available = ", ".join(self.registered_models.keys())
            raise ValueError(f"Model '{model_name}' not registered. Available: {available}")
        
        return self.registered_models[model_name]
    
    def list_registered_models(self) -> List[str]:
        """
        List names of all registered models.
        
        Returns:
        --------
        List[str]
            List of registered model names
        """
        return list(self.registered_models.keys())
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive summary for a registered model.
        
        Parameters:
        -----------
        model_name : str
            Name of the registered model
            
        Returns:
        --------
        Dict[str, Any]
            Model summary information
        """
        model = self.get_model(model_name)
        
        summary = {
            'model_name': model.model_name,
            'model_type': type(model.estimator).__name__,
            'requires_scaling': model.requires_scaling,
            'is_fitted': model.is_fitted_,
        }
        
        # Add training summary if fitted
        if model.is_fitted_:
            summary.update(model.get_training_summary())
        
        # Add stored results if available
        if model_name in self.model_results:
            summary.update(self.model_results[model_name])
        
        return summary
    
    def store_model_results(self, model_name: str, results: Dict[str, Any]) -> None:
        """
        Store evaluation results for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        results : Dict[str, Any]
            Results dictionary to store
        """
        if model_name not in self.registered_models:
            logger.warning(f"Storing results for unregistered model: {model_name}")
        
        self.model_results[model_name] = results
        logger.debug(f"Stored results for model: {model_name}")
    
    def get_model_results(self, model_name: str) -> Dict[str, Any]:
        """
        Get stored results for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
            
        Returns:
        --------
        Dict[str, Any]
            Stored results dictionary
        """
        return self.model_results.get(model_name, {})
    
    def get_leaderboard(self, metric: str = 'test_r2', ascending: bool = False) -> List[Dict[str, Any]]:
        """
        Get model leaderboard sorted by specified metric.
        
        Parameters:
        -----------
        metric : str, default='test_r2'
            Metric to sort by
        ascending : bool, default=False
            Sort order (False for descending)
            
        Returns:
        --------
        List[Dict[str, Any]]
            Sorted list of model summaries
        """
        leaderboard = []
        
        for model_name in self.registered_models.keys():
            try:
                summary = self.get_model_summary(model_name)
                if metric in summary:
                    leaderboard.append(summary)
            except Exception as e:
                logger.warning(f"Failed to get summary for {model_name}: {e}")
        
        # Sort by metric
        leaderboard.sort(key=lambda x: x.get(metric, float('-inf')), reverse=not ascending)
        
        return leaderboard
    
    def get_category_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary statistics by model category.
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Category-wise summaries
        """
        category_stats = {}
        
        for category, model_names in MODEL_CATEGORIES.items():
            registered_in_category = [name for name in model_names if name in self.registered_models]
            
            if not registered_in_category:
                continue
            
            # Compute category statistics
            test_r2_scores = []
            training_times = []
            
            for model_name in registered_in_category:
                try:
                    summary = self.get_model_summary(model_name)
                    if 'test_r2' in summary:
                        test_r2_scores.append(summary['test_r2'])
                    if 'training_time' in summary:
                        training_times.append(summary['training_time'])
                except Exception:
                    continue
            
            category_stats[category] = {
                'model_count': len(registered_in_category),
                'models': registered_in_category,
                'avg_test_r2': sum(test_r2_scores) / len(test_r2_scores) if test_r2_scores else 0,
                'best_test_r2': max(test_r2_scores) if test_r2_scores else 0,
                'avg_training_time': sum(training_times) / len(training_times) if training_times else 0,
            }
        
        return category_stats
    
    def save_registry(self, filepath: Union[str, Path]) -> None:
        """
        Save the model registry metadata to disk.
        
        Parameters:
        -----------
        filepath : Union[str, Path]
            Path where to save the registry
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        registry_data = {
            'registered_models': list(self.registered_models.keys()),
            'model_results': self.model_results,
            'category_summary': self.get_category_summary(),
            'leaderboard': self.get_leaderboard()
        }
        
        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2, default=str)
        
        logger.info(f"Registry saved to {filepath}")
    
    def clear_registry(self) -> None:
        """Clear all registered models and results."""
        self.registered_models.clear()
        self.model_results.clear()
        logger.info("Model registry cleared")
    
    def __len__(self) -> int:
        """Return number of registered models."""
        return len(self.registered_models)
    
    def __contains__(self, model_name: str) -> bool:
        """Check if model is registered."""
        return model_name in self.registered_models
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"ModelRegistry({len(self.registered_models)} models registered)"


# Global registry instance
_global_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """
    Get the global model registry instance.
    
    Returns:
    --------
    ModelRegistry
        Global registry instance
    """
    return _global_registry


def create_default_models(custom_params: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, BaseModel]:
    """
    Convenience function to create all default models.
    
    Parameters:
    -----------
    custom_params : Optional[Dict[str, Dict[str, Any]]], default=None
        Custom parameters for specific models
        
    Returns:
    --------
    Dict[str, BaseModel]
        Dictionary of created models
    """
    registry = get_registry()
    return registry.create_all_models(custom_params)


def get_model_configurations() -> Dict[str, str]:
    """
    Get available model configurations.
    
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping model names to descriptions
    """
    return list_available_models()