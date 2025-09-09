"""
Model Trainer
=============

Comprehensive model training orchestrator for vibration prediction.
Handles training multiple models, hyperparameter optimization, and comparison.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time
from pathlib import Path

from config.settings import get_settings
from config.model_config import get_hyperparameter_space
from utils.helpers import Timer
from .base import BaseModel
from .registry import ModelRegistry, get_registry

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelTrainer:
    """
    Orchestrates training of multiple machine learning models.
    
    This class handles the complete model training workflow including:
    - Model configuration and setup
    - Feature scaling for models that require it
    - Hyperparameter optimization
    - Cross-validation
    - Performance comparison and ranking
    """
    
    def __init__(self, 
                 registry: Optional[ModelRegistry] = None,
                 random_state: int = 42):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        registry : Optional[ModelRegistry], default=None
            Model registry to use. If None, uses global registry.
        random_state : int, default=42
            Random state for reproducibility
        """
        self.registry = registry or get_registry()
        self.random_state = random_state
        
        # Training data storage
        self.X_train_ = None
        self.X_test_ = None
        self.y_train_ = None
        self.y_test_ = None
        self.feature_names_ = None
        
        # Scalers for models that need scaling
        self.scaler_ = None
        self.X_train_scaled_ = None
        self.X_test_scaled_ = None
        
        # Training results
        self.training_results_ = []
        self.best_model_ = None
        
        logger.debug("ModelTrainer initialized")
    
    def setup_data(self, 
                   X_train: Union[pd.DataFrame, np.ndarray],
                   y_train: Union[pd.Series, np.ndarray],
                   X_test: Union[pd.DataFrame, np.ndarray],
                   y_test: Union[pd.Series, np.ndarray]) -> 'ModelTrainer':
        """
        Setup training and test data.
        
        Parameters:
        -----------
        X_train : Union[pd.DataFrame, np.ndarray]
            Training feature matrix
        y_train : Union[pd.Series, np.ndarray]
            Training target variable
        X_test : Union[pd.DataFrame, np.ndarray]
            Test feature matrix  
        y_test : Union[pd.Series, np.ndarray]
            Test target variable
            
        Returns:
        --------
        ModelTrainer
            Self for method chaining
        """
        logger.info("Setting up training data")
        
        # Store data
        self.X_train_ = X_train.copy() if hasattr(X_train, 'copy') else X_train
        self.X_test_ = X_test.copy() if hasattr(X_test, 'copy') else X_test
        self.y_train_ = y_train.copy() if hasattr(y_train, 'copy') else y_train
        self.y_test_ = y_test.copy() if hasattr(y_test, 'copy') else y_test
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names_ = list(X_train.columns)
        
        # Prepare scaled data for models that require it
        self.scaler_ = StandardScaler()
        self.X_train_scaled_ = self.scaler_.fit_transform(X_train)
        self.X_test_scaled_ = self.scaler_.transform(X_test)
        
        logger.info(f"Data setup complete - Train: {X_train.shape}, Test: {X_test.shape}")
        
        return self
    
    def train_model(self, 
                    model_name: str,
                    custom_params: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        Train a single model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model configuration to train
        custom_params : Optional[Dict[str, Any]], default=None
            Custom parameters to override defaults
            
        Returns:
        --------
        BaseModel
            Trained model instance
        """
        if self.X_train_ is None:
            raise ValueError("Training data not setup. Call setup_data() first.")
        
        logger.info(f"Training model: {model_name}")
        
        # Create model instance
        model = self.registry.create_model(model_name, custom_params)
        
        # Choose appropriate data (scaled or unscaled)
        if model.requires_scaling:
            X_train_data = self.X_train_scaled_
            X_test_data = self.X_test_scaled_
        else:
            X_train_data = self.X_train_
            X_test_data = self.X_test_
        
        # Train the model
        model.fit(X_train_data, self.y_train_, X_test_data, self.y_test_)
        
        # Register the trained model
        self.registry.register_model(model, replace=True)
        
        logger.info(f"Model {model_name} trained successfully")
        
        return model
    
    def train_multiple_models(self, 
                             model_names: Optional[List[str]] = None,
                             custom_params: Optional[Dict[str, Dict[str, Any]]] = None,
                             include_scaling_variants: bool = True) -> Dict[str, BaseModel]:
        """
        Train multiple models and compare their performance.
        
        Parameters:
        -----------
        model_names : Optional[List[str]], default=None
            List of model names to train. If None, trains all available models.
        custom_params : Optional[Dict[str, Dict[str, Any]]], default=None
            Custom parameters for specific models
        include_scaling_variants : bool, default=True
            Whether to include scaled versions of models that don't require scaling
            
        Returns:
        --------
        Dict[str, BaseModel]
            Dictionary of trained models
        """
        if self.X_train_ is None:
            raise ValueError("Training data not setup. Call setup_data() first.")
        
        # Use all available models if not specified
        if model_names is None:
            from config.model_config import DEFAULT_MODEL_CONFIGS
            model_names = list(DEFAULT_MODEL_CONFIGS.keys())
        
        logger.info(f"Training {len(model_names)} models")
        
        trained_models = {}
        custom_params = custom_params or {}
        
        # Train each model
        for model_name in model_names:
            try:
                # Train normal version
                params = custom_params.get(model_name, None)
                model = self.train_model(model_name, params)
                trained_models[model_name] = model
                
                # Train scaled version if requested and model doesn't require scaling
                if include_scaling_variants and not model.requires_scaling:
                    scaled_name = f"{model_name}_scaled"
                    scaled_model = self.registry.create_model(model_name, params, scaled_name)
                    
                    # Train on scaled data
                    scaled_model.fit(self.X_train_scaled_, self.y_train_, 
                                   self.X_test_scaled_, self.y_test_)
                    
                    # Register scaled version
                    self.registry.register_model(scaled_model, replace=True)
                    trained_models[scaled_name] = scaled_model
                    
            except Exception as e:
                logger.error(f"Failed to train model {model_name}: {e}")
                continue
        
        logger.info(f"Successfully trained {len(trained_models)} models")
        
        # Store training results
        self._compile_training_results()
        
        return trained_models
    
    def optimize_hyperparameters(self,
                                model_name: str,
                                search_type: str = 'grid',
                                n_iter: int = 20,
                                cv: int = 3,
                                n_jobs: int = -1) -> BaseModel:
        """
        Optimize hyperparameters for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to optimize
        search_type : str, default='grid'
            Type of search ('grid' or 'random')
        n_iter : int, default=20
            Number of iterations for random search
        cv : int, default=3
            Number of cross-validation folds
        n_jobs : int, default=-1
            Number of parallel jobs
            
        Returns:
        --------
        BaseModel
            Model with optimized hyperparameters
        """
        if self.X_train_ is None:
            raise ValueError("Training data not setup. Call setup_data() first.")
        
        logger.info(f"Optimizing hyperparameters for {model_name} using {search_type} search")
        
        # Get hyperparameter space
        param_space = get_hyperparameter_space(model_name)
        
        if not param_space:
            logger.warning(f"No hyperparameter space defined for {model_name}")
            return self.train_model(model_name)
        
        # Create base model
        base_model = self.registry.create_model(model_name)
        
        # Choose data based on scaling requirements
        if base_model.requires_scaling:
            X_data = self.X_train_scaled_
        else:
            X_data = self.X_train_
        
        # Setup search
        if search_type == 'grid':
            search = GridSearchCV(
                estimator=base_model.estimator,
                param_grid=param_space,
                cv=cv,
                scoring='r2',
                n_jobs=n_jobs,
                verbose=1
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                estimator=base_model.estimator,
                param_distributions=param_space,
                n_iter=n_iter,
                cv=cv,
                scoring='r2',
                n_jobs=n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown search_type: {search_type}")
        
        # Perform search
        with Timer(f"{search_type} search for {model_name}"):
            search.fit(X_data, self.y_train_)
        
        # Create optimized model
        optimized_name = f"{model_name}_optimized"
        optimized_model = BaseModel(
            model_name=optimized_name,
            estimator=search.best_estimator_,
            requires_scaling=base_model.requires_scaling,
            hyperparameters=search.best_params_
        )
        
        # Train on full data with optimized parameters
        if base_model.requires_scaling:
            X_test_data = self.X_test_scaled_
        else:
            X_test_data = self.X_test_
        
        optimized_model.fit(X_data, self.y_train_, X_test_data, self.y_test_)
        
        # Store optimization results
        optimization_results = {
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'cv_results': search.cv_results_,
            'search_type': search_type
        }
        
        self.registry.register_model(optimized_model, replace=True)
        self.registry.store_model_results(optimized_name, optimization_results)
        
        logger.info(f"Hyperparameter optimization complete for {model_name}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        logger.info(f"Best parameters: {search.best_params_}")
        
        return optimized_model
    
    def _compile_training_results(self) -> None:
        """Compile comprehensive training results."""
        results = []
        
        for model_name in self.registry.list_registered_models():
            try:
                summary = self.registry.get_model_summary(model_name)
                
                # Add rank information
                summary['model_alias'] = model_name
                results.append(summary)
                
            except Exception as e:
                logger.warning(f"Failed to compile results for {model_name}: {e}")
        
        # Sort by test R² score
        results.sort(key=lambda x: x.get('test_r2', 0), reverse=True)
        
        # Add ranks
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        self.training_results_ = results
        
        # Identify best model
        if results:
            self.best_model_ = results[0]['model_alias']
            logger.info(f"Best performing model: {self.best_model_} "
                       f"(R² = {results[0].get('test_r2', 0):.4f})")
    
    def get_training_results(self, as_dataframe: bool = True) -> Union[List[Dict], pd.DataFrame]:
        """
        Get comprehensive training results.
        
        Parameters:
        -----------
        as_dataframe : bool, default=True
            Whether to return results as DataFrame
            
        Returns:
        --------
        Union[List[Dict], pd.DataFrame]
            Training results
        """
        if not self.training_results_:
            self._compile_training_results()
        
        if as_dataframe:
            return pd.DataFrame(self.training_results_)
        
        return self.training_results_
    
    def get_model_leaderboard(self, metric: str = 'test_r2') -> pd.DataFrame:
        """
        Get model leaderboard sorted by specified metric.
        
        Parameters:
        -----------
        metric : str, default='test_r2'
            Metric to sort by
            
        Returns:
        --------
        pd.DataFrame
            Leaderboard DataFrame
        """
        results_df = self.get_training_results(as_dataframe=True)
        
        # Select relevant columns for leaderboard
        leaderboard_cols = [
            'rank', 'model_alias', 'model_type', 
            'test_r2', 'test_rmse', 'overfitting', 
            'training_time', 'n_features'
        ]
        
        # Filter available columns
        available_cols = [col for col in leaderboard_cols if col in results_df.columns]
        leaderboard = results_df[available_cols].copy()
        
        # Sort by metric if different from test_r2
        if metric != 'test_r2' and metric in leaderboard.columns:
            ascending = True if 'rmse' in metric or 'mae' in metric else False
            leaderboard = leaderboard.sort_values(metric, ascending=ascending).reset_index(drop=True)
            leaderboard['rank'] = range(1, len(leaderboard) + 1)
        
        return leaderboard
    
    def get_best_model(self, metric: str = 'test_r2') -> BaseModel:
        """
        Get the best performing model.
        
        Parameters:
        -----------
        metric : str, default='test_r2'
            Metric to use for determining best model
            
        Returns:
        --------
        BaseModel
            Best performing model instance
        """
        leaderboard = self.get_model_leaderboard(metric)
        
        if leaderboard.empty:
            raise ValueError("No trained models available")
        
        best_model_name = leaderboard.iloc[0]['model_alias']
        return self.registry.get_model(best_model_name)
    
    def save_training_results(self, filepath: Union[str, Path]) -> None:
        """
        Save training results to file.
        
        Parameters:
        -----------
        filepath : Union[str, Path]
            Path to save results
        """
        results_df = self.get_training_results(as_dataframe=True)
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix.lower() == '.csv':
            results_df.to_csv(filepath, index=False)
        else:
            results_df.to_json(filepath, orient='records', indent=2)
        
        logger.info(f"Training results saved to {filepath}")


def train_multiple_models(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    model_names: Optional[List[str]] = None,
    custom_params: Optional[Dict[str, Dict[str, Any]]] = None,
    include_scaling_variants: bool = True,
    random_state: int = 42
) -> Tuple[Dict[str, BaseModel], pd.DataFrame]:
    """
    Convenience function to train multiple models.
    
    Parameters:
    -----------
    X_train : Union[pd.DataFrame, np.ndarray]
        Training feature matrix
    y_train : Union[pd.Series, np.ndarray]
        Training target variable
    X_test : Union[pd.DataFrame, np.ndarray]
        Test feature matrix
    y_test : Union[pd.Series, np.ndarray]
        Test target variable
    model_names : Optional[List[str]], default=None
        List of model names to train
    custom_params : Optional[Dict[str, Dict[str, Any]]], default=None
        Custom parameters for models
    include_scaling_variants : bool, default=True
        Whether to include scaled variants
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    Tuple[Dict[str, BaseModel], pd.DataFrame]
        Trained models and results DataFrame
    """
    # Initialize trainer
    trainer = ModelTrainer(random_state=random_state)
    trainer.setup_data(X_train, y_train, X_test, y_test)
    
    # Train models
    models = trainer.train_multiple_models(
        model_names=model_names,
        custom_params=custom_params,
        include_scaling_variants=include_scaling_variants
    )
    
    # Get results
    results_df = trainer.get_training_results(as_dataframe=True)
    
    logger.info(f"Model training complete:")
    logger.info(f"  • Models trained: {len(models)}")
    logger.info(f"  • Best model: {trainer.best_model_}")
    
    if not results_df.empty:
        best_r2 = results_df.iloc[0].get('test_r2', 0)
        logger.info(f"  • Best R²: {best_r2:.4f}")
    
    return models, results_df