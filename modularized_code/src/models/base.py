"""
Base Model Class
================

Provides a unified interface for all machine learning models used in
vibration prediction. Handles common functionality like training, prediction,
evaluation, and persistence.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import time

from config.settings import get_settings
from utils.helpers import Timer

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseModel:
    """
    Base class for all machine learning models in the vibration prediction system.
    
    This class provides a consistent interface for model training, prediction,
    evaluation, and persistence across different algorithms.
    """
    
    def __init__(self, 
                 model_name: str,
                 estimator: BaseEstimator,
                 requires_scaling: bool = False,
                 hyperparameters: Optional[Dict[str, Any]] = None,
                 random_state: int = 42):
        """
        Initialize the base model.
        
        Parameters:
        -----------
        model_name : str
            Name identifier for the model
        estimator : BaseEstimator
            The sklearn-compatible estimator
        requires_scaling : bool, default=False
            Whether the model requires feature scaling
        hyperparameters : Optional[Dict[str, Any]], default=None
            Model hyperparameters
        random_state : int, default=42
            Random state for reproducibility
        """
        self.model_name = model_name
        self.estimator = estimator
        self.requires_scaling = requires_scaling
        self.hyperparameters = hyperparameters or {}
        self.random_state = random_state
        
        # Training metadata
        self.is_fitted_ = False
        self.training_time_ = None
        self.feature_names_ = None
        self.n_features_ = None
        self.training_samples_ = None
        
        # Performance metrics
        self.train_metrics_ = {}
        self.test_metrics_ = {}
        self.cv_scores_ = None
        
        logger.debug(f"Initialized {model_name} model")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
            X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_test: Optional[Union[pd.Series, np.ndarray]] = None) -> 'BaseModel':
        """
        Train the model on the provided data.
        
        Parameters:
        -----------
        X : Union[pd.DataFrame, np.ndarray]
            Training feature matrix
        y : Union[pd.Series, np.ndarray]
            Training target variable
        X_test : Optional[Union[pd.DataFrame, np.ndarray]], default=None
            Test feature matrix for evaluation
        y_test : Optional[Union[pd.Series, np.ndarray]], default=None
            Test target variable for evaluation
            
        Returns:
        --------
        BaseModel
            Self for method chaining
        """
        logger.info(f"Training {self.model_name} model")
        
        # Store training metadata
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        self.n_features_ = X.shape[1]
        self.training_samples_ = X.shape[0]
        
        # Train the model with timing
        with Timer(f"{self.model_name} training") as timer:
            self.estimator.fit(X, y)
        
        self.training_time_ = timer.elapsed_time
        self.is_fitted_ = True
        
        # Compute training metrics
        train_pred = self.estimator.predict(X)
        self.train_metrics_ = self._compute_metrics(y, train_pred, prefix="train")
        
        # Compute test metrics if provided
        if X_test is not None and y_test is not None:
            test_pred = self.estimator.predict(X_test)
            self.test_metrics_ = self._compute_metrics(y_test, test_pred, prefix="test")
        
        logger.info(f"{self.model_name} training complete - "
                   f"Train R²: {self.train_metrics_.get('train_r2', 0):.4f}, "
                   f"Test R²: {self.test_metrics_.get('test_r2', 0):.4f}")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix for prediction
            
        Returns:
        --------
        np.ndarray
            Model predictions
        """
        if not self.is_fitted_:
            raise ValueError(f"Model {self.model_name} must be fitted before making predictions")
        
        return self.estimator.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities (for classifiers that support it).
        
        Parameters:
        -----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix for prediction
            
        Returns:
        --------
        np.ndarray
            Class probabilities
        """
        if not self.is_fitted_:
            raise ValueError(f"Model {self.model_name} must be fitted before making predictions")
        
        if not hasattr(self.estimator, 'predict_proba'):
            raise AttributeError(f"Model {self.model_name} does not support probability prediction")
        
        return self.estimator.predict_proba(X)
    
    def cross_validate(self, X: Union[pd.DataFrame, np.ndarray], 
                      y: Union[pd.Series, np.ndarray],
                      cv: int = 5, scoring: str = 'r2') -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Parameters:
        -----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix
        y : Union[pd.Series, np.ndarray]
            Target variable
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='r2'
            Scoring metric for cross-validation
            
        Returns:
        --------
        Dict[str, float]
            Cross-validation results
        """
        logger.info(f"Running {cv}-fold cross-validation for {self.model_name}")
        
        scores = cross_val_score(self.estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        cv_results = {
            f'cv_{scoring}_mean': scores.mean(),
            f'cv_{scoring}_std': scores.std(),
            f'cv_{scoring}_scores': scores
        }
        
        self.cv_scores_ = cv_results
        
        logger.info(f"Cross-validation complete - "
                   f"Mean {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return cv_results
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if supported by the model.
        
        Returns:
        --------
        Optional[Dict[str, float]]
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted_:
            raise ValueError(f"Model {self.model_name} must be fitted before extracting feature importance")
        
        if hasattr(self.estimator, 'feature_importances_'):
            importances = self.estimator.feature_importances_
            if self.feature_names_:
                return dict(zip(self.feature_names_, importances))
            else:
                return {f'feature_{i}': imp for i, imp in enumerate(importances)}
        
        elif hasattr(self.estimator, 'coef_'):
            coefficients = np.abs(self.estimator.coef_)
            if self.feature_names_:
                return dict(zip(self.feature_names_, coefficients))
            else:
                return {f'feature_{i}': coef for i, coef in enumerate(coefficients)}
        
        else:
            logger.warning(f"Model {self.model_name} does not support feature importance extraction")
            return None
    
    def _compute_metrics(self, y_true: Union[pd.Series, np.ndarray], 
                        y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """
        Compute comprehensive regression metrics.
        
        Parameters:
        -----------
        y_true : Union[pd.Series, np.ndarray]
            True target values
        y_pred : np.ndarray
            Predicted target values
        prefix : str, default=""
            Prefix for metric names
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of computed metrics
        """
        prefix = f"{prefix}_" if prefix else ""
        
        metrics = {
            f"{prefix}r2": r2_score(y_true, y_pred),
            f"{prefix}rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            f"{prefix}mse": mean_squared_error(y_true, y_pred),
            f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        }
        
        # Add MAPE if no zero values in y_true
        if not np.any(y_true == 0):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics[f"{prefix}mape"] = mape
        
        return metrics
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
        --------
        Dict[str, Any]
            Training summary with metrics and metadata
        """
        if not self.is_fitted_:
            raise ValueError(f"Model {self.model_name} must be fitted to get training summary")
        
        summary = {
            'model_name': self.model_name,
            'model_type': type(self.estimator).__name__,
            'training_time': self.training_time_,
            'training_samples': self.training_samples_,
            'n_features': self.n_features_,
            'hyperparameters': self.hyperparameters,
            'requires_scaling': self.requires_scaling,
        }
        
        # Add metrics
        summary.update(self.train_metrics_)
        summary.update(self.test_metrics_)
        
        # Add overfitting metric if both train and test available
        if 'train_r2' in self.train_metrics_ and 'test_r2' in self.test_metrics_:
            summary['overfitting'] = self.train_metrics_['train_r2'] - self.test_metrics_['test_r2']
        
        # Add cross-validation results if available
        if self.cv_scores_:
            summary.update(self.cv_scores_)
        
        return summary
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        filepath : Union[str, Path]
            Path where to save the model
        """
        if not self.is_fitted_:
            raise ValueError(f"Model {self.model_name} must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'estimator': self.estimator,
            'model_name': self.model_name,
            'requires_scaling': self.requires_scaling,
            'hyperparameters': self.hyperparameters,
            'feature_names': self.feature_names_,
            'training_summary': self.get_training_summary()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model {self.model_name} saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'BaseModel':
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        filepath : Union[str, Path]
            Path to the saved model
            
        Returns:
        --------
        BaseModel
            Loaded model instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Create new instance
        model = cls(
            model_name=model_data['model_name'],
            estimator=model_data['estimator'],
            requires_scaling=model_data['requires_scaling'],
            hyperparameters=model_data['hyperparameters']
        )
        
        # Restore metadata
        model.feature_names_ = model_data['feature_names']
        model.is_fitted_ = True
        
        # Restore training summary data
        training_summary = model_data['training_summary']
        model.training_time_ = training_summary.get('training_time')
        model.training_samples_ = training_summary.get('training_samples')
        model.n_features_ = training_summary.get('n_features')
        
        logger.info(f"Model {model.model_name} loaded from {filepath}")
        
        return model
    
    def __str__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted_ else "not fitted"
        return f"{self.model_name} ({type(self.estimator).__name__}) - {status}"
    
    def __repr__(self) -> str:
        """Detailed representation of the model."""
        return (f"BaseModel(model_name='{self.model_name}', "
                f"estimator={type(self.estimator).__name__}, "
                f"requires_scaling={self.requires_scaling}, "
                f"is_fitted={self.is_fitted_})")