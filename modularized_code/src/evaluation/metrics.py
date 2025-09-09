"""
Model Evaluation Metrics
========================

Comprehensive metrics calculation for regression models used in
vibration prediction. Includes standard regression metrics and
domain-specific evaluations.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, 
    explained_variance_score, max_error
)
from scipy import stats
import warnings

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelMetrics:
    """
    Comprehensive metrics calculator for regression models.
    
    This class provides a unified interface for computing various
    evaluation metrics for vibration prediction models.
    """
    
    def __init__(self, decimal_places: int = 6):
        """
        Initialize metrics calculator.
        
        Parameters:
        -----------
        decimal_places : int, default=6
            Number of decimal places for metric precision
        """
        self.decimal_places = decimal_places
        self.computed_metrics = {}
        
    def calculate_basic_metrics(self, 
                               y_true: Union[pd.Series, np.ndarray],
                               y_pred: np.ndarray,
                               prefix: str = "") -> Dict[str, float]:
        """
        Calculate basic regression metrics.
        
        Parameters:
        -----------
        y_true : Union[pd.Series, np.ndarray]
            True target values
        y_pred : np.ndarray
            Predicted target values
        prefix : str, default=""
            Prefix for metric names (e.g., 'train_', 'test_')
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of basic metrics
        """
        prefix = f"{prefix}_" if prefix else ""
        
        # Convert to numpy arrays for consistency
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {
            f"{prefix}r2": round(r2_score(y_true, y_pred), self.decimal_places),
            f"{prefix}rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), self.decimal_places),
            f"{prefix}mse": round(mean_squared_error(y_true, y_pred), self.decimal_places),
            f"{prefix}mae": round(mean_absolute_error(y_true, y_pred), self.decimal_places),
            f"{prefix}explained_variance": round(explained_variance_score(y_true, y_pred), self.decimal_places),
            f"{prefix}max_error": round(max_error(y_true, y_pred), self.decimal_places)
        }
        
        return metrics
    
    def calculate_percentage_metrics(self,
                                   y_true: Union[pd.Series, np.ndarray],
                                   y_pred: np.ndarray,
                                   prefix: str = "") -> Dict[str, float]:
        """
        Calculate percentage-based metrics.
        
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
            Dictionary of percentage metrics
        """
        prefix = f"{prefix}_" if prefix else ""
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {}
        
        # MAPE (Mean Absolute Percentage Error) - only if no zeros in y_true
        if not np.any(y_true == 0):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics[f"{prefix}mape"] = round(mape, self.decimal_places)
        
        # Symmetric MAPE (handles zeros better)
        smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
        metrics[f"{prefix}smape"] = round(smape, self.decimal_places)
        
        # Percentage error relative to mean
        mean_absolute_percentage_error = (np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true)) * 100
        metrics[f"{prefix}rmse_percent"] = round(mean_absolute_percentage_error, self.decimal_places)
        
        return metrics
    
    def calculate_residual_metrics(self,
                                  y_true: Union[pd.Series, np.ndarray],
                                  y_pred: np.ndarray,
                                  prefix: str = "") -> Dict[str, float]:
        """
        Calculate residual-based metrics.
        
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
            Dictionary of residual metrics
        """
        prefix = f"{prefix}_" if prefix else ""
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        residuals = y_true - y_pred
        
        metrics = {
            f"{prefix}residual_mean": round(np.mean(residuals), self.decimal_places),
            f"{prefix}residual_std": round(np.std(residuals), self.decimal_places),
            f"{prefix}residual_min": round(np.min(residuals), self.decimal_places),
            f"{prefix}residual_max": round(np.max(residuals), self.decimal_places),
            f"{prefix}residual_q25": round(np.percentile(residuals, 25), self.decimal_places),
            f"{prefix}residual_q75": round(np.percentile(residuals, 75), self.decimal_places),
            f"{prefix}residual_iqr": round(np.percentile(residuals, 75) - np.percentile(residuals, 25), self.decimal_places)
        }
        
        # Add skewness and kurtosis if scipy is available
        try:
            metrics[f"{prefix}residual_skewness"] = round(stats.skew(residuals), self.decimal_places)
            metrics[f"{prefix}residual_kurtosis"] = round(stats.kurtosis(residuals), self.decimal_places)
        except:
            logger.warning("Could not calculate skewness and kurtosis")
        
        return metrics
    
    def calculate_accuracy_within_thresholds(self,
                                           y_true: Union[pd.Series, np.ndarray],
                                           y_pred: np.ndarray,
                                           thresholds: List[float] = [0.001, 0.002, 0.005, 0.01],
                                           prefix: str = "") -> Dict[str, float]:
        """
        Calculate prediction accuracy within specified thresholds.
        
        Parameters:
        -----------
        y_true : Union[pd.Series, np.ndarray]
            True target values
        y_pred : np.ndarray
            Predicted target values
        thresholds : List[float], default=[0.001, 0.002, 0.005, 0.01]
            Accuracy thresholds to evaluate
        prefix : str, default=""
            Prefix for metric names
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of accuracy metrics
        """
        prefix = f"{prefix}_" if prefix else ""
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {}
        
        for threshold in thresholds:
            accuracy = np.mean(np.abs(y_true - y_pred) <= threshold) * 100
            threshold_str = str(threshold).replace('.', 'p')  # 0.001 -> 0p001
            metrics[f"{prefix}accuracy_within_{threshold_str}"] = round(accuracy, 2)
        
        return metrics
    
    def calculate_data_statistics(self,
                                 y_true: Union[pd.Series, np.ndarray],
                                 prefix: str = "") -> Dict[str, float]:
        """
        Calculate statistics of the target data.
        
        Parameters:
        -----------
        y_true : Union[pd.Series, np.ndarray]
            True target values
        prefix : str, default=""
            Prefix for metric names
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of data statistics
        """
        prefix = f"{prefix}_" if prefix else ""
        
        y_true = np.array(y_true)
        
        metrics = {
            f"{prefix}target_mean": round(np.mean(y_true), self.decimal_places),
            f"{prefix}target_std": round(np.std(y_true), self.decimal_places),
            f"{prefix}target_min": round(np.min(y_true), self.decimal_places),
            f"{prefix}target_max": round(np.max(y_true), self.decimal_places),
            f"{prefix}target_median": round(np.median(y_true), self.decimal_places),
            f"{prefix}target_range": round(np.max(y_true) - np.min(y_true), self.decimal_places),
            f"{prefix}sample_count": len(y_true)
        }
        
        return metrics
    
    def calculate_comprehensive_metrics(self,
                                      y_true: Union[pd.Series, np.ndarray],
                                      y_pred: np.ndarray,
                                      prefix: str = "",
                                      include_thresholds: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Parameters:
        -----------
        y_true : Union[pd.Series, np.ndarray]
            True target values
        y_pred : np.ndarray
            Predicted target values
        prefix : str, default=""
            Prefix for metric names
        include_thresholds : Optional[List[float]], default=None
            Accuracy thresholds to include
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of all computed metrics
        """
        logger.debug(f"Computing comprehensive metrics with prefix '{prefix}'")
        
        all_metrics = {}
        
        # Basic regression metrics
        all_metrics.update(self.calculate_basic_metrics(y_true, y_pred, prefix))
        
        # Percentage metrics
        all_metrics.update(self.calculate_percentage_metrics(y_true, y_pred, prefix))
        
        # Residual metrics
        all_metrics.update(self.calculate_residual_metrics(y_true, y_pred, prefix))
        
        # Data statistics
        all_metrics.update(self.calculate_data_statistics(y_true, prefix))
        
        # Accuracy within thresholds
        if include_thresholds:
            all_metrics.update(self.calculate_accuracy_within_thresholds(
                y_true, y_pred, include_thresholds, prefix
            ))
        
        # Store computed metrics
        self.computed_metrics[prefix] = all_metrics
        
        logger.debug(f"Computed {len(all_metrics)} metrics with prefix '{prefix}'")
        
        return all_metrics
    
    def compare_models(self,
                      model_metrics: Dict[str, Dict[str, float]],
                      primary_metric: str = 'test_r2',
                      ascending: bool = False) -> pd.DataFrame:
        """
        Compare multiple models based on their metrics.
        
        Parameters:
        -----------
        model_metrics : Dict[str, Dict[str, float]]
            Dictionary mapping model names to their metrics
        primary_metric : str, default='test_r2'
            Primary metric to sort by
        ascending : bool, default=False
            Sort order
            
        Returns:
        --------
        pd.DataFrame
            Comparison DataFrame sorted by primary metric
        """
        comparison_df = pd.DataFrame.from_dict(model_metrics, orient='index')
        
        # Sort by primary metric
        if primary_metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(primary_metric, ascending=ascending)
        
        # Add ranking
        comparison_df['rank'] = range(1, len(comparison_df) + 1)
        
        return comparison_df
    
    def calculate_overfitting_metrics(self,
                                    train_metrics: Dict[str, float],
                                    test_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate overfitting-related metrics.
        
        Parameters:
        -----------
        train_metrics : Dict[str, float]
            Training metrics
        test_metrics : Dict[str, float]
            Test metrics
            
        Returns:
        --------
        Dict[str, float]
            Overfitting metrics
        """
        overfitting_metrics = {}
        
        # R² difference
        if 'train_r2' in train_metrics and 'test_r2' in test_metrics:
            overfitting_metrics['overfitting_r2'] = round(
                train_metrics['train_r2'] - test_metrics['test_r2'], self.decimal_places
            )
        
        # RMSE ratio
        if 'train_rmse' in train_metrics and 'test_rmse' in test_metrics:
            overfitting_metrics['rmse_ratio'] = round(
                test_metrics['test_rmse'] / train_metrics['train_rmse'], self.decimal_places
            )
        
        # Performance degradation percentage
        if 'train_r2' in train_metrics and 'test_r2' in test_metrics:
            if train_metrics['train_r2'] > 0:
                degradation_pct = (1 - test_metrics['test_r2'] / train_metrics['train_r2']) * 100
                overfitting_metrics['performance_degradation_pct'] = round(degradation_pct, 2)
        
        return overfitting_metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all computed metrics.
        
        Returns:
        --------
        Dict[str, Any]
            Summary of computed metrics
        """
        summary = {
            'total_metric_sets': len(self.computed_metrics),
            'metric_prefixes': list(self.computed_metrics.keys()),
            'total_metrics': sum(len(metrics) for metrics in self.computed_metrics.values())
        }
        
        return summary


def calculate_regression_metrics(y_true: Union[pd.Series, np.ndarray],
                                y_pred: np.ndarray,
                                include_percentages: bool = True,
                                include_residuals: bool = True,
                                include_thresholds: Optional[List[float]] = None,
                                decimal_places: int = 6) -> Dict[str, float]:
    """
    Convenience function to calculate comprehensive regression metrics.
    
    Parameters:
    -----------
    y_true : Union[pd.Series, np.ndarray]
        True target values
    y_pred : np.ndarray
        Predicted target values
    include_percentages : bool, default=True
        Whether to include percentage-based metrics
    include_residuals : bool, default=True
        Whether to include residual analysis metrics
    include_thresholds : Optional[List[float]], default=None
        Accuracy thresholds to evaluate
    decimal_places : int, default=6
        Number of decimal places for precision
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of computed metrics
    """
    calculator = ModelMetrics(decimal_places=decimal_places)
    
    # Basic metrics
    metrics = calculator.calculate_basic_metrics(y_true, y_pred)
    
    # Additional metrics based on parameters
    if include_percentages:
        metrics.update(calculator.calculate_percentage_metrics(y_true, y_pred))
    
    if include_residuals:
        metrics.update(calculator.calculate_residual_metrics(y_true, y_pred))
    
    if include_thresholds:
        metrics.update(calculator.calculate_accuracy_within_thresholds(y_true, y_pred, include_thresholds))
    
    # Data statistics
    metrics.update(calculator.calculate_data_statistics(y_true))
    
    return metrics


def evaluate_model_performance(y_train: Union[pd.Series, np.ndarray],
                              train_pred: np.ndarray,
                              y_test: Union[pd.Series, np.ndarray],
                              test_pred: np.ndarray,
                              accuracy_thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Complete model performance evaluation for both training and test sets.
    
    Parameters:
    -----------
    y_train : Union[pd.Series, np.ndarray]
        True training target values
    train_pred : np.ndarray
        Training predictions
    y_test : Union[pd.Series, np.ndarray]
        True test target values
    test_pred : np.ndarray
        Test predictions
    accuracy_thresholds : Optional[List[float]], default=None
        Accuracy thresholds to evaluate
        
    Returns:
    --------
    Dict[str, Any]
        Complete evaluation results
    """
    calculator = ModelMetrics()
    
    # Calculate metrics for both sets
    train_metrics = calculator.calculate_comprehensive_metrics(
        y_train, train_pred, "train", accuracy_thresholds
    )
    
    test_metrics = calculator.calculate_comprehensive_metrics(
        y_test, test_pred, "test", accuracy_thresholds
    )
    
    # Calculate overfitting metrics
    overfitting_metrics = calculator.calculate_overfitting_metrics(train_metrics, test_metrics)
    
    # Compile results
    evaluation_results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'overfitting_metrics': overfitting_metrics,
        'summary': calculator.get_metrics_summary()
    }
    
    return evaluation_results