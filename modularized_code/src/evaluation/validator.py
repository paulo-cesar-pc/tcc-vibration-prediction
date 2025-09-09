"""
Model Validation
================

Comprehensive model validation for robustness, reliability, and
deployment readiness assessment.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
import warnings

from config.settings import get_settings
from .metrics import ModelMetrics

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelValidator:
    """
    Comprehensive model validation for deployment readiness.
    
    This class performs various validation tests to assess model
    robustness, reliability, and suitability for production deployment.
    """
    
    def __init__(self, validation_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize model validator.
        
        Parameters:
        -----------
        validation_thresholds : Optional[Dict[str, float]], default=None
            Custom validation thresholds for different metrics
        """
        self.validation_thresholds = validation_thresholds or {
            'min_r2_score': 0.80,           # Minimum acceptable R² score
            'max_rmse_threshold': 0.005,     # Maximum acceptable RMSE
            'max_overfitting': 0.15,         # Maximum acceptable overfitting (R² diff)
            'max_residual_skew': 2.0,        # Maximum residual skewness
            'min_accuracy_within_001': 80.0,  # Minimum accuracy within ±0.001
            'max_prediction_drift': 0.10     # Maximum acceptable prediction drift
        }
        
        self.validation_results = {}
        logger.debug("ModelValidator initialized")
    
    def validate_basic_performance(self,
                                  y_true: Union[pd.Series, np.ndarray],
                                  y_pred: np.ndarray,
                                  dataset_name: str = "test") -> Dict[str, Any]:
        """
        Validate basic model performance metrics.
        
        Parameters:
        -----------
        y_true : Union[pd.Series, np.ndarray]
            True target values
        y_pred : np.ndarray
            Predicted values
        dataset_name : str, default="test"
            Name of the dataset being validated
            
        Returns:
        --------
        Dict[str, Any]
            Basic performance validation results
        """
        logger.info(f"Validating basic performance on {dataset_name} set")
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Perform validation checks
        validation_results = {
            'dataset': dataset_name,
            'r2_score': r2,
            'rmse': rmse,
            'checks': {}
        }
        
        # R² score check
        r2_pass = r2 >= self.validation_thresholds['min_r2_score']
        validation_results['checks']['r2_acceptable'] = {
            'passed': r2_pass,
            'actual': r2,
            'threshold': self.validation_thresholds['min_r2_score'],
            'message': f"R² score {'meets' if r2_pass else 'below'} minimum requirement"
        }
        
        # RMSE check
        rmse_pass = rmse <= self.validation_thresholds['max_rmse_threshold']
        validation_results['checks']['rmse_acceptable'] = {
            'passed': rmse_pass,
            'actual': rmse,
            'threshold': self.validation_thresholds['max_rmse_threshold'],
            'message': f"RMSE {'within' if rmse_pass else 'exceeds'} acceptable limit"
        }
        
        # Overall performance
        validation_results['overall_performance_acceptable'] = r2_pass and rmse_pass
        
        return validation_results
    
    def validate_overfitting(self,
                            y_train: Union[pd.Series, np.ndarray],
                            train_pred: np.ndarray,
                            y_test: Union[pd.Series, np.ndarray],
                            test_pred: np.ndarray) -> Dict[str, Any]:
        """
        Validate model for overfitting issues.
        
        Parameters:
        -----------
        y_train : Union[pd.Series, np.ndarray]
            True training values
        train_pred : np.ndarray
            Training predictions
        y_test : Union[pd.Series, np.ndarray]
            True test values
        test_pred : np.ndarray
            Test predictions
            
        Returns:
        --------
        Dict[str, Any]
            Overfitting validation results
        """
        logger.info("Validating model for overfitting")
        
        # Calculate R² scores
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        overfitting = train_r2 - test_r2
        
        # Calculate RMSE values
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        rmse_ratio = test_rmse / train_rmse if train_rmse > 0 else float('inf')
        
        validation_results = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'overfitting_r2': overfitting,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'rmse_ratio': rmse_ratio,
            'checks': {}
        }
        
        # Overfitting check
        overfitting_pass = overfitting <= self.validation_thresholds['max_overfitting']
        validation_results['checks']['overfitting_acceptable'] = {
            'passed': overfitting_pass,
            'actual': overfitting,
            'threshold': self.validation_thresholds['max_overfitting'],
            'message': f"Overfitting {'within' if overfitting_pass else 'exceeds'} acceptable limits"
        }
        
        # RMSE degradation check
        rmse_degradation_pass = rmse_ratio <= 1.5  # Test RMSE should not be more than 50% higher
        validation_results['checks']['rmse_degradation_acceptable'] = {
            'passed': rmse_degradation_pass,
            'actual': rmse_ratio,
            'threshold': 1.5,
            'message': f"RMSE degradation {'acceptable' if rmse_degradation_pass else 'too high'}"
        }
        
        validation_results['overall_overfitting_acceptable'] = overfitting_pass and rmse_degradation_pass
        
        return validation_results
    
    def validate_residual_distribution(self,
                                     y_true: Union[pd.Series, np.ndarray],
                                     y_pred: np.ndarray,
                                     dataset_name: str = "test") -> Dict[str, Any]:
        """
        Validate residual distribution properties.
        
        Parameters:
        -----------
        y_true : Union[pd.Series, np.ndarray]
            True target values
        y_pred : np.ndarray
            Predicted values
        dataset_name : str, default="test"
            Name of the dataset
            
        Returns:
        --------
        Dict[str, Any]
            Residual validation results
        """
        logger.info(f"Validating residual distribution for {dataset_name} set")
        
        residuals = np.array(y_true) - np.array(y_pred)
        
        validation_results = {
            'dataset': dataset_name,
            'residual_stats': {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'skewness': float(stats.skew(residuals)),
                'kurtosis': float(stats.kurtosis(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals))
            },
            'checks': {}
        }
        
        # Test for normality (Shapiro-Wilk test for small samples, Kolmogorov-Smirnov for large)
        if len(residuals) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                validation_results['normality_test'] = {
                    'test': 'shapiro-wilk',
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            except:
                logger.warning("Could not perform Shapiro-Wilk test")
        else:
            try:
                ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
                validation_results['normality_test'] = {
                    'test': 'kolmogorov-smirnov',
                    'statistic': ks_stat,
                    'p_value': ks_p,
                    'is_normal': ks_p > 0.05
                }
            except:
                logger.warning("Could not perform Kolmogorov-Smirnov test")
        
        # Check if residuals are approximately centered around zero
        mean_close_to_zero = abs(validation_results['residual_stats']['mean']) < 0.001
        validation_results['checks']['residuals_centered'] = {
            'passed': mean_close_to_zero,
            'actual': validation_results['residual_stats']['mean'],
            'threshold': 0.001,
            'message': f"Residuals {'are' if mean_close_to_zero else 'are not'} centered around zero"
        }
        
        # Check skewness
        skew_acceptable = abs(validation_results['residual_stats']['skewness']) <= self.validation_thresholds['max_residual_skew']
        validation_results['checks']['skewness_acceptable'] = {
            'passed': skew_acceptable,
            'actual': validation_results['residual_stats']['skewness'],
            'threshold': self.validation_thresholds['max_residual_skew'],
            'message': f"Residual skewness {'acceptable' if skew_acceptable else 'too high'}"
        }
        
        validation_results['overall_residuals_acceptable'] = mean_close_to_zero and skew_acceptable
        
        return validation_results
    
    def validate_prediction_accuracy(self,
                                   y_true: Union[pd.Series, np.ndarray],
                                   y_pred: np.ndarray,
                                   thresholds: List[float] = [0.001, 0.002, 0.005],
                                   dataset_name: str = "test") -> Dict[str, Any]:
        """
        Validate prediction accuracy within specified thresholds.
        
        Parameters:
        -----------
        y_true : Union[pd.Series, np.ndarray]
            True target values
        y_pred : np.ndarray
            Predicted values
        thresholds : List[float], default=[0.001, 0.002, 0.005]
            Accuracy thresholds to evaluate
        dataset_name : str, default="test"
            Name of the dataset
            
        Returns:
        --------
        Dict[str, Any]
            Accuracy validation results
        """
        logger.info(f"Validating prediction accuracy for {dataset_name} set")
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        validation_results = {
            'dataset': dataset_name,
            'accuracy_thresholds': {},
            'checks': {}
        }
        
        for threshold in thresholds:
            accuracy = np.mean(np.abs(y_true - y_pred) <= threshold) * 100
            validation_results['accuracy_thresholds'][threshold] = accuracy
            
            # Check if accuracy meets requirement for the smallest threshold
            if threshold == min(thresholds):
                accuracy_pass = accuracy >= self.validation_thresholds['min_accuracy_within_001']
                validation_results['checks']['accuracy_acceptable'] = {
                    'passed': accuracy_pass,
                    'actual': accuracy,
                    'threshold': self.validation_thresholds['min_accuracy_within_001'],
                    'tolerance': threshold,
                    'message': f"Accuracy within ±{threshold} {'meets' if accuracy_pass else 'below'} requirements"
                }
        
        return validation_results
    
    def validate_temporal_stability(self,
                                   y_true: Union[pd.Series, np.ndarray],
                                   y_pred: np.ndarray,
                                   time_index: Optional[pd.DatetimeIndex] = None,
                                   window_size: int = 100) -> Dict[str, Any]:
        """
        Validate temporal stability of predictions.
        
        Parameters:
        -----------
        y_true : Union[pd.Series, np.ndarray]
            True target values
        y_pred : np.ndarray
            Predicted values
        time_index : Optional[pd.DatetimeIndex], default=None
            Time index for temporal analysis
        window_size : int, default=100
            Window size for rolling analysis
            
        Returns:
        --------
        Dict[str, Any]
            Temporal stability validation results
        """
        logger.info("Validating temporal stability of predictions")
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        validation_results = {
            'window_size': window_size,
            'checks': {}
        }
        
        if len(y_true) < window_size * 2:
            logger.warning(f"Insufficient data for temporal stability analysis (need at least {window_size * 2} points)")
            validation_results['insufficient_data'] = True
            return validation_results
        
        # Calculate rolling R² scores
        rolling_r2 = []
        for i in range(window_size, len(y_true) - window_size + 1):
            window_true = y_true[i-window_size:i+window_size]
            window_pred = y_pred[i-window_size:i+window_size]
            r2 = r2_score(window_true, window_pred)
            rolling_r2.append(r2)
        
        rolling_r2 = np.array(rolling_r2)
        
        # Calculate stability metrics
        r2_mean = np.mean(rolling_r2)
        r2_std = np.std(rolling_r2)
        r2_range = np.max(rolling_r2) - np.min(rolling_r2)
        
        validation_results['temporal_stats'] = {
            'rolling_r2_mean': float(r2_mean),
            'rolling_r2_std': float(r2_std),
            'rolling_r2_range': float(r2_range),
            'rolling_r2_min': float(np.min(rolling_r2)),
            'rolling_r2_max': float(np.max(rolling_r2))
        }
        
        # Check for drift (high standard deviation indicates instability)
        stability_pass = r2_std <= self.validation_thresholds['max_prediction_drift']
        validation_results['checks']['temporal_stability'] = {
            'passed': stability_pass,
            'actual': r2_std,
            'threshold': self.validation_thresholds['max_prediction_drift'],
            'message': f"Temporal stability {'acceptable' if stability_pass else 'concerning'}"
        }
        
        validation_results['overall_temporal_stability_acceptable'] = stability_pass
        
        return validation_results
    
    def validate_deployment_readiness(self,
                                    y_train: Union[pd.Series, np.ndarray],
                                    train_pred: np.ndarray,
                                    y_test: Union[pd.Series, np.ndarray],
                                    test_pred: np.ndarray,
                                    feature_count: int,
                                    model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive deployment readiness assessment.
        
        Parameters:
        -----------
        y_train : Union[pd.Series, np.ndarray]
            True training values
        train_pred : np.ndarray
            Training predictions
        y_test : Union[pd.Series, np.ndarray]
            True test values
        test_pred : np.ndarray
            Test predictions
        feature_count : int
            Number of features used
        model_name : str, default="model"
            Name of the model being validated
            
        Returns:
        --------
        Dict[str, Any]
            Complete deployment readiness assessment
        """
        logger.info(f"Performing deployment readiness assessment for {model_name}")
        
        deployment_results = {
            'model_name': model_name,
            'feature_count': feature_count,
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'validations': {},
            'overall_assessment': {}
        }
        
        # Run all validation checks
        deployment_results['validations']['basic_performance'] = self.validate_basic_performance(y_test, test_pred)
        deployment_results['validations']['overfitting'] = self.validate_overfitting(y_train, train_pred, y_test, test_pred)
        deployment_results['validations']['residual_distribution'] = self.validate_residual_distribution(y_test, test_pred)
        deployment_results['validations']['prediction_accuracy'] = self.validate_prediction_accuracy(y_test, test_pred)
        
        # Temporal stability (if enough data)
        if len(y_test) >= 200:
            deployment_results['validations']['temporal_stability'] = self.validate_temporal_stability(y_test, test_pred)
        
        # Compile overall assessment
        passed_checks = 0
        total_checks = 0
        
        critical_failures = []
        warnings = []
        
        for validation_name, validation_results in deployment_results['validations'].items():
            if 'checks' in validation_results:
                for check_name, check_results in validation_results['checks'].items():
                    total_checks += 1
                    if check_results['passed']:
                        passed_checks += 1
                    else:
                        if 'r2_acceptable' in check_name or 'overfitting' in check_name:
                            critical_failures.append(f"{validation_name}.{check_name}")
                        else:
                            warnings.append(f"{validation_name}.{check_name}")
        
        # Determine deployment readiness
        pass_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        if len(critical_failures) == 0 and pass_rate >= 80:
            readiness_status = "READY"
            readiness_message = "Model passes all critical checks and is ready for deployment"
        elif len(critical_failures) == 0 and pass_rate >= 60:
            readiness_status = "CONDITIONAL"
            readiness_message = "Model ready for deployment with monitoring"
        else:
            readiness_status = "NOT_READY"
            readiness_message = "Model requires improvement before deployment"
        
        deployment_results['overall_assessment'] = {
            'readiness_status': readiness_status,
            'readiness_message': readiness_message,
            'pass_rate': round(pass_rate, 1),
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'critical_failures': critical_failures,
            'warnings': warnings
        }
        
        # Store results
        self.validation_results[model_name] = deployment_results
        
        logger.info(f"Deployment assessment complete for {model_name}: {readiness_status}")
        
        return deployment_results
    
    def get_validation_summary(self, model_name: str) -> Dict[str, Any]:
        """
        Get validation summary for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the validated model
            
        Returns:
        --------
        Dict[str, Any]
            Validation summary
        """
        if model_name not in self.validation_results:
            raise ValueError(f"No validation results found for model: {model_name}")
        
        return self.validation_results[model_name]['overall_assessment']


def validate_model_performance(y_train: Union[pd.Series, np.ndarray],
                              train_pred: np.ndarray,
                              y_test: Union[pd.Series, np.ndarray],
                              test_pred: np.ndarray,
                              model_name: str = "model",
                              feature_count: Optional[int] = None,
                              validation_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Convenience function for complete model validation.
    
    Parameters:
    -----------
    y_train : Union[pd.Series, np.ndarray]
        True training values
    train_pred : np.ndarray
        Training predictions
    y_test : Union[pd.Series, np.ndarray]
        True test values
    test_pred : np.ndarray
        Test predictions
    model_name : str, default="model"
        Name of the model
    feature_count : Optional[int], default=None
        Number of features used
    validation_thresholds : Optional[Dict[str, float]], default=None
        Custom validation thresholds
        
    Returns:
    --------
    Dict[str, Any]
        Complete validation results
    """
    validator = ModelValidator(validation_thresholds)
    
    feature_count = feature_count or 0
    
    return validator.validate_deployment_readiness(
        y_train, train_pred, y_test, test_pred, 
        feature_count, model_name
    )