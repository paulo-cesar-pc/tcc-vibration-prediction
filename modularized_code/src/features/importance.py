"""
Feature Importance Analysis Module
=================================

This module provides comprehensive feature importance analysis using multiple methods:
- Random Forest feature importance
- Statistical importance (F-statistic, mutual information)  
- Correlation-based importance
- Permutation importance
- Multi-method importance combination

Based on the analysis from notebook section 5-feature_importance.ipynb
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.utils import resample

from config.settings import get_settings
from utils.helpers import Timer

logger = logging.getLogger(__name__)
settings = get_settings()


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis using multiple methods.
    
    This class implements various feature importance techniques to identify
    the most predictive features for vibration prediction models.
    """
    
    def __init__(self, random_state: int = 42, sample_size: Optional[int] = 5000):
        """
        Initialize the feature importance analyzer.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducibility
        sample_size : Optional[int], default=5000
            Maximum number of samples to use for analysis (for performance)
        """
        self.random_state = random_state
        self.sample_size = sample_size
        self.importance_results_ = {}
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, target_column: str = 'vibration') -> 'FeatureImportanceAnalyzer':
        """
        Fit the importance analyzer and compute all importance metrics.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        target_column : str, default='vibration'
            Name of target column (for exclusion checks)
            
        Returns:
        --------
        FeatureImportanceAnalyzer
            Self for method chaining
        """
        logger.info("Starting comprehensive feature importance analysis")
        
        # Store feature names
        self.feature_names_ = list(X.columns)
        
        # Sample data if needed for performance
        X_sample, y_sample = self._sample_data(X, y)
        
        with Timer("Feature importance analysis"):
            # Compute all importance methods
            self._compute_rf_importance(X_sample, y_sample)
            self._compute_statistical_importance(X_sample, y_sample)
            self._compute_correlation_importance(X_sample, y_sample)
            self._compute_permutation_importance(X_sample, y_sample)
            self._compute_combined_importance()
            self._analyze_feature_types()
            
        logger.info(f"Feature importance analysis complete for {len(self.feature_names_)} features")
        return self
    
    def _sample_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Sample data if dataset is large for performance."""
        if self.sample_size and len(X) > self.sample_size:
            logger.info(f"Sampling {self.sample_size:,} rows from {len(X):,} total rows")
            X_sample, y_sample = resample(
                X, y, 
                n_samples=self.sample_size, 
                random_state=self.random_state
            )
            return X_sample, y_sample
        return X, y
    
    def _compute_rf_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Compute Random Forest feature importance."""
        logger.info("Computing Random Forest feature importance")
        
        # Optimized RF for importance analysis
        rf_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf_model.fit(X, y)
        
        # Store results
        self.importance_results_['random_forest'] = {
            'importances': rf_model.feature_importances_,
            'feature_names': list(X.columns),
            'model_r2': rf_model.score(X, y),
            'model': rf_model
        }
        
        logger.info(f"Random Forest importance computed (R² = {rf_model.score(X, y):.3f})")
    
    def _compute_statistical_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Compute statistical feature importance using F-statistic and mutual information."""
        logger.info("Computing statistical feature importance")
        
        # F-statistic based importance
        f_scores, f_pvalues = f_regression(X, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=np.max(f_scores[~np.isinf(f_scores)]))
        
        # Mutual information importance
        mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        mi_scores = np.nan_to_num(mi_scores, nan=0.0)
        
        # Store results
        self.importance_results_['statistical'] = {
            'f_scores': f_scores,
            'f_pvalues': f_pvalues,
            'mi_scores': mi_scores,
            'feature_names': list(X.columns)
        }
        
        logger.info("Statistical importance computed (F-statistic and mutual information)")
    
    def _compute_correlation_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Compute correlation-based feature importance."""
        logger.info("Computing correlation-based importance")
        
        # Absolute correlation with target
        correlations = X.corrwith(y).abs()
        correlations = correlations.fillna(0)  # Handle NaN correlations
        
        # Store results
        self.importance_results_['correlation'] = {
            'correlations': correlations.values,
            'feature_names': list(X.columns)
        }
        
        logger.info("Correlation-based importance computed")
    
    def _compute_permutation_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Compute permutation importance using the fitted Random Forest model."""
        logger.info("Computing permutation importance")
        
        if 'random_forest' not in self.importance_results_:
            raise ValueError("Random Forest must be computed before permutation importance")
        
        rf_model = self.importance_results_['random_forest']['model']
        
        # Compute permutation importance
        perm_importance = permutation_importance(
            rf_model, X, y, 
            n_repeats=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Store results
        self.importance_results_['permutation'] = {
            'importances_mean': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std,
            'feature_names': list(X.columns)
        }
        
        logger.info("Permutation importance computed")
    
    def _compute_combined_importance(self) -> None:
        """Compute combined importance score from all methods."""
        logger.info("Computing combined importance score")
        
        # Normalize all importance scores to [0, 1]
        methods = {}
        
        if 'random_forest' in self.importance_results_:
            rf_imp = self.importance_results_['random_forest']['importances']
            methods['rf'] = rf_imp / np.max(rf_imp) if np.max(rf_imp) > 0 else rf_imp
        
        if 'statistical' in self.importance_results_:
            f_scores = self.importance_results_['statistical']['f_scores']
            mi_scores = self.importance_results_['statistical']['mi_scores']
            methods['f_score'] = f_scores / np.max(f_scores) if np.max(f_scores) > 0 else f_scores
            methods['mi_score'] = mi_scores / np.max(mi_scores) if np.max(mi_scores) > 0 else mi_scores
        
        if 'correlation' in self.importance_results_:
            corr = self.importance_results_['correlation']['correlations']
            methods['correlation'] = corr / np.max(corr) if np.max(corr) > 0 else corr
        
        if 'permutation' in self.importance_results_:
            perm_imp = self.importance_results_['permutation']['importances_mean']
            methods['permutation'] = perm_imp / np.max(perm_imp) if np.max(perm_imp) > 0 else perm_imp
        
        # Compute weighted average (equal weights for now)
        if methods:
            combined_scores = np.mean(list(methods.values()), axis=0)
            
            self.importance_results_['combined'] = {
                'importances': combined_scores,
                'feature_names': self.feature_names_,
                'method_weights': {method: 1.0 for method in methods.keys()},
                'individual_methods': methods
            }
        
        logger.info("Combined importance score computed")
    
    def _analyze_feature_types(self) -> None:
        """Analyze importance by feature type (rolling, temporal, original)."""
        logger.info("Analyzing importance by feature type")
        
        if 'combined' not in self.importance_results_:
            return
        
        # Categorize features
        feature_types = {
            'Rolling Mean': [],
            'Rolling Std': [], 
            'Rolling Min': [],
            'Rolling Max': [],
            'Temporal': [],
            'Lag': [],
            'Rate of Change': [],
            'Original': []
        }
        
        combined_imp = self.importance_results_['combined']['importances']
        
        for i, feature in enumerate(self.feature_names_):
            importance = combined_imp[i]
            
            if '_rolling_mean_' in feature:
                feature_types['Rolling Mean'].append((feature, importance))
            elif '_rolling_std_' in feature:
                feature_types['Rolling Std'].append((feature, importance))
            elif '_rolling_min_' in feature:
                feature_types['Rolling Min'].append((feature, importance))
            elif '_rolling_max_' in feature:
                feature_types['Rolling Max'].append((feature, importance))
            elif feature in ['hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos']:
                feature_types['Temporal'].append((feature, importance))
            elif '_lag_' in feature:
                feature_types['Lag'].append((feature, importance))
            elif '_rate_of_change' in feature or '_roc_' in feature:
                feature_types['Rate of Change'].append((feature, importance))
            else:
                feature_types['Original'].append((feature, importance))
        
        # Compute statistics by type
        type_stats = {}
        for feat_type, features in feature_types.items():
            if features:
                importances = [imp for _, imp in features]
                type_stats[feat_type] = {
                    'total_importance': np.sum(importances),
                    'avg_importance': np.mean(importances),
                    'count': len(features),
                    'top_feature': max(features, key=lambda x: x[1]) if features else None
                }
        
        self.importance_results_['by_type'] = type_stats
        logger.info("Feature type analysis complete")
    
    def get_top_features(self, method: str = 'combined', k: int = 20) -> List[Tuple[str, float]]:
        """
        Get top K features by importance.
        
        Parameters:
        -----------
        method : str, default='combined'
            Importance method to use ('combined', 'random_forest', 'correlation', etc.)
        k : int, default=20
            Number of top features to return
            
        Returns:
        --------
        List[Tuple[str, float]]
            List of (feature_name, importance_score) tuples
        """
        if method not in self.importance_results_:
            raise ValueError(f"Method '{method}' not available. Available: {list(self.importance_results_.keys())}")
        
        result = self.importance_results_[method]
        
        if method == 'statistical':
            # Use F-scores for statistical method
            importances = result['f_scores']
        elif method == 'permutation':
            importances = result['importances_mean']
        elif method == 'correlation':
            importances = result['correlations']
        else:
            importances = result['importances']
        
        feature_names = result['feature_names']
        
        # Create list of (feature, importance) and sort
        feature_importance_pairs = list(zip(feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance_pairs[:k]
    
    def get_cumulative_importance_features(self, threshold: float = 0.8, method: str = 'combined') -> List[str]:
        """
        Get features that contribute to a cumulative importance threshold.
        
        Parameters:
        -----------
        threshold : float, default=0.8
            Cumulative importance threshold (0.8 = 80%)
        method : str, default='combined'
            Importance method to use
            
        Returns:
        --------
        List[str]
            List of feature names
        """
        top_features = self.get_top_features(method=method, k=len(self.feature_names_))
        
        # Calculate cumulative importance
        total_importance = sum(imp for _, imp in top_features)
        cumulative = 0
        selected_features = []
        
        for feature, importance in top_features:
            selected_features.append(feature)
            cumulative += importance
            if cumulative / total_importance >= threshold:
                break
        
        return selected_features
    
    def get_importance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive importance analysis summary.
        
        Returns:
        --------
        Dict[str, Any]
            Summary dictionary with key statistics
        """
        summary = {
            'total_features': len(self.feature_names_),
            'methods_computed': list(self.importance_results_.keys()),
            'sample_size': self.sample_size
        }
        
        # Add method-specific summaries
        for method, result in self.importance_results_.items():
            if method == 'by_type':
                summary['feature_types'] = result
            elif method == 'combined':
                summary['top_10_features'] = self.get_top_features(method='combined', k=10)
                summary['features_for_80_percent'] = len(self.get_cumulative_importance_features(0.8))
                summary['features_for_90_percent'] = len(self.get_cumulative_importance_features(0.9))
            elif method == 'random_forest':
                summary['rf_model_r2'] = result.get('model_r2', 0)
        
        return summary
    
    def create_importance_dataframe(self, method: str = 'combined') -> pd.DataFrame:
        """
        Create a DataFrame with feature importance rankings.
        
        Parameters:
        -----------
        method : str, default='combined'
            Importance method to use
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: feature, importance, rank, feature_type
        """
        top_features = self.get_top_features(method=method, k=len(self.feature_names_))
        
        # Create DataFrame
        df = pd.DataFrame(top_features, columns=['feature', 'importance'])
        df['rank'] = range(1, len(df) + 1)
        
        # Add feature type
        df['feature_type'] = df['feature'].apply(self._categorize_feature)
        
        return df
    
    def _categorize_feature(self, feature: str) -> str:
        """Categorize a feature by its name."""
        if '_rolling_mean_' in feature:
            return 'Rolling Mean'
        elif '_rolling_std_' in feature:
            return 'Rolling Std'
        elif '_rolling_min_' in feature:
            return 'Rolling Min'
        elif '_rolling_max_' in feature:
            return 'Rolling Max'
        elif feature in ['hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos']:
            return 'Temporal'
        elif '_lag_' in feature:
            return 'Lag'
        elif '_rate_of_change' in feature or '_roc_' in feature:
            return 'Rate of Change'
        else:
            return 'Original'


def analyze_feature_importance(
    X: pd.DataFrame, 
    y: pd.Series,
    sample_size: Optional[int] = 5000,
    target_column: str = 'vibration',
    random_state: int = 42
) -> FeatureImportanceAnalyzer:
    """
    Convenience function to perform complete feature importance analysis.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    sample_size : Optional[int], default=5000
        Maximum samples for analysis
    target_column : str, default='vibration'
        Target column name
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    FeatureImportanceAnalyzer
        Fitted analyzer with all importance results
    """
    analyzer = FeatureImportanceAnalyzer(
        random_state=random_state,
        sample_size=sample_size
    )
    
    analyzer.fit(X, y, target_column=target_column)
    
    # Log summary
    summary = analyzer.get_importance_summary()
    logger.info(f"Feature importance analysis complete:")
    logger.info(f"  • Total features: {summary['total_features']}")
    logger.info(f"  • Methods computed: {', '.join(summary['methods_computed'])}")
    
    if 'features_for_80_percent' in summary:
        logger.info(f"  • Features for 80% importance: {summary['features_for_80_percent']}")
    
    if 'rf_model_r2' in summary:
        logger.info(f"  • Random Forest R²: {summary['rf_model_r2']:.3f}")
    
    return analyzer