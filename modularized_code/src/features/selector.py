"""
Feature Selection Module
========================

This module provides various feature selection strategies for vibration prediction:
- Top-K feature selection based on importance scores
- Cumulative importance threshold selection  
- Statistical feature selection (F-statistic, mutual information)
- Custom feature selection with manual overrides
- Feature set comparison and validation

Based on the analysis from notebook section 6-feature_selection.ipynb
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

from config.settings import get_settings
from utils.helpers import Timer
from .importance import FeatureImportanceAnalyzer

logger = logging.getLogger(__name__)
settings = get_settings()


class FeatureSelector:
    """
    Comprehensive feature selection using multiple strategies.
    
    This class implements various feature selection methods and provides
    tools for comparing their effectiveness on vibration prediction tasks.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the feature selector.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducibility
        """
        self.random_state = random_state
        self.feature_sets_ = {}
        self.evaluation_results_ = {}
        self.X_train_ = None
        self.X_test_ = None
        self.y_train_ = None
        self.y_test_ = None
        
    def fit(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
            y_train: pd.Series, y_test: pd.Series) -> 'FeatureSelector':
        """
        Fit the feature selector with train/test data.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature matrix
        X_test : pd.DataFrame
            Test feature matrix
        y_train : pd.Series
            Training target variable
        y_test : pd.Series
            Test target variable
            
        Returns:
        --------
        FeatureSelector
            Self for method chaining
        """
        logger.info("Fitting feature selector with train/test data")
        
        self.X_train_ = X_train.copy()
        self.X_test_ = X_test.copy()
        self.y_train_ = y_train.copy()
        self.y_test_ = y_test.copy()
        
        logger.info(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Total features available: {X_train.shape[1]}")
        
        return self
    
    def select_top_k_features(self, k: int = 20, method: str = 'random_forest') -> List[str]:
        """
        Select top K features based on importance scores.
        
        Parameters:
        -----------
        k : int, default=20
            Number of features to select
        method : str, default='random_forest'
            Method to use for importance ('random_forest', 'f_statistic', 'mutual_info')
            
        Returns:
        --------
        List[str]
            List of selected feature names
        """
        logger.info(f"Selecting top {k} features using {method} method")
        
        if method == 'random_forest':
            return self._select_rf_top_k(k)
        elif method == 'f_statistic':
            return self._select_statistical_top_k(k, score_func=f_regression)
        elif method == 'mutual_info':
            return self._select_statistical_top_k(k, score_func=mutual_info_regression)
        else:
            raise ValueError(f"Unknown method '{method}'. Available: 'random_forest', 'f_statistic', 'mutual_info'")
    
    def _select_rf_top_k(self, k: int) -> List[str]:
        """Select top K features using Random Forest importance."""
        rf_model = RandomForestRegressor(
            n_estimators=25,
            max_depth=4,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf_model.fit(self.X_train_, self.y_train_)
        
        # Get feature importance and sort
        feature_importance = list(zip(self.X_train_.columns, rf_model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        selected_features = [f[0] for f in feature_importance[:k]]
        
        # Store in feature sets
        self.feature_sets_[f'top_{k}_rf'] = {
            'features': selected_features,
            'method': 'random_forest',
            'k': k,
            'importance_scores': dict(feature_importance[:k])
        }
        
        return selected_features
    
    def _select_statistical_top_k(self, k: int, score_func) -> List[str]:
        """Select top K features using statistical tests."""
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(self.X_train_, self.y_train_)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.X_train_.columns[i] for i in selected_indices]
        
        # Store in feature sets
        method_name = 'f_statistic' if score_func == f_regression else 'mutual_info'
        self.feature_sets_[f'top_{k}_{method_name}'] = {
            'features': selected_features,
            'method': method_name,
            'k': k,
            'scores': dict(zip(selected_features, selector.scores_[selected_indices]))
        }
        
        return selected_features
    
    def select_by_cumulative_importance(self, threshold: float = 0.8, 
                                      method: str = 'random_forest') -> List[str]:
        """
        Select features that contribute to X% of cumulative importance.
        
        Parameters:
        -----------
        threshold : float, default=0.8
            Cumulative importance threshold (0.8 = 80%)
        method : str, default='random_forest'
            Method to compute importance
            
        Returns:
        --------
        List[str]
            List of selected feature names
        """
        logger.info(f"Selecting features for {threshold*100}% cumulative importance using {method}")
        
        if method == 'random_forest':
            rf_model = RandomForestRegressor(
                n_estimators=25,
                max_depth=4,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            rf_model.fit(self.X_train_, self.y_train_)
            
            # Get sorted feature importance
            feature_importance = list(zip(self.X_train_.columns, rf_model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Select features up to threshold
            cumulative_importance = 0
            selected_features = []
            
            for feature, importance in feature_importance:
                selected_features.append(feature)
                cumulative_importance += importance
                if cumulative_importance >= threshold:
                    break
            
            # Store in feature sets
            set_name = f'cumulative_{int(threshold*100)}_{method}'
            self.feature_sets_[set_name] = {
                'features': selected_features,
                'method': method,
                'threshold': threshold,
                'actual_cumulative': cumulative_importance,
                'feature_count': len(selected_features)
            }
            
            return selected_features
        
        else:
            raise ValueError(f"Cumulative selection not implemented for method '{method}'")
    
    def create_feature_sets(self) -> Dict[str, List[str]]:
        """
        Create multiple feature sets using different selection strategies.
        
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary of feature set name to feature list
        """
        logger.info("Creating multiple feature sets using different strategies")
        
        feature_sets = {}
        
        with Timer("Creating feature sets"):
            # Strategy 1: Top 20 features by Random Forest
            feature_sets['Top 20 RF'] = self.select_top_k_features(k=20, method='random_forest')
            
            # Strategy 2: Top 20 features by F-statistic
            feature_sets['Top 20 Statistical'] = self.select_top_k_features(k=20, method='f_statistic')
            
            # Strategy 3: Features for 80% cumulative importance
            feature_sets['Cumulative 80%'] = self.select_by_cumulative_importance(threshold=0.8)
            
            # Strategy 4: Features for 90% cumulative importance
            feature_sets['Cumulative 90%'] = self.select_by_cumulative_importance(threshold=0.9)
            
            # Strategy 5: All features (baseline)
            feature_sets['All Features'] = list(self.X_train_.columns)
        
        # Store feature sets
        for name, features in feature_sets.items():
            if name not in self.feature_sets_:
                self.feature_sets_[name] = {
                    'features': features,
                    'method': 'combined_strategy',
                    'feature_count': len(features)
                }
        
        logger.info(f"Created {len(feature_sets)} feature sets:")
        for name, features in feature_sets.items():
            logger.info(f"  • {name}: {len(features)} features")
        
        return feature_sets
    
    def compare_feature_sets(self, feature_sets: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        Compare performance of different feature sets using Random Forest.
        
        Parameters:
        -----------
        feature_sets : Optional[Dict[str, List[str]]], default=None
            Feature sets to compare. If None, uses all stored feature sets.
            
        Returns:
        --------
        pd.DataFrame
            Comparison results with performance metrics
        """
        if feature_sets is None:
            if not self.feature_sets_:
                feature_sets = self.create_feature_sets()
            else:
                feature_sets = {name: data['features'] for name, data in self.feature_sets_.items()}
        
        logger.info(f"Comparing {len(feature_sets)} feature sets")
        
        results = []
        
        for set_name, features in feature_sets.items():
            logger.info(f"Evaluating feature set: {set_name} ({len(features)} features)")
            
            # Select features for train/test
            available_features = [f for f in features if f in self.X_train_.columns]
            X_train_subset = self.X_train_[available_features]
            X_test_subset = self.X_test_[available_features]
            
            # Train simple Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            rf_model.fit(X_train_subset, self.y_train_)
            
            # Make predictions
            train_pred = rf_model.predict(X_train_subset)
            test_pred = rf_model.predict(X_test_subset)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train_, train_pred)
            test_r2 = r2_score(self.y_test_, test_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train_, train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test_, test_pred))
            overfitting = train_r2 - test_r2
            
            results.append({
                'Feature Set': set_name,
                'Features': len(available_features),
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Overfitting': overfitting
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Test R²', ascending=False)
        
        # Store evaluation results
        self.evaluation_results_['comparison'] = results_df
        
        logger.info("Feature set comparison complete")
        logger.info(f"Best performing set: {results_df.iloc[0]['Feature Set']} "
                   f"(R² = {results_df.iloc[0]['Test R²']:.4f})")
        
        return results_df
    
    def get_feature_overlap_analysis(self, feature_sets: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Analyze overlap between different feature selection methods.
        
        Parameters:
        -----------
        feature_sets : Optional[Dict[str, List[str]]], default=None
            Feature sets to analyze. If None, uses stored feature sets.
            
        Returns:
        --------
        Dict[str, Any]
            Overlap analysis results
        """
        if feature_sets is None:
            feature_sets = {name: data['features'] for name, data in self.feature_sets_.items() 
                          if 'Top' in name}  # Focus on top-K methods
        
        if len(feature_sets) < 2:
            logger.warning("Need at least 2 feature sets for overlap analysis")
            return {}
        
        logger.info(f"Analyzing feature overlap between {len(feature_sets)} feature sets")
        
        # Convert to sets for easier comparison
        feature_sets_as_sets = {name: set(features) for name, features in feature_sets.items()}
        
        overlap_results = {}
        
        # Pairwise overlap analysis
        set_names = list(feature_sets_as_sets.keys())
        for i, set1_name in enumerate(set_names):
            for set2_name in set_names[i+1:]:
                set1 = feature_sets_as_sets[set1_name]
                set2 = feature_sets_as_sets[set2_name]
                
                overlap = set1.intersection(set2)
                union = set1.union(set2)
                
                overlap_results[f"{set1_name} ∩ {set2_name}"] = {
                    'overlap_count': len(overlap),
                    'overlap_features': sorted(list(overlap)),
                    'jaccard_similarity': len(overlap) / len(union) if union else 0,
                    'overlap_percentage': len(overlap) / min(len(set1), len(set2)) * 100
                }
        
        # Find core features (appearing in multiple sets)
        all_features = {}
        for set_name, features in feature_sets_as_sets.items():
            for feature in features:
                all_features[feature] = all_features.get(feature, 0) + 1
        
        # Core features appearing in most sets
        threshold = max(1, len(feature_sets) // 2)  # At least half the sets
        core_features = [feature for feature, count in all_features.items() if count >= threshold]
        
        overlap_results['core_features'] = {
            'features': sorted(core_features),
            'count': len(core_features),
            'threshold': threshold
        }
        
        return overlap_results
    
    def recommend_feature_set(self, criteria: str = 'balanced') -> Tuple[str, List[str]]:
        """
        Recommend the best feature set based on specified criteria.
        
        Parameters:
        -----------
        criteria : str, default='balanced'
            Selection criteria ('best_performance', 'balanced', 'minimal')
            
        Returns:
        --------
        Tuple[str, List[str]]
            Tuple of (recommended_set_name, feature_list)
        """
        if 'comparison' not in self.evaluation_results_:
            logger.info("Running feature set comparison for recommendation")
            self.compare_feature_sets()
        
        results_df = self.evaluation_results_['comparison']
        
        if criteria == 'best_performance':
            # Best R² score regardless of complexity
            best_idx = results_df['Test R²'].idxmax()
            recommended = results_df.iloc[best_idx]
            
        elif criteria == 'balanced':
            # Good performance with reasonable feature count (< 50 features, R² > 0.85)
            candidates = results_df[
                (results_df['Test R²'] > 0.85) & 
                (results_df['Features'] < 50) &
                (results_df['Overfitting'] < 0.15)
            ]
            
            if not candidates.empty:
                # Among candidates, pick the one with fewest features
                best_idx = candidates['Features'].idxmin()
                recommended = candidates.loc[best_idx]
            else:
                # Fall back to best performance
                best_idx = results_df['Test R²'].idxmax()
                recommended = results_df.iloc[best_idx]
                
        elif criteria == 'minimal':
            # Minimal features with reasonable performance (R² > 0.80)
            candidates = results_df[results_df['Test R²'] > 0.80]
            
            if not candidates.empty:
                best_idx = candidates['Features'].idxmin()
                recommended = candidates.loc[best_idx]
            else:
                # Fall back to best performance
                best_idx = results_df['Test R²'].idxmax()
                recommended = results_df.iloc[best_idx]
        
        else:
            raise ValueError(f"Unknown criteria '{criteria}'. Available: 'best_performance', 'balanced', 'minimal'")
        
        set_name = recommended['Feature Set']
        features = self.feature_sets_[set_name]['features']
        
        logger.info(f"Recommended feature set ({criteria}): {set_name}")
        logger.info(f"  • Features: {len(features)}")
        logger.info(f"  • Test R²: {recommended['Test R²']:.4f}")
        logger.info(f"  • Test RMSE: {recommended['Test RMSE']:.3f}")
        
        return set_name, features
    
    def validate_feature_selection(self, selected_features: List[str], 
                                 target_column: str = 'vibration') -> Dict[str, Any]:
        """
        Validate a feature selection to check for data leakage and other issues.
        
        Parameters:
        -----------
        selected_features : List[str]
            List of selected features to validate
        target_column : str, default='vibration'
            Target column name to check for leakage
            
        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        logger.info(f"Validating feature selection with {len(selected_features)} features")
        
        validation_results = {
            'total_features': len(selected_features),
            'data_leakage_check': {},
            'feature_availability': {},
            'feature_types': {}
        }
        
        # Check for data leakage (vibration-related features)
        vibration_features = [f for f in selected_features 
                            if target_column.lower() in f.lower() or 'vib' in f.lower()]
        
        validation_results['data_leakage_check'] = {
            'potential_leakage_features': vibration_features,
            'leakage_count': len(vibration_features),
            'is_clean': len(vibration_features) == 0
        }
        
        # Check feature availability in training data
        available_features = [f for f in selected_features if f in self.X_train_.columns]
        missing_features = [f for f in selected_features if f not in self.X_train_.columns]
        
        validation_results['feature_availability'] = {
            'available_features': available_features,
            'missing_features': missing_features,
            'availability_rate': len(available_features) / len(selected_features) * 100
        }
        
        # Analyze feature types
        feature_type_counts = {}
        for feature in available_features:
            if '_rolling_mean_' in feature:
                feature_type_counts['Rolling Mean'] = feature_type_counts.get('Rolling Mean', 0) + 1
            elif '_rolling_std_' in feature:
                feature_type_counts['Rolling Std'] = feature_type_counts.get('Rolling Std', 0) + 1
            elif feature in ['hour', 'day_of_week', 'month']:
                feature_type_counts['Temporal'] = feature_type_counts.get('Temporal', 0) + 1
            else:
                feature_type_counts['Original'] = feature_type_counts.get('Original', 0) + 1
        
        validation_results['feature_types'] = feature_type_counts
        
        # Log validation results
        if validation_results['data_leakage_check']['is_clean']:
            logger.info("✅ No data leakage detected")
        else:
            logger.warning(f"⚠️ Potential data leakage: {vibration_features}")
        
        logger.info(f"Feature availability: {validation_results['feature_availability']['availability_rate']:.1f}%")
        logger.info(f"Feature types: {feature_type_counts}")
        
        return validation_results


def select_features_for_modeling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    strategy: str = 'balanced',
    target_column: str = 'vibration',
    random_state: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Convenience function to select features for modeling.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix
    X_test : pd.DataFrame
        Test feature matrix
    y_train : pd.Series
        Training target variable
    y_test : pd.Series
        Test target variable
    strategy : str, default='balanced'
        Selection strategy ('best_performance', 'balanced', 'minimal')
    target_column : str, default='vibration'
        Target column name
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    Tuple[List[str], pd.DataFrame]
        Selected features and comparison results
    """
    # Initialize selector
    selector = FeatureSelector(random_state=random_state)
    selector.fit(X_train, X_test, y_train, y_test)
    
    # Create and compare feature sets
    feature_sets = selector.create_feature_sets()
    comparison_results = selector.compare_feature_sets(feature_sets)
    
    # Get recommendation
    recommended_set, selected_features = selector.recommend_feature_set(criteria=strategy)
    
    # Validate selection
    validation = selector.validate_feature_selection(selected_features, target_column)
    
    logger.info(f"Feature selection complete:")
    logger.info(f"  • Strategy: {strategy}")
    logger.info(f"  • Recommended set: {recommended_set}")
    logger.info(f"  • Selected features: {len(selected_features)}")
    logger.info(f"  • Data leakage clean: {validation['data_leakage_check']['is_clean']}")
    
    return selected_features, comparison_results