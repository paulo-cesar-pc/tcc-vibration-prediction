"""
Feature Engineering Module
==========================

Advanced feature engineering for industrial vibration prediction.
Creates rolling statistics, temporal features, and domain-specific transformations
while preventing data leakage.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from math import ceil

from config.settings import FEATURE_CONFIG
from utils.helpers import Timer, log_dataframe_info


class FeatureEngineer:
    """
    Create engineered features for industrial vibration prediction.
    
    Features:
    - Rolling statistical features (mean, std, min, max)
    - Temporal features (hour, day, month, etc.)
    - Rate of change and velocity features
    - Cross-variable interactions
    - Domain-specific transformations
    """
    
    def __init__(self,
                 rolling_windows: Optional[List[int]] = None,
                 rolling_statistics: Optional[List[str]] = None,
                 key_patterns: Optional[List[str]] = None,
                 temporal_features: Optional[List[str]] = None):
        """
        Initialize FeatureEngineer with configuration.
        
        Parameters:
        -----------
        rolling_windows : Optional[List[int]]
            Rolling window sizes for feature engineering
        rolling_statistics : Optional[List[str]]
            Statistics to calculate ('mean', 'std', 'min', 'max')
        key_patterns : Optional[List[str]]
            Variable name patterns to focus on
        temporal_features : Optional[List[str]]
            Temporal features to create
        """
        self.rolling_windows = rolling_windows or FEATURE_CONFIG["rolling_windows"]
        self.rolling_statistics = rolling_statistics or FEATURE_CONFIG["rolling_statistics"]
        self.key_patterns = key_patterns or FEATURE_CONFIG["key_patterns"]
        self.temporal_features = temporal_features or FEATURE_CONFIG["temporal_features"]
        self.max_features_per_pattern = FEATURE_CONFIG["max_features_per_pattern"]
        
        # Feature engineering statistics
        self.engineering_stats = {}
        
    def identify_key_variables(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """
        Identify key process variables for feature engineering.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        target_col : str
            Target column to exclude
            
        Returns:
        --------
        List[str]
            List of key variable names
        """
        # Exclude ALL vibration columns to prevent data leakage
        feature_cols = [col for col in df.columns if 'VIBRATION' not in col.upper()]
        
        key_vars = []
        pattern_counts = {}
        
        # Find variables matching key patterns
        for pattern in self.key_patterns:
            pattern_cols = [col for col in feature_cols if pattern in col.upper()]
            
            # Limit number of features per pattern to avoid explosion
            selected_cols = pattern_cols[:self.max_features_per_pattern]
            key_vars.extend(selected_cols)
            pattern_counts[pattern] = len(selected_cols)
        
        # Remove duplicates while preserving order
        key_vars = list(dict.fromkeys(key_vars))
        
        print(f"🔍 Identified key variables for feature engineering:")
        for pattern, count in pattern_counts.items():
            print(f"  • {pattern}: {count} variables")
        print(f"  • Total key variables: {len(key_vars)}")
        
        return key_vars
    
    def create_rolling_features(self, df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        variables : List[str]
            Variables to create rolling features for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with rolling features added
        """
        df_rolling = df.copy()
        feature_count = 0
        
        print(f"🔄 Creating rolling features:")
        print(f"  • Variables: {len(variables)}")
        print(f"  • Windows: {self.rolling_windows}")
        print(f"  • Statistics: {self.rolling_statistics}")
        
        for var in variables:
            if var not in df_rolling.columns:
                continue
                
            for window in self.rolling_windows:
                for stat in self.rolling_statistics:
                    feature_name = f"{var}_rolling_{stat}_{window}"
                    
                    try:
                        if stat == 'mean':
                            df_rolling[feature_name] = df_rolling[var].rolling(window).mean()
                        elif stat == 'std':
                            df_rolling[feature_name] = df_rolling[var].rolling(window).std()
                        elif stat == 'min':
                            df_rolling[feature_name] = df_rolling[var].rolling(window).min()
                        elif stat == 'max':
                            df_rolling[feature_name] = df_rolling[var].rolling(window).max()
                        elif stat == 'median':
                            df_rolling[feature_name] = df_rolling[var].rolling(window).median()
                        elif stat == 'var':
                            df_rolling[feature_name] = df_rolling[var].rolling(window).var()
                        else:
                            print(f"    Warning: Unknown statistic '{stat}' for {var}")
                            continue
                            
                        feature_count += 1
                        
                    except Exception as e:
                        print(f"    Error creating {feature_name}: {e}")
        
        print(f"  • Created {feature_count} rolling features")
        return df_rolling
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from datetime index.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with datetime index
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with temporal features added
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            print("  ⚠️ Warning: DataFrame does not have DatetimeIndex, skipping temporal features")
            return df
        
        df_temporal = df.copy()
        feature_count = 0
        
        print(f"📅 Creating temporal features:")
        
        # Standard temporal features
        if 'hour' in self.temporal_features:
            df_temporal['hour'] = df_temporal.index.hour
            feature_count += 1
            
        if 'day_of_week' in self.temporal_features:
            df_temporal['day_of_week'] = df_temporal.index.dayofweek
            feature_count += 1
            
        if 'month' in self.temporal_features:
            df_temporal['month'] = df_temporal.index.month
            feature_count += 1
            
        if 'day_of_year' in self.temporal_features:
            df_temporal['day_of_year'] = df_temporal.index.dayofyear
            feature_count += 1
            
        if 'week_of_year' in self.temporal_features:
            df_temporal['week_of_year'] = df_temporal.index.isocalendar().week
            feature_count += 1
            
        if 'quarter' in self.temporal_features:
            df_temporal['quarter'] = df_temporal.index.quarter
            feature_count += 1
        
        # Cyclical encoding for certain features (optional enhancement)
        cyclical_features = ['hour', 'day_of_week', 'month']
        cyclical_periods = {'hour': 24, 'day_of_week': 7, 'month': 12}
        
        for feature in cyclical_features:
            if feature in self.temporal_features and feature in df_temporal.columns:
                period = cyclical_periods[feature]
                
                # Create sin and cos components
                df_temporal[f'{feature}_sin'] = np.sin(2 * np.pi * df_temporal[feature] / period)
                df_temporal[f'{feature}_cos'] = np.cos(2 * np.pi * df_temporal[feature] / period)
                feature_count += 2
        
        print(f"  • Created {feature_count} temporal features")
        return df_temporal
    
    def create_lag_features(self, 
                           df: pd.DataFrame, 
                           variables: List[str], 
                           lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Create lag features for specified variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        variables : List[str]
            Variables to create lag features for
        lags : List[int]
            Lag periods to create
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with lag features added
        """
        df_lagged = df.copy()
        feature_count = 0
        
        print(f"⏰ Creating lag features:")
        print(f"  • Variables: {len(variables)}")
        print(f"  • Lags: {lags}")
        
        for var in variables:
            if var not in df_lagged.columns:
                continue
                
            for lag in lags:
                feature_name = f"{var}_lag_{lag}"
                df_lagged[feature_name] = df_lagged[var].shift(lag)
                feature_count += 1
        
        print(f"  • Created {feature_count} lag features")
        return df_lagged
    
    def create_rate_of_change_features(self, 
                                     df: pd.DataFrame, 
                                     variables: List[str],
                                     periods: List[int] = [1, 3, 6]) -> pd.DataFrame:
        """
        Create rate of change features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        variables : List[str]
            Variables to calculate rate of change for
        periods : List[int]
            Periods for rate of change calculation
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with rate of change features added
        """
        df_roc = df.copy()
        feature_count = 0
        
        print(f"📈 Creating rate of change features:")
        print(f"  • Variables: {len(variables)}")
        print(f"  • Periods: {periods}")
        
        for var in variables:
            if var not in df_roc.columns:
                continue
                
            for period in periods:
                try:
                    # Percentage change
                    feature_name = f"{var}_pct_change_{period}"
                    df_roc[feature_name] = df_roc[var].pct_change(periods=period)
                    
                    # Absolute difference
                    feature_name = f"{var}_diff_{period}"
                    df_roc[feature_name] = df_roc[var].diff(periods=period)
                    
                    feature_count += 2
                    
                except Exception as e:
                    print(f"    Error creating rate of change for {var}: {e}")
        
        print(f"  • Created {feature_count} rate of change features")
        return df_roc
    
    def create_interaction_features(self, 
                                  df: pd.DataFrame, 
                                  variables: List[str],
                                  max_interactions: int = 10) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        variables : List[str]
            Variables to create interactions between
        max_interactions : int
            Maximum number of interactions to create
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with interaction features added
        """
        df_interactions = df.copy()
        feature_count = 0
        
        print(f"🔗 Creating interaction features:")
        print(f"  • Variables: {len(variables)}")
        print(f"  • Max interactions: {max_interactions}")
        
        # Create interactions between first few variables to avoid explosion
        selected_vars = variables[:min(5, len(variables))]
        
        interaction_count = 0
        for i, var1 in enumerate(selected_vars):
            for j, var2 in enumerate(selected_vars):
                if i >= j or interaction_count >= max_interactions:
                    continue
                
                if var1 not in df_interactions.columns or var2 not in df_interactions.columns:
                    continue
                
                try:
                    # Multiplicative interaction
                    feature_name = f"{var1}_{var2}_interaction"
                    df_interactions[feature_name] = df_interactions[var1] * df_interactions[var2]
                    
                    # Ratio feature (if denominator is not zero)
                    feature_name = f"{var1}_{var2}_ratio"
                    df_interactions[feature_name] = df_interactions[var1] / (df_interactions[var2] + 1e-8)
                    
                    feature_count += 2
                    interaction_count += 1
                    
                except Exception as e:
                    print(f"    Error creating interaction {var1}*{var2}: {e}")
        
        print(f"  • Created {feature_count} interaction features")
        return df_interactions
    
    def clean_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean engineered features by removing infinite values and handling missing data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with engineered features
            
        Returns:
        --------
        pd.DataFrame
            Cleaned DataFrame
        """
        print(f"🧹 Cleaning engineered features:")
        
        initial_shape = df.shape
        
        # Replace infinite values with NaN
        df_clean = df.replace([np.inf, -np.inf], np.nan)
        
        # Count infinite values that were replaced
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            print(f"  • Replaced {inf_count} infinite values with NaN")
        
        # Remove columns with excessive missing data
        missing_ratios = df_clean.isnull().sum() / len(df_clean)
        excessive_missing = missing_ratios > FEATURE_CONFIG.get("correlation_threshold", 0.7)
        cols_to_drop = missing_ratios[excessive_missing].index
        
        if len(cols_to_drop) > 0:
            print(f"  • Dropping {len(cols_to_drop)} features with >70% missing data")
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        # Fill remaining missing values
        # For rolling features, forward fill is appropriate
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Final check for remaining missing values
        final_missing = df_clean.isnull().sum().sum()
        if final_missing > 0:
            print(f"  • Filling {final_missing} remaining missing values with column medians")
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        
        print(f"  • Shape after cleaning: {initial_shape} → {df_clean.shape}")
        
        return df_clean
    
    def engineer_features(self, 
                         df: pd.DataFrame, 
                         target_col: str,
                         create_rolling: bool = True,
                         create_temporal: bool = True,
                         create_lags: bool = False,
                         create_roc: bool = False,
                         create_interactions: bool = False) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        target_col : str
            Target column name (excluded from feature engineering)
        create_rolling : bool
            Whether to create rolling features
        create_temporal : bool
            Whether to create temporal features
        create_lags : bool
            Whether to create lag features
        create_roc : bool
            Whether to create rate of change features
        create_interactions : bool
            Whether to create interaction features
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with engineered features
        """
        with Timer("Feature engineering", verbose=True):
            print(f"🔧 Starting feature engineering pipeline")
            print(f"  • Initial shape: {df.shape}")
            print(f"  • Target column: {target_col}")
            
            # Initialize stats
            self.engineering_stats = {
                'initial_shape': df.shape,
                'target_column': target_col,
                'features_created': {}
            }
            
            # CRITICAL: Identify key variables excluding ALL vibration columns
            key_vars = self.identify_key_variables(df, target_col)
            
            # Verify no data leakage
            vibration_features = [col for col in key_vars if 'VIBRATION' in col.upper()]
            if vibration_features:
                raise ValueError(f"Data leakage detected! Vibration features in key variables: {vibration_features}")
            
            print(f"\n✅ Data leakage check passed - no vibration features in predictors")
            
            df_engineered = df.copy()
            
            # Step 1: Rolling features
            if create_rolling and key_vars:
                df_engineered = self.create_rolling_features(df_engineered, key_vars)
                self.engineering_stats['features_created']['rolling'] = True
            
            # Step 2: Temporal features
            if create_temporal:
                df_engineered = self.create_temporal_features(df_engineered)
                self.engineering_stats['features_created']['temporal'] = True
            
            # Step 3: Lag features (optional)
            if create_lags and key_vars:
                df_engineered = self.create_lag_features(df_engineered, key_vars[:3])  # Limit to avoid too many features
                self.engineering_stats['features_created']['lag'] = True
            
            # Step 4: Rate of change features (optional)
            if create_roc and key_vars:
                df_engineered = self.create_rate_of_change_features(df_engineered, key_vars[:3])
                self.engineering_stats['features_created']['rate_of_change'] = True
            
            # Step 5: Interaction features (optional)
            if create_interactions and len(key_vars) >= 2:
                df_engineered = self.create_interaction_features(df_engineered, key_vars)
                self.engineering_stats['features_created']['interactions'] = True
            
            # Step 6: Clean engineered features
            df_engineered = self.clean_engineered_features(df_engineered)
            
            # Final statistics
            self.engineering_stats['final_shape'] = df_engineered.shape
            self.engineering_stats['features_added'] = df_engineered.shape[1] - df.shape[1]
            
            # Store for feature types categorization
            self._last_engineered_df = df_engineered
            self._last_target_col = target_col
            
            # Add feature types to stats
            self.engineering_stats['feature_types'] = self.get_feature_types(df_engineered, target_col)
            
            # Log final results
            self._log_engineering_summary()
            
            return df_engineered
    
    def _log_engineering_summary(self) -> None:
        """Log comprehensive feature engineering summary."""
        stats = self.engineering_stats
        
        print(f"\n✅ Feature engineering completed successfully!")
        print(f"📊 Engineering Summary:")
        print(f"  • Initial features: {stats['initial_shape'][1]}")
        print(f"  • Final features: {stats['final_shape'][1]}")
        print(f"  • Features added: {stats['features_added']}")
        print(f"  • Target column: {stats['target_column']}")
        
        # Show which types of features were created
        created = stats['features_created']
        print(f"  • Feature types created:")
        for feature_type, was_created in created.items():
            status = "✅" if was_created else "❌"
            print(f"    {status} {feature_type.replace('_', ' ').title()}")
        
        # Feature categories
        final_cols = stats['final_shape'][1]
        predictive_features = final_cols - 1  # Exclude target
        print(f"  • Predictive features available: {predictive_features}")
    
    def get_feature_types(self, df: pd.DataFrame, target_col: str) -> Dict[str, List[str]]:
        """
        Categorize features by type for analysis.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        target_col : str
            Target column name
            
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary categorizing features by type
        """
        feature_cols = [col for col in df.columns if col != target_col]
        
        categories = {
            'original': [],
            'rolling_mean': [],
            'rolling_std': [],
            'rolling_other': [],
            'temporal': [],
            'lag': [],
            'rate_of_change': [],
            'interaction': [],
            'other': []
        }
        
        for col in feature_cols:
            if '_rolling_mean_' in col:
                categories['rolling_mean'].append(col)
            elif '_rolling_std_' in col:
                categories['rolling_std'].append(col)
            elif '_rolling_' in col:
                categories['rolling_other'].append(col)
            elif col in self.temporal_features or '_sin' in col or '_cos' in col:
                categories['temporal'].append(col)
            elif '_lag_' in col:
                categories['lag'].append(col)
            elif '_pct_change_' in col or '_diff_' in col:
                categories['rate_of_change'].append(col)
            elif '_interaction' in col or '_ratio' in col:
                categories['interaction'].append(col)
            elif not any(pattern in col for pattern in ['_rolling_', '_lag_', '_pct_', '_diff_', '_interaction', '_ratio']):
                categories['original'].append(col)
            else:
                categories['other'].append(col)
        
        return categories
    
    def get_engineering_stats(self) -> Dict[str, Any]:
        """Get comprehensive feature engineering statistics."""
        stats = self.engineering_stats.copy()
        
        # Add feature_types categorization if available
        if hasattr(self, '_last_engineered_df') and hasattr(self, '_last_target_col'):
            stats['feature_types'] = self.get_feature_types(self._last_engineered_df, self._last_target_col)
        else:
            # Provide empty categorization as fallback
            stats['feature_types'] = {
                'original': [],
                'rolling_mean': [],
                'rolling_std': [],
                'rolling_other': [],
                'temporal': [],
                'lag': [],
                'rate_of_change': [],
                'interaction': [],
                'other': []
            }
        
        return stats


# Convenience function for direct usage
def engineer_features(df: pd.DataFrame, 
                     target_col: str,
                     **kwargs) -> pd.DataFrame:
    """
    Convenience function to engineer features directly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str
        Target column name
    **kwargs
        Additional parameters for FeatureEngineer
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features
    """
    engineer = FeatureEngineer(**kwargs)
    return engineer.engineer_features(df, target_col)