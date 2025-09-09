"""
Data Preprocessing Module
========================

Advanced preprocessing operations for time series industrial data.
Handles resampling, scaling, and data preparation for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.settings import DATA_CONFIG
from utils.helpers import Timer, log_dataframe_info


class DataPreprocessor:
    """
    Preprocess time series data for machine learning.
    
    Features:
    - Time series resampling with multiple aggregation strategies
    - Train/test splits with temporal awareness
    - Data scaling and normalization
    - Feature and target separation
    """
    
    def __init__(self,
                 target_frequency: str = None,
                 test_size: float = None,
                 random_state: int = None):
        """
        Initialize DataPreprocessor with configuration.
        
        Parameters:
        -----------
        target_frequency : str, optional
            Target frequency for resampling (e.g., '5T' for 5 minutes)
        test_size : float, optional
            Proportion of data for testing
        random_state : int, optional
            Random state for reproducible splits
        """
        self.target_frequency = target_frequency or DATA_CONFIG["target_frequency"]
        self.test_size = test_size or DATA_CONFIG["test_size"]
        self.random_state = random_state or DATA_CONFIG["random_state"]
        self.aggregation_method = DATA_CONFIG["aggregation_method"]
        
        # Store preprocessing info
        self.preprocessing_info = {}
        
    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temporal patterns in the data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with datetime index
            
        Returns:
        --------
        Dict[str, Any]
            Temporal analysis results
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for temporal analysis")
        
        # Calculate frequency information
        time_deltas = df.index.to_series().diff().dropna()
        most_common_delta = time_deltas.mode()[0] if len(time_deltas.mode()) > 0 else None
        
        analysis = {
            'time_range': (df.index.min(), df.index.max()),
            'duration_days': (df.index.max() - df.index.min()).days,
            'total_points': len(df),
            'most_common_interval': most_common_delta,
            'unique_intervals': time_deltas.nunique(),
            'missing_timestamps': self._detect_missing_timestamps(df),
            'temporal_statistics': {
                'mean_interval': time_deltas.mean(),
                'std_interval': time_deltas.std(),
                'min_interval': time_deltas.min(),
                'max_interval': time_deltas.max()
            }
        }
        
        return analysis
    
    def _detect_missing_timestamps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect missing timestamps in time series."""
        if len(df) < 2:
            return {'missing_count': 0, 'gaps': []}
        
        # Create expected time range
        start_time = df.index.min()
        end_time = df.index.max()
        
        # Infer frequency from most common interval
        time_deltas = df.index.to_series().diff().dropna()
        if len(time_deltas) == 0:
            return {'missing_count': 0, 'gaps': []}
        
        most_common_delta = time_deltas.mode()[0]
        
        # Generate expected timestamps
        expected_timestamps = pd.date_range(
            start=start_time, 
            end=end_time, 
            freq=most_common_delta
        )
        
        # Find missing timestamps
        actual_timestamps = set(df.index)
        expected_timestamps = set(expected_timestamps)
        missing_timestamps = expected_timestamps - actual_timestamps
        
        # Identify gaps (consecutive missing periods)
        gaps = []
        if missing_timestamps:
            missing_sorted = sorted(missing_timestamps)
            gap_start = missing_sorted[0]
            gap_end = missing_sorted[0]
            
            for i in range(1, len(missing_sorted)):
                if missing_sorted[i] - missing_sorted[i-1] == most_common_delta:
                    gap_end = missing_sorted[i]
                else:
                    gaps.append((gap_start, gap_end))
                    gap_start = missing_sorted[i]
                    gap_end = missing_sorted[i]
            gaps.append((gap_start, gap_end))
        
        return {
            'missing_count': len(missing_timestamps),
            'gaps': gaps,
            'gap_count': len(gaps),
            'longest_gap': max(gaps, key=lambda x: x[1] - x[0]) if gaps else None
        }
    
    def resample_data(self, 
                     df: pd.DataFrame, 
                     target_col: str,
                     frequency: str = None,
                     aggregation: Union[str, Dict] = None) -> pd.DataFrame:
        """
        Resample time series data to target frequency.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with datetime index
        target_col : str
            Name of target column
        frequency : str, optional
            Target frequency for resampling
        aggregation : Union[str, Dict], optional
            Aggregation method(s)
            
        Returns:
        --------
        pd.DataFrame
            Resampled DataFrame
        """
        if frequency is None:
            frequency = self.target_frequency
            
        if aggregation is None:
            aggregation = self.aggregation_method
        
        print(f"📊 Resampling data to {frequency} intervals...")
        
        # Analyze original frequency
        temporal_analysis = self.analyze_temporal_patterns(df)
        original_interval = temporal_analysis['most_common_interval']
        
        print(f"  • Original interval: {original_interval}")
        print(f"  • Target interval: {frequency}")
        print(f"  • Original shape: {df.shape}")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        # Define aggregation strategy
        if isinstance(aggregation, str):
            # Use same aggregation for all numeric columns
            agg_funcs = {col: aggregation for col in numeric_cols}
            
            # Special handling for target column and different variable types
            for col in numeric_cols:
                if col == target_col:
                    agg_funcs[col] = 'mean'  # Always use mean for target
                elif 'TEMPERATURE' in col.upper() or 'PRESSURE' in col.upper():
                    agg_funcs[col] = 'mean'
                elif 'FLOW' in col.upper() or 'CURRENT' in col.upper():
                    agg_funcs[col] = 'mean'
                elif 'POWER' in col.upper():
                    agg_funcs[col] = 'mean'
                else:
                    agg_funcs[col] = aggregation
            
            # Handle categorical columns
            for col in categorical_cols:
                agg_funcs[col] = 'first'  # Take first value in each period
                
        elif isinstance(aggregation, dict):
            agg_funcs = aggregation.copy()
        else:
            raise ValueError("aggregation must be str or dict")
        
        # Perform resampling
        df_resampled = df.resample(frequency).agg(agg_funcs)
        
        # Remove rows with all NaN values
        df_resampled = df_resampled.dropna(how='all')
        
        # Forward fill then backward fill remaining NaN values
        df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate resampling statistics
        data_reduction = len(df) - len(df_resampled)
        reduction_percent = (data_reduction / len(df)) * 100
        
        print(f"  • Resampled shape: {df_resampled.shape}")
        print(f"  • Data reduction: {data_reduction:,} rows ({reduction_percent:.1f}%)")
        print(f"  • Final interval: {frequency}")
        
        # Store preprocessing info
        self.preprocessing_info['resampling'] = {
            'original_shape': df.shape,
            'resampled_shape': df_resampled.shape,
            'frequency': frequency,
            'aggregation_method': aggregation,
            'data_reduction': data_reduction,
            'reduction_percentage': reduction_percent
        }
        
        return df_resampled
    
    def create_time_aware_split(self, 
                              df: pd.DataFrame, 
                              target_col: str,
                              test_size: float = None,
                              validation_size: float = None) -> Tuple:
        """
        Create time-aware train/validation/test splits.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        target_col : str
            Name of target column
        test_size : float, optional
            Size of test set
        validation_size : float, optional
            Size of validation set
            
        Returns:
        --------
        Tuple
            (X_train, X_val, X_test, y_train, y_val, y_test) if validation_size provided
            (X_train, X_test, y_train, y_test) otherwise
        """
        if test_size is None:
            test_size = self.test_size
            
        validation_size = validation_size if validation_size is not None else DATA_CONFIG.get("validation_size", 0.0)
        
        print(f"📊 Creating time-aware data splits:")
        print(f"  • Test size: {test_size:.1%}")
        if validation_size > 0:
            print(f"  • Validation size: {validation_size:.1%}")
            print(f"  • Training size: {1 - test_size - validation_size:.1%}")
        else:
            print(f"  • Training size: {1 - test_size:.1%}")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Calculate split indices (temporal order preserved)
        total_samples = len(df)
        
        if validation_size > 0:
            # Three-way split: train / validation / test
            test_start_idx = int(total_samples * (1 - test_size))
            val_start_idx = int(total_samples * (1 - test_size - validation_size))
            
            # Split data
            X_train = X.iloc[:val_start_idx]
            X_val = X.iloc[val_start_idx:test_start_idx]
            X_test = X.iloc[test_start_idx:]
            
            y_train = y.iloc[:val_start_idx]
            y_val = y.iloc[val_start_idx:test_start_idx]
            y_test = y.iloc[test_start_idx:]
            
            # Log split information
            print(f"  • Training: {len(X_train):,} samples ({len(X_train)/total_samples:.1%})")
            print(f"  • Validation: {len(X_val):,} samples ({len(X_val)/total_samples:.1%})")
            print(f"  • Test: {len(X_test):,} samples ({len(X_test)/total_samples:.1%})")
            print(f"  • Features: {X_train.shape[1]}")
            
            # Store split info
            self.preprocessing_info['data_split'] = {
                'type': 'time_aware_three_way',
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'feature_count': X_train.shape[1],
                'time_ranges': {
                    'train': (X_train.index.min(), X_train.index.max()),
                    'val': (X_val.index.min(), X_val.index.max()),
                    'test': (X_test.index.min(), X_test.index.max())
                }
            }
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        else:
            # Two-way split: train / test
            split_idx = int(total_samples * (1 - test_size))
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            # Log split information
            print(f"  • Training: {len(X_train):,} samples ({len(X_train)/total_samples:.1%})")
            print(f"  • Test: {len(X_test):,} samples ({len(X_test)/total_samples:.1%})")
            print(f"  • Features: {X_train.shape[1]}")
            
            # Store split info
            self.preprocessing_info['data_split'] = {
                'type': 'time_aware_two_way',
                'train_size': len(X_train),
                'test_size': len(X_test),
                'feature_count': X_train.shape[1],
                'time_ranges': {
                    'train': (X_train.index.min(), X_train.index.max()),
                    'test': (X_test.index.min(), X_test.index.max())
                }
            }
            
            return X_train, X_test, y_train, y_test
    
    def scale_features(self, 
                      X_train: pd.DataFrame, 
                      X_test: pd.DataFrame,
                      X_val: Optional[pd.DataFrame] = None,
                      method: str = 'standard') -> Tuple:
        """
        Scale features using specified method.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        X_val : Optional[pd.DataFrame]
            Validation features
        method : str
            Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
        --------
        Tuple
            (X_train_scaled, X_test_scaled, [X_val_scaled,] scaler)
        """
        print(f"🎯 Scaling features using {method} scaling...")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit scaler on training data
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        
        # Transform test data
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )
        
        print(f"  • Fitted scaler on {len(X_train_scaled)} training samples")
        print(f"  • Scaled {len(X_test_scaled)} test samples")
        
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val),
                index=X_val.index,
                columns=X_val.columns
            )
            print(f"  • Scaled {len(X_val_scaled)} validation samples")
            
            return X_train_scaled, X_val_scaled, X_test_scaled, scaler
        
        return X_train_scaled, X_test_scaled, scaler
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing information."""
        return self.preprocessing_info.copy()
    
    def preprocess_pipeline(self, 
                          df: pd.DataFrame, 
                          target_col: str,
                          resample: bool = True,
                          scale_features: bool = False,
                          validation_split: bool = False) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        target_col : str
            Name of target column
        resample : bool
            Whether to resample data
        scale_features : bool
            Whether to scale features
        validation_split : bool
            Whether to create validation set
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing processed data and metadata
        """
        with Timer("Data preprocessing", verbose=True):
            results = {'target_column': target_col}
            
            # Step 1: Resample if requested
            if resample:
                df_processed = self.resample_data(df, target_col)
            else:
                df_processed = df.copy()
                print("⚠️ Resampling skipped")
            
            # Step 2: Create train/test splits
            validation_size = DATA_CONFIG.get("validation_size", 0.15) if validation_split else 0.0
            
            if validation_split:
                X_train, X_val, X_test, y_train, y_val, y_test = self.create_time_aware_split(
                    df_processed, target_col, validation_size=validation_size
                )
                results.update({
                    'X_val': X_val,
                    'y_val': y_val,
                })
            else:
                X_train, X_test, y_train, y_test = self.create_time_aware_split(
                    df_processed, target_col, validation_size=0.0
                )
                X_val = None
            
            # Step 3: Scale features if requested
            if scale_features:
                if X_val is not None:
                    X_train_scaled, X_val_scaled, X_test_scaled, scaler = self.scale_features(
                        X_train, X_test, X_val
                    )
                    results.update({
                        'X_train': X_train_scaled,
                        'X_val': X_val_scaled,
                        'X_test': X_test_scaled,
                        'scaler': scaler
                    })
                else:
                    X_train_scaled, X_test_scaled, scaler = self.scale_features(
                        X_train, X_test
                    )
                    results.update({
                        'X_train': X_train_scaled,
                        'X_test': X_test_scaled,
                        'scaler': scaler
                    })
            else:
                results.update({
                    'X_train': X_train,
                    'X_test': X_test,
                })
                print("⚠️ Feature scaling skipped")
            
            # Add targets
            results.update({
                'y_train': y_train,
                'y_test': y_test,
                'preprocessing_info': self.get_preprocessing_info()
            })
            
            # Log final results
            print(f"\n✅ Preprocessing pipeline completed!")
            print(f"  • Final training shape: {results['X_train'].shape}")
            print(f"  • Final test shape: {results['X_test'].shape}")
            if 'X_val' in results:
                print(f"  • Final validation shape: {results['X_val'].shape}")
            print(f"  • Target variable: {target_col}")
            
            return results


# Convenience functions for direct usage
def resample_aggregate(df: pd.DataFrame, 
                      target_col: str, 
                      frequency: str = '5T',
                      **kwargs) -> pd.DataFrame:
    """
    Convenience function to resample data directly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str
        Name of target column
    frequency : str
        Target frequency
    **kwargs
        Additional parameters for DataPreprocessor
        
    Returns:
    --------
    pd.DataFrame
        Resampled DataFrame
    """
    preprocessor = DataPreprocessor(target_frequency=frequency, **kwargs)
    return preprocessor.resample_data(df, target_col)


def prepare_model_data(df: pd.DataFrame, 
                      target_col: str, 
                      test_size: float = 0.2,
                      **kwargs) -> Tuple:
    """
    Convenience function to prepare data for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str
        Name of target column
    test_size : float
        Size of test set
    **kwargs
        Additional parameters for DataPreprocessor
        
    Returns:
    --------
    Tuple
        (X_train, X_test, y_train, y_test)
    """
    preprocessor = DataPreprocessor(test_size=test_size, **kwargs)
    return preprocessor.create_time_aware_split(df, target_col, validation_size=0.0)