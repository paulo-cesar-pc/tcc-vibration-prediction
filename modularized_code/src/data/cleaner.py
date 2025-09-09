"""
Data Cleaning Module
===================

Robust data cleaning and validation for industrial vibration data.
Handles missing values, outliers, and data quality issues specific to sensor data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from scipy import stats

from config.settings import DATA_CONFIG
from utils.helpers import validate_data_structure, Timer, log_dataframe_info


class DataCleaner:
    """
    Clean and validate industrial sensor data with domain-specific logic.
    
    Features:
    - Vibration data validation and filtering
    - Missing data handling strategies
    - Outlier detection and removal
    - Data quality reporting
    - Categorical variable processing
    """
    
    def __init__(self,
                 vibration_min: float = None,
                 vibration_max: float = None,
                 missing_threshold: float = None,
                 outlier_method: str = "percentile"):
        """
        Initialize DataCleaner with configuration.
        
        Parameters:
        -----------
        vibration_min : float, optional
            Minimum valid vibration value (mm/s)
        vibration_max : float, optional  
            Maximum valid vibration value (mm/s)
        missing_threshold : float, optional
            Threshold for dropping columns with missing data (0-1)
        outlier_method : str
            Method for outlier detection ('percentile', 'iqr', 'zscore')
        """
        self.vibration_min = vibration_min or DATA_CONFIG["vibration_min_value"]
        self.vibration_max = vibration_max or DATA_CONFIG["vibration_max_value"]
        self.missing_threshold = missing_threshold or DATA_CONFIG["missing_threshold"]
        self.outlier_method = outlier_method
        
        # Outlier detection parameters
        self.outlier_percentiles = {
            "low": DATA_CONFIG["outlier_percentile_low"],
            "high": DATA_CONFIG["outlier_percentile_high"]
        }
        
        # Cleaning statistics
        self.cleaning_stats = {}
        
    def identify_target_column(self, df: pd.DataFrame) -> str:
        """
        Identify the primary vibration target column.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        str
            Name of target vibration column
            
        Raises:
        -------
        ValueError
            If no vibration column found
        """
        vibration_cols = [col for col in df.columns if 'VIBRATION' in col.upper()]
        
        if not vibration_cols:
            raise ValueError("No vibration column found in data")
        
        # Use the first vibration column as target
        target_col = vibration_cols[0]
        
        # Log available vibration columns
        if len(vibration_cols) > 1:
            print(f"🎯 Multiple vibration columns found: {vibration_cols}")
            print(f"   Using '{target_col}' as target variable")
        else:
            print(f"🎯 Target variable identified: {target_col}")
        
        return target_col
    
    def validate_vibration_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Validate and filter vibration data within realistic ranges.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        target_col : str
            Name of target vibration column
            
        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame
        """
        initial_len = len(df)
        
        # Log target column statistics before filtering
        target_stats = df[target_col].describe()
        print(f"\n🔍 Target Column Analysis (before filtering):")
        print(f"  • Column: {target_col}")
        print(f"  • Count: {int(target_stats['count']):,} readings")
        print(f"  • Range: {target_stats['min']:.3f} to {target_stats['max']:.3f} mm/s")
        print(f"  • Mean: {target_stats['mean']:.3f} ± {target_stats['std']:.3f}")
        
        # Filter realistic vibration values
        print(f"\n🚿 Filtering vibration data:")
        print(f"  • Valid range: {self.vibration_min} to {self.vibration_max} mm/s")
        
        # Count values outside range
        too_low = (df[target_col] <= self.vibration_min).sum()
        too_high = (df[target_col] > self.vibration_max).sum()
        
        # Apply filter
        df_filtered = df[
            (df[target_col] > self.vibration_min) & 
            (df[target_col] <= self.vibration_max)
        ].copy()
        
        filtered_count = initial_len - len(df_filtered)
        
        print(f"  • Values too low (≤{self.vibration_min}): {too_low:,}")
        print(f"  • Values too high (>{self.vibration_max}): {too_high:,}")
        print(f"  • Total filtered: {filtered_count:,} ({filtered_count/initial_len*100:.1f}%)")
        print(f"  • Remaining: {len(df_filtered):,} readings")
        
        # Update cleaning stats
        self.cleaning_stats['vibration_filtering'] = {
            'initial_count': initial_len,
            'filtered_count': filtered_count,
            'final_count': len(df_filtered),
            'filter_percentage': filtered_count/initial_len*100
        }
        
        return df_filtered
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data with multiple strategies.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with missing data handled
        """
        print(f"\n🔧 Handling missing data:")
        
        initial_cols = len(df.columns)
        initial_rows = len(df)
        
        # Analyze missing data pattern
        missing_data = df.isnull().sum()
        missing_percentages = (missing_data / len(df)) * 100
        columns_with_missing = missing_percentages[missing_percentages > 0]
        
        print(f"  • Columns with missing data: {len(columns_with_missing)}/{len(df.columns)}")
        if len(columns_with_missing) > 0:
            print(f"  • Average missing percentage: {missing_percentages.mean():.1f}%")
        
        # Remove columns with excessive missing data
        cols_to_drop = missing_percentages[missing_percentages > self.missing_threshold * 100].index
        if len(cols_to_drop) > 0:
            print(f"  • Dropping {len(cols_to_drop)} columns with >{self.missing_threshold*100}% missing:")
            for col in cols_to_drop[:5]:  # Show first 5
                print(f"    - {col}: {missing_percentages[col]:.1f}% missing")
            if len(cols_to_drop) > 5:
                print(f"    - ... and {len(cols_to_drop)-5} more")
            
            df = df.drop(columns=cols_to_drop)
        
        # Fill remaining missing values using forward/backward fill
        # This is appropriate for time series sensor data
        print(f"  • Filling remaining missing values with forward/backward fill")
        
        # Count remaining missing values before filling
        remaining_missing_before = df.isnull().sum().sum()
        
        df_filled = df.fillna(method='ffill').fillna(method='bfill')
        
        # Count missing values after filling
        remaining_missing_after = df_filled.isnull().sum().sum()
        
        print(f"  • Missing values before fill: {remaining_missing_before:,}")
        print(f"  • Missing values after fill: {remaining_missing_after:,}")
        
        # If still missing values, use column mean for numeric columns
        if remaining_missing_after > 0:
            print(f"  • Filling remaining gaps with column means")
            numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
            df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())
        
        final_missing = df_filled.isnull().sum().sum()
        
        # Update cleaning stats
        self.cleaning_stats['missing_data'] = {
            'initial_columns': initial_cols,
            'dropped_columns': len(cols_to_drop),
            'final_columns': len(df_filled.columns),
            'missing_before': remaining_missing_before,
            'missing_after': final_missing
        }
        
        print(f"  • Final missing values: {final_missing}")
        print(f"  • Columns retained: {len(df_filled.columns)}/{initial_cols}")
        
        return df_filled
    
    def detect_outliers(self, 
                       df: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = None) -> Dict[str, np.ndarray]:
        """
        Detect outliers in specified columns using various methods.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        columns : Optional[List[str]]
            Columns to check for outliers (if None, use all numeric)
        method : str, optional
            Detection method ('percentile', 'iqr', 'zscore')
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary mapping column names to outlier masks
        """
        if method is None:
            method = self.outlier_method
            
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_masks = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            data = df[col].dropna()
            if len(data) == 0:
                continue
            
            if method == "percentile":
                lower_bound = data.quantile(self.outlier_percentiles["low"])
                upper_bound = data.quantile(self.outlier_percentiles["high"])
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == "iqr":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
                # Map z-scores back to original dataframe
                outlier_mask = pd.Series(False, index=df.index)
                outlier_mask.loc[data.index] = z_scores > 3
                
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            outlier_masks[col] = outlier_mask
        
        return outlier_masks
    
    def remove_outliers(self, 
                       df: pd.DataFrame, 
                       target_col: str,
                       columns: Optional[List[str]] = None,
                       method: str = None) -> pd.DataFrame:
        """
        Remove outliers from specified columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        target_col : str
            Target column name (protected from outlier removal)
        columns : Optional[List[str]]
            Columns to process for outliers
        method : str, optional
            Detection method
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with outliers removed
        """
        if columns is None:
            # Process all numeric columns except target
            columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                      if col != target_col]
        
        print(f"\n🎯 Outlier Detection and Removal:")
        print(f"  • Method: {method or self.outlier_method}")
        print(f"  • Processing {len(columns)} columns")
        
        initial_len = len(df)
        
        # Detect outliers
        outlier_masks = self.detect_outliers(df, columns, method)
        
        # Combine all outlier masks (row is outlier if ANY column has outlier)
        combined_mask = pd.Series(False, index=df.index)
        outlier_counts = {}
        
        for col, mask in outlier_masks.items():
            outlier_count = mask.sum()
            outlier_counts[col] = outlier_count
            combined_mask |= mask
        
        # Remove outlier rows
        df_clean = df[~combined_mask].copy()
        removed_count = initial_len - len(df_clean)
        
        print(f"  • Total outlier rows removed: {removed_count:,} ({removed_count/initial_len*100:.1f}%)")
        print(f"  • Remaining rows: {len(df_clean):,}")
        
        # Show outlier counts by column
        if outlier_counts:
            top_outlier_cols = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  • Top columns with outliers:")
            for col, count in top_outlier_cols:
                if count > 0:
                    print(f"    - {col}: {count:,} outliers")
        
        # Update cleaning stats
        self.cleaning_stats['outlier_removal'] = {
            'method': method or self.outlier_method,
            'columns_processed': len(columns),
            'total_outliers_removed': removed_count,
            'removal_percentage': removed_count/initial_len*100,
            'outlier_counts_by_column': outlier_counts
        }
        
        return df_clean
    
    def process_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process categorical variables by creating dummy variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with processed categorical variables
        """
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_cols:
            print("🔢 No categorical columns found")
            return df
        
        print(f"\n🔢 Processing categorical variables:")
        print(f"  • Found {len(categorical_cols)} categorical columns: {categorical_cols}")
        
        df_processed = df.copy()
        dummy_info = {}
        
        for col in categorical_cols:
            unique_values = df_processed[col].nunique()
            print(f"  • {col}: {unique_values} unique values")
            
            # Create dummy variables with prefix to avoid naming conflicts
            if unique_values > 1 and unique_values < 20:  # Reasonable limit for dummies
                dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed = df_processed.drop(columns=[col])
                
                dummy_info[col] = {
                    'unique_values': unique_values,
                    'dummy_columns': list(dummies.columns),
                    'dummy_count': len(dummies.columns)
                }
                
                print(f"    Created {len(dummies.columns)} dummy variables")
            else:
                print(f"    Skipped (too many unique values: {unique_values})")
        
        total_dummies = sum(info['dummy_count'] for info in dummy_info.values())
        
        # Update cleaning stats
        self.cleaning_stats['categorical_processing'] = {
            'categorical_columns': len(categorical_cols),
            'processed_columns': len(dummy_info),
            'total_dummies_created': total_dummies,
            'dummy_info': dummy_info
        }
        
        print(f"  • Total dummy variables created: {total_dummies}")
        print(f"  • Final DataFrame shape: {df_processed.shape}")
        
        return df_processed
    
    def clean_data(self, 
                   df: pd.DataFrame, 
                   remove_outliers: bool = False,
                   outlier_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, str]:
        """
        Complete data cleaning pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw input DataFrame
        remove_outliers : bool
            Whether to remove outliers
        outlier_columns : Optional[List[str]]
            Specific columns for outlier removal
            
        Returns:
        --------
        Tuple[pd.DataFrame, str]
            (cleaned_DataFrame, target_column_name)
        """
        with Timer("Data cleaning", verbose=True):
            print(f"🧹 Starting data cleaning pipeline")
            print(f"  • Initial shape: {df.shape}")
            
            # Initialize cleaning stats
            self.cleaning_stats = {'initial_shape': df.shape}
            
            # Step 1: Identify target column
            target_col = self.identify_target_column(df)
            
            # Step 2: Validate vibration data
            df_clean = self.validate_vibration_data(df, target_col)
            
            # Step 3: Handle missing data
            df_clean = self.handle_missing_data(df_clean)
            
            # Step 4: Process categorical variables
            df_clean = self.process_categorical_variables(df_clean)
            
            # Step 5: Optional outlier removal
            if remove_outliers:
                df_clean = self.remove_outliers(df_clean, target_col, outlier_columns)
            else:
                print(f"\n⚠️ Outlier removal skipped (remove_outliers=False)")
            
            # Final validation and statistics
            self.cleaning_stats['final_shape'] = df_clean.shape
            self._log_cleaning_summary(target_col)
            
            return df_clean, target_col
    
    def _log_cleaning_summary(self, target_col: str) -> None:
        """Log comprehensive cleaning summary."""
        stats = self.cleaning_stats
        
        print(f"\n✅ Data cleaning completed successfully!")
        print(f"📊 Cleaning Summary:")
        print(f"  • Initial shape: {stats['initial_shape']}")
        print(f"  • Final shape: {stats['final_shape']}")
        print(f"  • Rows removed: {stats['initial_shape'][0] - stats['final_shape'][0]:,}")
        print(f"  • Columns changed: {stats['initial_shape'][1]} → {stats['final_shape'][1]}")
        print(f"  • Target column: {target_col}")
        
        # Data reduction percentage
        row_reduction = ((stats['initial_shape'][0] - stats['final_shape'][0]) / 
                        stats['initial_shape'][0] * 100)
        print(f"  • Data reduction: {row_reduction:.1f}%")
        
        # Memory usage estimate
        estimated_memory = (stats['final_shape'][0] * stats['final_shape'][1] * 8) / (1024**2)
        print(f"  • Estimated memory: {estimated_memory:.1f} MB")
    
    def get_cleaning_report(self) -> Dict:
        """Get comprehensive cleaning statistics."""
        return {
            'cleaning_stats': self.cleaning_stats.copy(),
            'config': {
                'vibration_range': (self.vibration_min, self.vibration_max),
                'missing_threshold': self.missing_threshold,
                'outlier_method': self.outlier_method
            }
        }


# Convenience function for direct usage
def clean_data(df: pd.DataFrame, 
               remove_outliers: bool = False,
               **kwargs) -> Tuple[pd.DataFrame, str]:
    """
    Convenience function to clean data directly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw input DataFrame
    remove_outliers : bool
        Whether to remove outliers
    **kwargs
        Additional parameters for DataCleaner
        
    Returns:
    --------
    Tuple[pd.DataFrame, str]
        (cleaned_DataFrame, target_column_name)
    """
    cleaner = DataCleaner(**kwargs)
    return cleaner.clean_data(df, remove_outliers=remove_outliers)