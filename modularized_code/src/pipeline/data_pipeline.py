"""
Data Processing Pipeline
========================

Complete data processing pipeline for industrial vibration prediction.
Handles data loading, cleaning, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pickle

from config.settings import get_settings
from data.loader import DataLoader
from data.cleaner import DataCleaner
from data.preprocessor import DataPreprocessor
from features.engineer import FeatureEngineer
from features.selector import FeatureSelector
from utils.helpers import Timer

logger = logging.getLogger(__name__)
settings = get_settings()


class DataPipeline:
    """
    Complete data processing pipeline for vibration prediction.
    
    This class orchestrates the complete data workflow from raw data
    loading through feature engineering and selection.
    """
    
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 random_state: int = 42):
        """
        Initialize data pipeline.
        
        Parameters:
        -----------
        config : Optional[Dict[str, Any]], default=None
            Configuration dictionary for pipeline components
        random_state : int, default=42
            Random state for reproducibility
        """
        self.config = config or {}
        self.random_state = random_state
        
        # Pipeline components
        self.data_loader = None
        self.data_cleaner = None
        self.data_preprocessor = None
        self.feature_engineer = None
        self.feature_selector = None
        
        # Pipeline state
        self.raw_data = None
        self.cleaned_data = None
        self.preprocessed_data = None
        self.engineered_features = None
        self.selected_features = None
        self.train_test_split = None
        
        # Metadata
        self.pipeline_metadata = {
            'steps_completed': [],
            'execution_times': {},
            'data_shapes': {},
            'feature_counts': {}
        }
        
        logger.info("DataPipeline initialized")
    
    def load_data(self, 
                  data_path: Union[str, Path],
                  **loader_kwargs) -> pd.DataFrame:
        """
        Load raw data from file.
        
        Parameters:
        -----------
        data_path : Union[str, Path]
            Path to data file
        **loader_kwargs
            Additional arguments for data loader
            
        Returns:
        --------
        pd.DataFrame
            Loaded raw data
        """
        logger.info(f"Loading data from {data_path}")
        
        with Timer("Data loading") as timer:
            self.data_loader = DataLoader()
            self.raw_data = self.data_loader.load_from_csv(data_path, **loader_kwargs)
        
        # Update metadata
        self._update_metadata("load_data", timer.elapsed_time, self.raw_data.shape)
        
        logger.info(f"Data loaded: {self.raw_data.shape[0]:,} rows, {self.raw_data.shape[1]} columns")
        
        return self.raw_data
    
    def clean_data(self, 
                   target_column: str = 'vibration',
                   **cleaner_kwargs) -> pd.DataFrame:
        """
        Clean the loaded data.
        
        Parameters:
        -----------
        target_column : str, default='vibration'
            Name of target column
        **cleaner_kwargs
            Additional arguments for data cleaner
            
        Returns:
        --------
        pd.DataFrame
            Cleaned data
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Cleaning data")
        
        with Timer("Data cleaning") as timer:
            self.data_cleaner = DataCleaner()
            self.cleaned_data, detected_target = self.data_cleaner.clean_data(
                self.raw_data, 
                **cleaner_kwargs
            )
        
        # Update metadata
        self._update_metadata("clean_data", timer.elapsed_time, self.cleaned_data.shape)
        
        # Log cleaning summary
        cleaning_report = self.data_cleaner.get_cleaning_report()
        logger.info(f"Data cleaning complete:")
        logger.info(f"  • Detected target column: {detected_target}")
        logger.info(f"  • Rows removed: {cleaning_report.get('rows_removed', 0):,}")
        logger.info(f"  • Missing values handled: {cleaning_report.get('missing_values_handled', 0):,}")
        logger.info(f"  • Final shape: {self.cleaned_data.shape}")
        
        return self.cleaned_data
    
    def preprocess_data(self, 
                       target_column: str = 'vibration',
                       test_size: float = 0.2,
                       **preprocessor_kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Preprocess data and create train/test split.
        
        Parameters:
        -----------
        target_column : str, default='vibration'
            Name of target column
        test_size : float, default=0.2
            Fraction of data for testing
        **preprocessor_kwargs
            Additional arguments for preprocessor
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Call clean_data() first.")
        
        logger.info("Preprocessing data")
        
        with Timer("Data preprocessing") as timer:
            self.data_preprocessor = DataPreprocessor()
            preprocessing_results = self.data_preprocessor.preprocess_pipeline(
                self.cleaned_data,
                target_col=target_column,  # Note: using target_col not target_column
                validation_split=False,
                **preprocessor_kwargs
            )
            
            # Extract results from preprocessing pipeline
            X_train = preprocessing_results.get('X_train')
            X_test = preprocessing_results.get('X_test')  
            y_train = preprocessing_results.get('y_train')
            y_test = preprocessing_results.get('y_test')
            
            # Reconstruct the full preprocessed dataset for feature engineering
            import pandas as pd
            self.preprocessed_data = pd.concat([
                pd.concat([X_train, y_train], axis=1),
                pd.concat([X_test, y_test], axis=1)
            ], axis=0).sort_index()
            
            # Check if validation data was created
            X_val = preprocessing_results.get('X_val')
            y_val = preprocessing_results.get('y_val')
            
            if X_val is not None and y_val is not None:
                logger.info(f"  • Validation set created: {X_val.shape}")
            else:
                logger.info("  • No validation set created")
        
        # Store split data
        self.train_test_split = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        # Update metadata
        preprocessed_shape = self.preprocessed_data.shape if self.preprocessed_data is not None else (0, 0)
        self._update_metadata("preprocess_data", timer.elapsed_time, preprocessed_shape)
        self.pipeline_metadata['data_shapes']['train'] = X_train.shape
        self.pipeline_metadata['data_shapes']['test'] = X_test.shape
        
        logger.info(f"Data preprocessing complete:")
        if self.preprocessed_data is not None:
            logger.info(f"  • Preprocessed shape: {self.preprocessed_data.shape}")
        logger.info(f"  • Training set: {X_train.shape}")
        logger.info(f"  • Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def engineer_features(self,
                         target_column: str = 'vibration',
                         **engineer_kwargs) -> pd.DataFrame:
        """
        Engineer features from preprocessed data.
        
        Parameters:
        -----------
        target_column : str, default='vibration'
            Name of target column
        **engineer_kwargs
            Additional arguments for feature engineer
            
        Returns:
        --------
        pd.DataFrame
            Data with engineered features
        """
        if self.preprocessed_data is None:
            raise ValueError("No preprocessed data available. Call preprocess_data() first.")
        
        logger.info("Engineering features")
        
        with Timer("Feature engineering") as timer:
            self.feature_engineer = FeatureEngineer()
            self.engineered_features = self.feature_engineer.engineer_features(
                self.preprocessed_data,
                target_col=target_column,
                **engineer_kwargs
            )
        
        # Update metadata
        self._update_metadata("engineer_features", timer.elapsed_time, self.engineered_features.shape)
        
        # Log feature engineering summary
        engineering_summary = self.feature_engineer.get_engineering_stats()
        logger.info(f"Feature engineering complete:")
        logger.info(f"  • Original features: {self.preprocessed_data.shape[1]}")
        logger.info(f"  • Engineered features: {self.engineered_features.shape[1]}")
        logger.info(f"  • Features added: {self.engineered_features.shape[1] - self.preprocessed_data.shape[1]}")
        
        for feature_type, count in engineering_summary['feature_types'].items():
            logger.info(f"  • {feature_type}: {count} features")
        
        return self.engineered_features
    
    def select_features(self,
                       target_column: str = 'vibration',
                       selection_strategy: str = 'balanced',
                       **selector_kwargs) -> List[str]:
        """
        Select optimal features for modeling.
        
        Parameters:
        -----------
        target_column : str, default='vibration'
            Name of target column
        selection_strategy : str, default='balanced'
            Feature selection strategy
        **selector_kwargs
            Additional arguments for feature selector
            
        Returns:
        --------
        List[str]
            Selected feature names
        """
        if self.engineered_features is None:
            raise ValueError("No engineered features available. Call engineer_features() first.")
        
        if self.train_test_split is None:
            raise ValueError("No train/test split available. Call preprocess_data() first.")
        
        logger.info(f"Selecting features using {selection_strategy} strategy")
        
        with Timer("Feature selection") as timer:
            # Create feature selector
            self.feature_selector = FeatureSelector()
            
            # Prepare feature data for selection
            feature_data = self.engineered_features.drop(columns=[target_column])
            target_data = self.engineered_features[target_column]
            
            # Split features according to existing train/test split
            train_indices = self.train_test_split['X_train'].index
            test_indices = self.train_test_split['X_test'].index
            
            # Use .loc to match by index values instead of positions
            X_train_features = feature_data.loc[train_indices]
            X_test_features = feature_data.loc[test_indices] 
            y_train_features = target_data.loc[train_indices]
            y_test_features = target_data.loc[test_indices]
            
            # Fit selector and get recommendations
            self.feature_selector.fit(X_train_features, X_test_features, y_train_features, y_test_features)
            _, selected_features = self.feature_selector.recommend_feature_set(selection_strategy)
        
        # Store selected features
        self.selected_features = selected_features
        
        # Update metadata
        self._update_metadata("select_features", timer.elapsed_time)
        self.pipeline_metadata['feature_counts']['selected'] = len(selected_features)
        
        # Validate selection
        validation = self.feature_selector.validate_feature_selection(selected_features, target_column)
        
        logger.info(f"Feature selection complete:")
        logger.info(f"  • Strategy: {selection_strategy}")
        logger.info(f"  • Features selected: {len(selected_features)}")
        logger.info(f"  • Data leakage check: {'✅ Clean' if validation['data_leakage_check']['is_clean'] else '⚠️ Potential issues'}")
        
        return selected_features
    
    def get_final_data(self, 
                      target_column: str = 'vibration') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Get final processed data ready for modeling.
        
        Parameters:
        -----------
        target_column : str, default='vibration'
            Name of target column
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test with selected features
        """
        if self.selected_features is None:
            raise ValueError("No features selected. Call select_features() first.")
        
        if self.engineered_features is None:
            raise ValueError("No engineered features available.")
        
        # Create final dataset with selected features only
        final_data = self.engineered_features[self.selected_features + [target_column]]
        
        # Split according to existing train/test split
        train_indices = self.train_test_split['X_train'].index
        test_indices = self.train_test_split['X_test'].index
        
        X_train_final = final_data.loc[train_indices].drop(columns=[target_column])
        X_test_final = final_data.loc[test_indices].drop(columns=[target_column])
        y_train_final = final_data.loc[train_indices][target_column]
        y_test_final = final_data.loc[test_indices][target_column]
        
        logger.info(f"Final data ready for modeling:")
        logger.info(f"  • Training set: {X_train_final.shape}")
        logger.info(f"  • Test set: {X_test_final.shape}")
        logger.info(f"  • Selected features: {len(self.selected_features)}")
        
        return X_train_final, X_test_final, y_train_final, y_test_final
    
    def run_complete_pipeline(self,
                             data_path: Union[str, Path],
                             target_column: str = 'vibration',
                             test_size: float = 0.2,
                             selection_strategy: str = 'balanced') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Run the complete data pipeline in one call.
        
        Parameters:
        -----------
        data_path : Union[str, Path]
            Path to data file
        target_column : str, default='vibration'
            Name of target column
        test_size : float, default=0.2
            Test set size
        selection_strategy : str, default='balanced'
            Feature selection strategy
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            Final processed data ready for modeling
        """
        logger.info("Starting complete data pipeline")
        
        pipeline_start_time = Timer("Complete data pipeline")
        pipeline_start_time.__enter__()
        
        try:
            # Step 1: Load data
            self.load_data(data_path)
            
            # Step 2: Clean data
            self.clean_data(target_column=target_column)
            
            # Step 3: Preprocess data
            self.preprocess_data(target_column=target_column, test_size=test_size)
            
            # Step 4: Engineer features
            self.engineer_features(target_column=target_column)
            
            # Step 5: Select features
            self.select_features(target_column=target_column, selection_strategy=selection_strategy)
            
            # Step 6: Get final data
            final_data = self.get_final_data(target_column=target_column)
            
        except Exception as e:
            logger.error(f"Pipeline failed at step: {e}")
            raise
        
        finally:
            pipeline_start_time.__exit__(None, None, None)
            total_time = pipeline_start_time.elapsed_time
            self.pipeline_metadata['total_execution_time'] = total_time
        
        logger.info(f"Complete data pipeline finished in {total_time:.2f}s")
        
        return final_data
    
    def save_pipeline_state(self, filepath: Union[str, Path]) -> None:
        """
        Save the current pipeline state to disk.
        
        Parameters:
        -----------
        filepath : Union[str, Path]
            Path to save pipeline state
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        pipeline_state = {
            'config': self.config,
            'random_state': self.random_state,
            'metadata': self.pipeline_metadata,
            'selected_features': self.selected_features,
            'train_test_split_indices': {
                'train_indices': self.train_test_split['X_train'].index.tolist() if self.train_test_split else None,
                'test_indices': self.train_test_split['X_test'].index.tolist() if self.train_test_split else None
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_state, f)
        
        logger.info(f"Pipeline state saved to {filepath}")
    
    def load_pipeline_state(self, filepath: Union[str, Path]) -> None:
        """
        Load pipeline state from disk.
        
        Parameters:
        -----------
        filepath : Union[str, Path]
            Path to load pipeline state from
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline state file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            pipeline_state = pickle.load(f)
        
        self.config = pipeline_state['config']
        self.random_state = pipeline_state['random_state']
        self.pipeline_metadata = pipeline_state['metadata']
        self.selected_features = pipeline_state['selected_features']
        
        logger.info(f"Pipeline state loaded from {filepath}")
    
    def _update_metadata(self, step_name: str, execution_time: float, data_shape: Optional[Tuple] = None):
        """Update pipeline metadata."""
        self.pipeline_metadata['steps_completed'].append(step_name)
        self.pipeline_metadata['execution_times'][step_name] = execution_time
        
        if data_shape:
            self.pipeline_metadata['data_shapes'][step_name] = data_shape
            self.pipeline_metadata['feature_counts'][step_name] = data_shape[1]
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline execution summary.
        
        Returns:
        --------
        Dict[str, Any]
            Pipeline execution summary
        """
        summary = {
            'steps_completed': len(self.pipeline_metadata['steps_completed']),
            'total_steps': 6,  # load, clean, preprocess, engineer, select, finalize
            'execution_summary': self.pipeline_metadata.copy()
        }
        
        # Add current data info
        if self.selected_features:
            summary['final_feature_count'] = len(self.selected_features)
        
        if self.train_test_split:
            summary['train_samples'] = len(self.train_test_split['X_train'])
            summary['test_samples'] = len(self.train_test_split['X_test'])
        
        return summary


def create_data_pipeline(config: Optional[Dict[str, Any]] = None,
                        random_state: int = 42) -> DataPipeline:
    """
    Create a data pipeline with optional configuration.
    
    Parameters:
    -----------
    config : Optional[Dict[str, Any]], default=None
        Configuration for pipeline components
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    DataPipeline
        Configured data pipeline instance
    """
    return DataPipeline(config=config, random_state=random_state)