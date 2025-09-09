"""
Model Evaluation Visualizer
===========================

Comprehensive visualization tools for model evaluation including
performance plots, residual analysis, and comparison visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from config.settings import get_settings
from utils.plotting import setup_plot_style, save_plot

logger = logging.getLogger(__name__)
settings = get_settings()

# Set up plotting style
setup_plot_style()


class ModelVisualizer:
    """
    Comprehensive visualization tools for model evaluation.
    
    This class provides various plotting methods for model performance
    assessment, residual analysis, and comparison visualizations.
    """
    
    def __init__(self, 
                 figure_size: Tuple[int, int] = (12, 8),
                 save_plots: bool = False,
                 output_dir: Optional[str] = None):
        """
        Initialize the model visualizer.
        
        Parameters:
        -----------
        figure_size : Tuple[int, int], default=(12, 8)
            Default figure size for plots
        save_plots : bool, default=False
            Whether to automatically save plots
        output_dir : Optional[str], default=None
            Directory to save plots (if save_plots=True)
        """
        self.figure_size = figure_size
        self.save_plots = save_plots
        self.output_dir = Path(output_dir) if output_dir else Path("plots")
        
        if self.save_plots:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug("ModelVisualizer initialized")
    
    def plot_predictions_vs_actual(self,
                                  y_true: Union[pd.Series, np.ndarray],
                                  y_pred: np.ndarray,
                                  dataset_name: str = "Test",
                                  model_name: str = "Model",
                                  ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Create predictions vs actual scatter plot.
        
        Parameters:
        -----------
        y_true : Union[pd.Series, np.ndarray]
            True target values
        y_pred : np.ndarray
            Predicted values
        dataset_name : str, default="Test"
            Name of the dataset
        model_name : str, default="Model"
            Name of the model
        ax : Optional[plt.Axes], default=None
            Matplotlib axes to plot on
            
        Returns:
        --------
        plt.Axes
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Create scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        # Calculate and display R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{model_name}: {dataset_name} Set Predictions vs Actual\\nR² = {r2:.4f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ensure equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        if self.save_plots:
            filename = f"{model_name}_{dataset_name}_predictions_vs_actual.png"
            save_plot(self.output_dir / filename)
        
        return ax
    
    def plot_residuals_analysis(self,
                               y_true: Union[pd.Series, np.ndarray],
                               y_pred: np.ndarray,
                               dataset_name: str = "Test",
                               model_name: str = "Model",
                               fig: Optional[plt.Figure] = None) -> plt.Figure:
        """
        Create comprehensive residuals analysis plot.
        
        Parameters:
        -----------
        y_true : Union[pd.Series, np.ndarray]
            True target values
        y_pred : np.ndarray
            Predicted values
        dataset_name : str, default="Test"
            Name of the dataset
        model_name : str, default="Model"
            Name of the model
        fig : Optional[plt.Figure], default=None
            Matplotlib figure to plot on
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        if fig is None:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        else:
            axes = fig.get_axes()
            if len(axes) < 4:
                raise ValueError("Figure must have at least 4 subplots for residuals analysis")
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        residuals = y_true - y_pred
        
        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title(f'{model_name}: Residuals vs Predicted\\n{dataset_name} Set')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals histogram
        axes[0, 1].hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=np.mean(residuals), color='r', linestyle='--', alpha=0.8, label=f'Mean: {np.mean(residuals):.4f}')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title(f'Residuals Distribution\\nStd: {np.std(residuals):.4f}')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. Q-Q plot for normality
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Residuals vs Index (if time series)
        indices = np.arange(len(residuals))
        axes[1, 1].plot(indices, residuals, alpha=0.7, linewidth=0.8)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 1].set_xlabel('Observation Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Observation Order')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(f'{model_name} - Residuals Analysis ({dataset_name} Set)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"{model_name}_{dataset_name}_residuals_analysis.png"
            save_plot(self.output_dir / filename)
        
        return fig
    
    def plot_model_comparison(self,
                             model_results: pd.DataFrame,
                             metric: str = 'test_r2',
                             top_n: Optional[int] = None,
                             fig: Optional[plt.Figure] = None) -> plt.Figure:
        """
        Create model comparison visualization.
        
        Parameters:
        -----------
        model_results : pd.DataFrame
            DataFrame with model results
        metric : str, default='test_r2'
            Metric to use for comparison
        top_n : Optional[int], default=None
            Number of top models to show
        fig : Optional[plt.Figure], default=None
            Matplotlib figure to plot on
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        if fig is None:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        else:
            axes = fig.get_axes()
            if len(axes) < 4:
                raise ValueError("Figure must have at least 4 subplots for model comparison")
        
        # Sort and limit models if specified
        results_df = model_results.copy()
        if metric in results_df.columns:
            results_df = results_df.sort_values(metric, ascending=False)
        
        if top_n:
            results_df = results_df.head(top_n)
        
        model_names = results_df.index if results_df.index.name else range(len(results_df))
        
        # 1. R² Score Comparison (if available)
        if 'train_r2' in results_df.columns and 'test_r2' in results_df.columns:
            x = np.arange(len(results_df))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, results_df['train_r2'], width, label='Train R²', alpha=0.8)
            axes[0, 0].bar(x + width/2, results_df['test_r2'], width, label='Test R²', alpha=0.8)
            axes[0, 0].set_xlabel('Models')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].set_title('R² Score Comparison')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels([str(name)[:10] for name in model_names], rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. RMSE Comparison (if available)
        if 'train_rmse' in results_df.columns and 'test_rmse' in results_df.columns:
            x = np.arange(len(results_df))
            
            axes[0, 1].bar(x - width/2, results_df['train_rmse'], width, label='Train RMSE', alpha=0.8)
            axes[0, 1].bar(x + width/2, results_df['test_rmse'], width, label='Test RMSE', alpha=0.8)
            axes[0, 1].set_xlabel('Models')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].set_title('RMSE Comparison')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels([str(name)[:10] for name in model_names], rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Overfitting Analysis (if available)
        if 'overfitting' in results_df.columns:
            colors = ['green' if x <= 0.1 else 'orange' if x <= 0.2 else 'red' 
                     for x in results_df['overfitting']]
            
            axes[1, 0].bar(model_names, results_df['overfitting'], color=colors, alpha=0.7)
            axes[1, 0].set_xlabel('Models')
            axes[1, 0].set_ylabel('Overfitting (Train R² - Test R²)')
            axes[1, 0].set_title('Overfitting Analysis')
            axes[1, 0].set_xticklabels([str(name)[:10] for name in model_names], rotation=45)
            axes[1, 0].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Moderate (0.1)')
            axes[1, 0].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='High (0.2)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Performance vs Training Time (if available)
        if 'training_time' in results_df.columns and 'test_r2' in results_df.columns:
            scatter = axes[1, 1].scatter(results_df['training_time'], results_df['test_r2'], s=100, alpha=0.7)
            for i, name in enumerate(model_names):
                axes[1, 1].annotate(str(name)[:8], 
                                   (results_df.iloc[i]['training_time'], results_df.iloc[i]['test_r2']),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[1, 1].set_xlabel('Training Time (seconds)')
            axes[1, 1].set_ylabel('Test R²')
            axes[1, 1].set_title('Performance vs Training Time')
            axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"model_comparison_{metric}.png"
            save_plot(self.output_dir / filename)
        
        return fig
    
    def plot_time_series_evaluation(self,
                                   y_true: Union[pd.Series, np.ndarray],
                                   y_pred: np.ndarray,
                                   time_index: Optional[pd.DatetimeIndex] = None,
                                   model_name: str = "Model",
                                   dataset_name: str = "Test",
                                   chunk_size: int = 2000,
                                   fig: Optional[plt.Figure] = None) -> plt.Figure:
        """
        Create time series evaluation plots.
        
        Parameters:
        -----------
        y_true : Union[pd.Series, np.ndarray]
            True target values
        y_pred : np.ndarray
            Predicted values
        time_index : Optional[pd.DatetimeIndex], default=None
            Time index for the data
        model_name : str, default="Model"
            Name of the model
        dataset_name : str, default="Test"
            Name of the dataset
        chunk_size : int, default=2000
            Size of chunks for visualization
        fig : Optional[plt.Figure], default=None
            Matplotlib figure to plot on
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if time_index is None:
            time_index = pd.RangeIndex(len(y_true))
        
        # Determine number of chunks
        n_chunks = (len(y_true) + chunk_size - 1) // chunk_size
        n_cols = 2
        n_rows = (n_chunks + n_cols - 1) // n_cols
        
        if fig is None:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
        else:
            axes = fig.get_axes()
        
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        # Plot each chunk
        for i in range(n_chunks):
            if i >= len(axes_flat):
                break
                
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(y_true))
            
            chunk_time = time_index[start_idx:end_idx]
            chunk_true = y_true[start_idx:end_idx]
            chunk_pred = y_pred[start_idx:end_idx]
            
            axes_flat[i].plot(chunk_time, chunk_true, label='Actual', alpha=0.7, linewidth=1.5)
            axes_flat[i].plot(chunk_time, chunk_pred, label='Predicted', alpha=0.8, linewidth=1.5)
            
            # Calculate R² for this chunk
            from sklearn.metrics import r2_score
            chunk_r2 = r2_score(chunk_true, chunk_pred)
            
            axes_flat[i].set_ylabel('Values')
            axes_flat[i].set_title(f'Chunk {i+1}/{n_chunks} (R² = {chunk_r2:.3f})')
            axes_flat[i].grid(True, alpha=0.3)
            
            if i == 0:  # Only show legend on first plot
                axes_flat[i].legend()
        
        # Hide unused subplots
        for i in range(n_chunks, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        fig.suptitle(f'{model_name}: Time Series Evaluation ({dataset_name} Set)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"{model_name}_{dataset_name}_time_series.png"
            save_plot(self.output_dir / filename)
        
        return fig
    
    def plot_feature_importance(self,
                               feature_importance: Dict[str, float],
                               model_name: str = "Model",
                               top_n: int = 20,
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot feature importance.
        
        Parameters:
        -----------
        feature_importance : Dict[str, float]
            Dictionary of feature names and importance scores
        model_name : str, default="Model"
            Name of the model
        top_n : int, default=20
            Number of top features to show
        ax : Optional[plt.Axes], default=None
            Matplotlib axes to plot on
            
        Returns:
        --------
        plt.Axes
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Sort and select top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f[:30] + '...' if len(f) > 30 else f for f in features])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{model_name}: Top {len(top_features)} Feature Importance')
        ax.grid(True, alpha=0.3)
        
        if self.save_plots:
            filename = f"{model_name}_feature_importance.png"
            save_plot(self.output_dir / filename)
        
        return ax


def create_evaluation_plots(y_train: Union[pd.Series, np.ndarray],
                           train_pred: np.ndarray,
                           y_test: Union[pd.Series, np.ndarray],
                           test_pred: np.ndarray,
                           model_name: str = "Model",
                           feature_importance: Optional[Dict[str, float]] = None,
                           time_index: Optional[pd.DatetimeIndex] = None,
                           save_plots: bool = False,
                           output_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """
    Create comprehensive evaluation plots for a model.
    
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
    model_name : str, default="Model"
        Name of the model
    feature_importance : Optional[Dict[str, float]], default=None
        Feature importance scores
    time_index : Optional[pd.DatetimeIndex], default=None
        Time index for time series plots
    save_plots : bool, default=False
        Whether to save plots
    output_dir : Optional[str], default=None
        Directory to save plots
        
    Returns:
    --------
    Dict[str, plt.Figure]
        Dictionary of created figures
    """
    visualizer = ModelVisualizer(save_plots=save_plots, output_dir=output_dir)
    
    figures = {}
    
    # 1. Predictions vs Actual (combined)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    visualizer.plot_predictions_vs_actual(y_train, train_pred, "Training", model_name, ax1)
    visualizer.plot_predictions_vs_actual(y_test, test_pred, "Test", model_name, ax2)
    fig.suptitle(f'{model_name}: Predictions vs Actual', fontsize=16, fontweight='bold')
    plt.tight_layout()
    figures['predictions_vs_actual'] = fig
    
    # 2. Residuals Analysis
    fig_residuals = visualizer.plot_residuals_analysis(y_test, test_pred, "Test", model_name)
    figures['residuals_analysis'] = fig_residuals
    
    # 3. Time Series Evaluation (if time index provided)
    if time_index is not None and len(time_index) == len(y_test):
        fig_ts = visualizer.plot_time_series_evaluation(y_test, test_pred, time_index, model_name)
        figures['time_series'] = fig_ts
    
    # 4. Feature Importance (if provided)
    if feature_importance:
        fig, ax = plt.subplots(figsize=(12, 8))
        visualizer.plot_feature_importance(feature_importance, model_name, ax=ax)
        figures['feature_importance'] = fig
    
    logger.info(f"Created {len(figures)} evaluation plots for {model_name}")
    
    return figures