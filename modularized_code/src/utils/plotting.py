"""
Plotting Utilities
=================

Standardized plotting functions for the vibration prediction system.
All plots follow consistent styling and include comprehensive customization options.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from math import ceil

from config.settings import VISUALIZATION_CONFIG


def setup_matplotlib_style(
    style: str = "default",
    palette: str = "husl",
    figure_size: Tuple[int, int] = (12, 8)
) -> None:
    """
    Setup matplotlib and seaborn styling consistently.
    
    Parameters:
    -----------
    style : str
        Matplotlib style name
    palette : str
        Seaborn color palette
    figure_size : Tuple[int, int]
        Default figure size
    """
    plt.style.use(style)
    sns.set_palette(palette)
    plt.rcParams['figure.figsize'] = figure_size
    plt.rcParams['savefig.dpi'] = VISUALIZATION_CONFIG['dpi']
    plt.rcParams['savefig.bbox'] = VISUALIZATION_CONFIG['bbox_inches']


def plot_time_series(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Time Series Plot",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_stats: bool = True,
    max_points: Optional[int] = None
) -> plt.Figure:
    """
    Plot time series data with optional statistics overlay.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    columns : Optional[List[str]]
        Columns to plot (if None, plot all numeric columns)
    title : str
        Plot title
    figsize : Optional[Tuple[int, int]]
        Figure size
    save_path : Optional[Union[str, Path]]
        Path to save the plot
    show_stats : bool
        Whether to show statistics text box
    max_points : Optional[int]
        Maximum number of points to plot (for performance)
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    if figsize is None:
        figsize = VISUALIZATION_CONFIG['figure_size']
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Sample data if too many points
    plot_df = df.copy()
    if max_points and len(plot_df) > max_points:
        step = len(plot_df) // max_points
        plot_df = plot_df.iloc[::step]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in columns:
        if col in plot_df.columns:
            ax.plot(plot_df.index, plot_df[col], 
                   label=col, alpha=VISUALIZATION_CONFIG['alpha'],
                   linewidth=VISUALIZATION_CONFIG['line_width'])
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    if len(columns) > 1:
        ax.legend()
    
    # Add statistics text box
    if show_stats and len(columns) == 1:
        col = columns[0]
        stats_text = (f'Mean: {df[col].mean():.3f}\n'
                     f'Std: {df[col].std():.3f}\n'
                     f'Min: {df[col].min():.3f}\n'
                     f'Max: {df[col].max():.3f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_histogram(
    data: Union[pd.Series, np.ndarray],
    title: str = "Histogram",
    bins: Union[int, str] = 50,
    density: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot histogram with optional density curve.
    
    Parameters:
    -----------
    data : Union[pd.Series, np.ndarray]
        Data to plot
    title : str
        Plot title
    bins : Union[int, str]
        Number of bins or binning strategy
    density : bool
        Whether to show density curve overlay
    figsize : Optional[Tuple[int, int]]
        Figure size
    save_path : Optional[Union[str, Path]]
        Path to save the plot
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    if figsize is None:
        figsize = VISUALIZATION_CONFIG['figure_size']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Remove NaN values
    clean_data = pd.Series(data).dropna()
    
    if len(clean_data) == 0:
        ax.text(0.5, 0.5, 'No data available', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title(title, fontsize=14, fontweight='bold')
        return fig
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(clean_data, bins=bins, alpha=0.7,
                                   edgecolor='black', linewidth=0.5)
    
    # Add density curve if requested and enough data
    if density and len(clean_data) > 10:
        try:
            ax2 = ax.twinx()
            clean_data.plot.density(ax=ax2, color='red', linewidth=2, alpha=0.8)
            ax2.set_ylabel('Density', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        except Exception:
            pass  # Skip density if it fails
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f'Count: {len(clean_data):,}\n'
                  f'Mean: {clean_data.mean():.3f}\n'
                  f'Std: {clean_data.std():.3f}\n'
                  f'Min: {clean_data.min():.3f}\n'
                  f'Max: {clean_data.max():.3f}')
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Correlation Matrix",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    annot: bool = True,
    cmap: str = "RdBu_r"
) -> plt.Figure:
    """
    Plot correlation matrix heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to calculate correlations
    columns : Optional[List[str]]
        Columns to include (if None, use all numeric columns)
    title : str
        Plot title
    figsize : Optional[Tuple[int, int]]
        Figure size
    save_path : Optional[Union[str, Path]]
        Path to save the plot
    annot : bool
        Whether to annotate correlation values
    cmap : str
        Colormap for heatmap
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if figsize is None:
        # Adjust figure size based on number of features
        size = max(8, len(columns) * 0.5)
        figsize = (size, size)
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=annot, cmap=cmap,
                center=0, square=True, ax=ax,
                cbar_kws={"shrink": 0.8})
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    feature_col: str = "feature",
    importance_col: str = "importance"
) -> plt.Figure:
    """
    Plot feature importance as horizontal bar chart.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with feature importance data
    top_n : int
        Number of top features to show
    title : str
        Plot title
    figsize : Optional[Tuple[int, int]]
        Figure size
    save_path : Optional[Union[str, Path]]
        Path to save the plot
    feature_col : str
        Name of feature column
    importance_col : str
        Name of importance column
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    if figsize is None:
        figsize = (12, max(6, top_n * 0.4))
    
    # Get top N features
    top_features = importance_df.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_features)), top_features[importance_col])
    
    # Customize appearance
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f[:40] + '...' if len(f) > 40 else f 
                       for f in top_features[feature_col]])
    ax.set_xlabel('Importance')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
               f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metrics: List[str] = ["Test R²", "Test RMSE"],
    title: str = "Model Comparison",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot model performance comparison.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with model results
    metrics : List[str]
        Metrics to compare
    title : str
        Plot title
    figsize : Optional[Tuple[int, int]]
        Figure size
    save_path : Optional[Union[str, Path]]
        Path to save the plot
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    if figsize is None:
        figsize = (14, 8)
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    models = results_df['Model'].tolist()
    x = np.arange(len(models))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = results_df[metric].tolist()
        
        bars = ax.bar(x, values, alpha=0.7)
        ax.set_xlabel('Models')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m[:15] + '...' if len(m) > 15 else m for m in models],
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_residuals(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    title: str = "Residual Analysis",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot residual analysis (predicted vs actual and residuals vs predicted).
    
    Parameters:
    -----------
    y_true : Union[pd.Series, np.ndarray]
        True values
    y_pred : Union[pd.Series, np.ndarray]
        Predicted values
    title : str
        Plot title
    figsize : Optional[Tuple[int, int]]
        Figure size
    save_path : Optional[Union[str, Path]]
        Path to save the plot
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    if figsize is None:
        figsize = (14, 6)
    
    residuals = np.array(y_true) - np.array(y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Predicted vs Actual
    ax1.scatter(y_true, y_pred, alpha=VISUALIZATION_CONFIG['scatter_alpha'],
               s=VISUALIZATION_CONFIG['scatter_size'])
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predicted vs Actual')
    ax1.grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    ax2.scatter(y_pred, residuals, alpha=VISUALIZATION_CONFIG['scatter_alpha'],
               s=VISUALIZATION_CONFIG['scatter_size'])
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def save_plot(
    fig: plt.Figure,
    save_path: Union[str, Path],
    dpi: Optional[int] = None,
    format: str = "png",
    bbox_inches: str = "tight"
) -> None:
    """
    Save plot to file with consistent formatting.
    
    Parameters:
    -----------
    fig : plt.Figure
        Figure to save
    save_path : Union[str, Path]
        Path to save the file
    dpi : Optional[int]
        DPI for saved image
    format : str
        File format
    bbox_inches : str
        Bounding box setting
    """
    if dpi is None:
        dpi = VISUALIZATION_CONFIG['save_dpi']
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(save_path, dpi=dpi, format=format, bbox_inches=bbox_inches)
    print(f"📊 Plot saved to {save_path}")


def create_subplot_grid(
    n_plots: int,
    cols: int = 3,
    figsize_per_plot: Tuple[int, int] = (4, 3)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a grid of subplots for multiple plots.
    
    Parameters:
    -----------
    n_plots : int
        Number of plots needed
    cols : int
        Number of columns in grid
    figsize_per_plot : Tuple[int, int]
        Size of each subplot
        
    Returns:
    --------
    Tuple[plt.Figure, np.ndarray]
        Figure and axes array
    """
    rows = ceil(n_plots / cols)
    
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows)
    )
    
    # Ensure axes is always 2D array
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Hide unused subplots
    if n_plots < rows * cols:
        for i in range(n_plots, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
    
    return fig, axes


# Alias for backward compatibility
setup_plot_style = setup_matplotlib_style