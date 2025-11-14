"""
Detailed evaluation utilities for Air Quality ML.

This module contains evaluation functions with visualizations and detailed analysis
used specifically by the evaluation scripts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .config import WHO_THRESHOLDS, CITY_COL, TARGET_COL
from .utils import setup_plotting, save_plot, print_results_summary


def evaluate_model_detailed(model, X, y, groups=None, model_name="Model"):
    """
    Perform detailed evaluation with visualizations.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target variable
        groups: Grouping variable
        model_name: Name for plots and reports
        
    Returns:
        Dictionary with detailed evaluation results
    """
    print(f"Detailed evaluation of {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    from pipeline.evaluator import Evaluator
    evaluator = Evaluator()
    metrics = evaluator.calculate_metrics(y, y_pred)
    print_results_summary(metrics)
    
    # Create visualizations
    _create_evaluation_plots(y, y_pred, groups, model_name)
    
    # Air quality specific analysis
    air_quality_metrics = _analyze_air_quality_performance(y, y_pred)
    
    return {
        'metrics': metrics,
        'air_quality_metrics': air_quality_metrics,
        'predictions': y_pred
    }


def _create_evaluation_plots(y_true, y_pred, groups, model_name):
    """Create comprehensive evaluation plots."""
    
    setup_plotting()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=30)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('Actual PM2.5 (µg/m³)')
    ax1.set_ylabel('Predicted PM2.5 (µg/m³)')
    ax1.set_title('Actual vs Predicted Values')
    ax1.grid(True, alpha=0.3)
    
    # Add R² to plot
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Residuals plot
    ax2 = axes[0, 1]
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.6, s=30)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted PM2.5 (µg/m³)')
    ax2.set_ylabel('Residuals (µg/m³)')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution comparison
    ax3 = axes[1, 0]
    ax3.hist(y_true, bins=30, alpha=0.7, label='Actual', density=True)
    ax3.hist(y_pred, bins=30, alpha=0.7, label='Predicted', density=True)
    ax3.set_xlabel('PM2.5 (µg/m³)')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance by group (if available)
    ax4 = axes[1, 1]
    if groups is not None:
        _plot_performance_by_group(ax4, y_true, y_pred, groups)
    else:
        ax4.text(0.5, 0.5, 'No grouping variable provided',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Performance by Group')
    
    plt.tight_layout()
    save_plot(f'{model_name.lower()}_evaluation.png')
    plt.show()


def _plot_performance_by_group(ax, y_true, y_pred, groups):
    """Plot RMSE performance by group (e.g., city)."""
    
    group_rmse = []
    group_names = []
    
    for group in groups.unique():
        mask = groups == group
        if mask.sum() > 0:
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            rmse = np.sqrt(mean_squared_error(group_y_true, group_y_pred))
            group_rmse.append(rmse)
            group_names.append(str(group))
    
    bars = ax.bar(range(len(group_names)), group_rmse)
    ax.set_xlabel('Group')
    ax.set_ylabel('RMSE (µg/m³)')
    ax.set_title('RMSE by Group')
    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rmse in zip(bars, group_rmse):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{rmse:.2f}', ha='center', va='bottom')


def _analyze_air_quality_performance(y_true, y_pred):
    """Analyze performance according to WHO air quality guidelines."""
    
    def categorize_pm25(values):
        """Categorize PM2.5 values according to WHO guidelines."""
        categories = np.zeros_like(values, dtype=int)
        categories[values <= WHO_THRESHOLDS['good']] = 0
        categories[(values > WHO_THRESHOLDS['good']) & 
                  (values <= WHO_THRESHOLDS['moderate'])] = 1
        categories[(values > WHO_THRESHOLDS['moderate']) & 
                  (values <= WHO_THRESHOLDS['unhealthy'])] = 2
        categories[values > WHO_THRESHOLDS['unhealthy']] = 3
        return categories
    
    # Categorize predictions
    true_categories = categorize_pm25(y_true)
    pred_categories = categorize_pm25(y_pred)
    
    # Calculate category accuracy
    category_accuracy = np.mean(true_categories == pred_categories)
    
    # Performance in different ranges
    range_performance = {}
    ranges = [
        ("Good (0-12)", 0, WHO_THRESHOLDS['good']),
        ("Moderate (12-35)", WHO_THRESHOLDS['good'], WHO_THRESHOLDS['moderate']),
        ("Unhealthy (35-55)", WHO_THRESHOLDS['moderate'], WHO_THRESHOLDS['unhealthy']),
        ("Very Unhealthy (55+)", WHO_THRESHOLDS['unhealthy'], float('inf'))
    ]
    
    for range_name, min_val, max_val in ranges:
        if max_val == float('inf'):
            mask = y_true >= min_val
        else:
            mask = (y_true >= min_val) & (y_true < max_val)
        
        if mask.sum() > 0:
            range_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            range_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            range_bias = np.mean(y_pred[mask] - y_true[mask])
            
            range_performance[range_name] = {
                'rmse': range_rmse,
                'mae': range_mae,
                'bias': range_bias,
                'n_samples': mask.sum()
            }
    
    return {
        'category_accuracy': category_accuracy,
        'range_performance': range_performance
    }
