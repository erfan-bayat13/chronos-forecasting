import numpy as np
import torch
from typing import Union, Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path

def to_numpy(tensor_or_array: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy array."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return tensor_or_array

def calculate_prediction_intervals(
    predictions: Union[torch.Tensor, np.ndarray],
    quantile_levels: List[float] = [0.1, 0.5, 0.9]
) -> Dict[str, np.ndarray]:
    """
    Calculate prediction intervals from a set of predictions.
    
    Args:
        predictions: Shape (n_perturbations, batch_size, n_samples, sequence_length) or
                    Shape (batch_size, n_samples, sequence_length)
        quantile_levels: List of quantiles to compute
        
    Returns:
        Dictionary containing lower bounds, upper bounds, and median for each interval
    """
    predictions = to_numpy(predictions)
    
    # Reshape if we have perturbations
    if predictions.ndim == 4:
        n_pert, batch_size, n_samples, seq_len = predictions.shape
        predictions = predictions.reshape(-1, seq_len)  # Combine perturbations and samples
    elif predictions.ndim == 3:
        batch_size, n_samples, seq_len = predictions.shape
        predictions = predictions.reshape(-1, seq_len)
    
    # Calculate quantiles
    quantiles = np.quantile(predictions, quantile_levels, axis=0)
    
    # Create intervals dictionary
    intervals = {
        'lower': quantiles[0],  # 10th percentile
        'median': quantiles[1],  # 50th percentile
        'upper': quantiles[2],   # 90th percentile
    }
    
    return intervals

def evaluate_coverage_and_width(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    quantile_levels: List[float] = [0.1, 0.5, 0.9],
    scale: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Evaluate prediction intervals coverage and width.
    
    Args:
        predictions: Predicted values
        targets: True values
        quantile_levels: Quantile levels for intervals
        scale: Optional scale factor for normalizing interval widths
        
    Returns:
        Dictionary containing coverage rate and interval width metrics
    """
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    if scale is not None:
        scale = to_numpy(scale)
    
    # Calculate prediction intervals
    intervals = calculate_prediction_intervals(predictions, quantile_levels)
    
    # Calculate coverage
    in_interval = np.logical_and(
        targets >= intervals['lower'],
        targets <= intervals['upper']
    )
    coverage_rate = np.mean(in_interval)
    
    # Calculate interval widths
    interval_width = intervals['upper'] - intervals['lower']
    mean_width = np.mean(interval_width)
    
    # Calculate normalized width if scale is provided
    if scale is not None:
        normalized_width = mean_width / scale
    else:
        normalized_width = mean_width
    
    # Calculate additional metrics
    median_absolute_error = np.median(np.abs(intervals['median'] - targets))
    
    # Calculate consistency across perturbations if available
    if predictions.ndim == 4:
        median_predictions = np.median(predictions, axis=2)  # median across samples
        consistency_score = np.mean(np.std(median_predictions, axis=0))
    else:
        consistency_score = None
    
    metrics = {
        'coverage_rate': float(coverage_rate),
        'nominal_coverage': float(quantile_levels[2] - quantile_levels[0]),  # e.g., 0.8 for [0.1, 0.9]
        'coverage_error': float(abs(coverage_rate - (quantile_levels[2] - quantile_levels[0]))),
        'mean_interval_width': float(mean_width),
        'normalized_interval_width': float(normalized_width),
        'median_absolute_error': float(median_absolute_error),
    }
    
    if consistency_score is not None:
        metrics['consistency_score'] = float(consistency_score)
    
    return metrics

def plot_interval_evaluation(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    context: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Path] = None,
    title: Optional[str] = None
) -> None:
    """
    Create visualization of prediction intervals and coverage.
    
    Args:
        predictions: Predicted values
        targets: True values
        context: Optional historical context data
        save_path: Path to save the plot
        title: Optional plot title
    """
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    if context is not None:
        context = to_numpy(context)
    
    intervals = calculate_prediction_intervals(predictions)
    
    plt.figure(figsize=(12, 6))
    
    # Plot historical context if provided
    if context is not None:
        plt.plot(range(len(context)), context, 
                color='gray', alpha=0.5, label='Historical')
        offset = len(context)
    else:
        offset = 0
    
    # Plot prediction intervals
    x = range(offset, offset + len(targets))
    plt.fill_between(x, intervals['lower'], intervals['upper'],
                    alpha=0.3, label='80% PI')
    plt.plot(x, intervals['median'], 'b-', label='Median forecast')
    plt.plot(x, targets, 'r--', label='Actual')
    
    # Add coverage rate to title
    in_interval = np.logical_and(
        targets >= intervals['lower'],
        targets <= intervals['upper']
    )
    coverage_rate = np.mean(in_interval)
    
    if title:
        plt.title(f"{title}\nEmpirical coverage rate: {coverage_rate:.1%}")
    else:
        plt.title(f"Prediction Intervals\nEmpirical coverage rate: {coverage_rate:.1%}")
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def evaluate_calibration(
    original_preds: Union[torch.Tensor, np.ndarray],
    calibrated_preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    context: Optional[Union[torch.Tensor, np.ndarray]] = None,
    output_dir: Union[str, Path] = None,
    config_name: str = "default",
    noise_config: Optional[dict] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate both original and calibrated predictions.
    
    Args:
        original_preds: Original model predictions
        calibrated_preds: Predictions after C3 calibration
        targets: True values
        context: Optional historical context
        output_dir: Directory for saving plots
        config_name: Name of current configuration for saving files
        
    Returns:
        Dictionary containing metrics for both original and calibrated predictions
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate scale for normalized metrics
    if context is not None:
        scale = np.nanstd(to_numpy(context))
    else:
        scale = np.nanstd(to_numpy(targets))
    
    # Evaluate original predictions
    original_metrics = evaluate_coverage_and_width(
        original_preds, targets, scale=scale
    )
    
    # Evaluate calibrated predictions
    calibrated_metrics = evaluate_coverage_and_width(
        calibrated_preds, targets, scale=scale
    )
    
    # Create plots if output directory is provided
    if output_dir:
        plot_interval_evaluation(
            original_preds, targets, context,
            save_path=output_dir / f"{config_name}_original.png",
            title="Original Predictions"
        )
        
        plot_interval_evaluation(
            calibrated_preds, targets, context,
            save_path=output_dir / f"{config_name}_calibrated.png",
            title="Calibrated Predictions"
        )
    
    # Combine all metrics into a single flat dictionary
    combined_metrics = {
        'noise_distribution': noise_config['dist'] if noise_config else 'N/A',
        'noise_type': noise_config['type'] if noise_config else 'N/A',
        'noise_strength': noise_config['strength'] if noise_config else 0.0,
        # Original metrics
        'original_coverage_rate': original_metrics['coverage_rate'],
        'original_coverage_error': original_metrics['coverage_error'],
        'original_interval_width': original_metrics['mean_interval_width'],
        'original_normalized_width': original_metrics['normalized_interval_width'],
        'original_median_error': original_metrics['median_absolute_error'],
        # Calibrated metrics
        'calibrated_coverage_rate': calibrated_metrics['coverage_rate'],
        'calibrated_coverage_error': calibrated_metrics['coverage_error'],
        'calibrated_interval_width': calibrated_metrics['mean_interval_width'],
        'calibrated_normalized_width': calibrated_metrics['normalized_interval_width'],
        'calibrated_median_error': calibrated_metrics['median_absolute_error'],
        # Improvements
        'coverage_improvement': calibrated_metrics['coverage_rate'] - original_metrics['coverage_rate'],
        'width_reduction': 1 - (calibrated_metrics['normalized_interval_width'] / original_metrics['normalized_interval_width']),
        'error_reduction': 1 - (calibrated_metrics['median_absolute_error'] / original_metrics['median_absolute_error'])
    }
    
    return combined_metrics