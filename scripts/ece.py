import numpy as np
import torch
from typing import Union, Tuple, Dict

def calculate_ece(
    confidences: Union[np.ndarray, torch.Tensor],
    accuracies: Union[np.ndarray, torch.Tensor],
    num_bins: int = 15,
    return_bins: bool = False
) -> Union[float, Tuple[float, Dict]]:
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        confidences: Model's predicted probabilities/confidences
        accuracies: Binary array indicating correct (1) or incorrect (0) predictions
        num_bins: Number of bins to use for confidence buckets
        return_bins: If True, return additional bin statistics
        
    Returns:
        ece: Expected Calibration Error score
        bin_stats: (Optional) Dictionary containing bin statistics
    """
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.detach().cpu().numpy()
    if isinstance(accuracies, torch.Tensor):
        accuracies = accuracies.detach().cpu().numpy()
        
    confidences = np.asarray(confidences).flatten()
    accuracies = np.asarray(accuracies).flatten()
    
    # Create confidence bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_stats = {
        'accuracies': [],
        'confidences': [],
        'sizes': [],
        'bin_edges': list(zip(bin_lowers, bin_uppers))
    }
    
    ece = 0.0
    total_samples = len(confidences)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_accuracy - bin_confidence)
            
            bin_stats['accuracies'].append(float(bin_accuracy))
            bin_stats['confidences'].append(float(bin_confidence))
            bin_stats['sizes'].append(int(bin_size))
        else:
            bin_stats['accuracies'].append(0.0)
            bin_stats['confidences'].append(0.0)
            bin_stats['sizes'].append(0)
    
    if return_bins:
        return float(ece), bin_stats
    return float(ece)

def compute_forecast_ece(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    tolerance: float = 0.1,
    num_bins: int = 15,
    return_bins: bool = False
) -> Union[float, Tuple[float, Dict]]:
    """
    Compute ECE for forecasting predictions.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Ensure predictions have sample dimension
    if predictions.ndim == 3:  # (batch, n_samples, seq_len)
        pass
    elif predictions.ndim == 4:  # (n_perturbations, batch, n_samples, seq_len)
        predictions = np.mean(predictions, axis=0)
    else:
        raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
    
    # Calculate spread-based confidence more robustly
    prediction_std = np.std(predictions, axis=1)  # (batch, seq_len)
    prediction_iqr = np.percentile(predictions, 75, axis=1) - np.percentile(predictions, 25, axis=1)
    
    # Combine std and IQR for more robust spread estimate
    spread = (prediction_std + prediction_iqr) / 2
    max_spread = np.maximum(np.max(spread), 1e-8)
    confidences = np.clip(1 - (spread / max_spread), 0, 1)
    
    # Calculate accuracy using both absolute and relative errors
    prediction_means = np.mean(predictions, axis=1)  # (batch, seq_len)
    
    # Relative error with better scaling
    scale = np.maximum(np.abs(targets), np.abs(prediction_means))
    scale = np.maximum(scale, np.std(targets) / 10)  # Avoid too small scales
    relative_errors = np.abs(prediction_means - targets) / (scale + 1e-8)
    
    # Absolute error normalized by target std
    abs_errors = np.abs(prediction_means - targets) / (np.std(targets) + 1e-8)
    
    # Combine both error metrics
    combined_errors = (relative_errors + abs_errors) / 2
    accuracies = (combined_errors <= tolerance).astype(float)
    
    return calculate_ece(
        confidences.flatten(),
        accuracies.flatten(),
        num_bins=num_bins,
        return_bins=return_bins
    )

def analyze_calibration(
    original_preds: Union[torch.Tensor, np.ndarray],
    calibrated_preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    tolerance: float = 0.1,
    num_bins: int = 15
) -> Dict:
    """
    Analyze calibration comparing original and calibrated predictions.
    
    Args:
        original_preds: Original model predictions
        calibrated_preds: Predictions after calibration
        targets: True values
        tolerance: Relative tolerance for accuracy
        num_bins: Number of bins for ECE calculation
        
    Returns:
        Dictionary containing calibration metrics
    """
    # Calculate ECE for original predictions
    original_ece, original_bins = compute_forecast_ece(
        original_preds, targets, tolerance, num_bins, return_bins=True
    )
    
    # Calculate ECE for calibrated predictions
    calibrated_ece, calibrated_bins = compute_forecast_ece(
        calibrated_preds, targets, tolerance, num_bins, return_bins=True
    )
    
    # Compute improvement metrics
    ece_improvement = original_ece - calibrated_ece
    ece_reduction_percent = (ece_improvement / original_ece * 100) if original_ece > 0 else 0
    
    return {
        'original_ece': original_ece,
        'calibrated_ece': calibrated_ece,
        'ece_improvement': ece_improvement,
        'ece_reduction_percent': ece_reduction_percent,
        'original_bin_stats': original_bins,
        'calibrated_bin_stats': calibrated_bins
    }