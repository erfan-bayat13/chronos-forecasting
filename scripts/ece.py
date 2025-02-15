import numpy as np
import torch
from typing import Union, Tuple, Dict, List
import pandas as pd 

def calculate_ece(
    confidences: Union[np.ndarray, torch.Tensor],
    accuracies: Union[np.ndarray, torch.Tensor],
    num_bins: int = 15,
    return_bins: bool = False
) -> Union[float, Tuple[float, Dict]]:
    """
    Calculate Expected Calibration Error (ECE) for a given set of predictions.

    Args:
        confidences: Model's predicted probabilities/confidences.
        accuracies: Binary array indicating correct (1) or incorrect (0) predictions.
        num_bins: Number of bins to use for confidence buckets.
        return_bins: If True, return additional bin statistics.
        
    Returns:
        ece: Expected Calibration Error score.
        bin_stats: (Optional) Dictionary containing bin statistics.
    """
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.detach().cpu().numpy()
    if isinstance(accuracies, torch.Tensor):
        accuracies = accuracies.detach().cpu().numpy()
        
    # Ensure arrays are flattened and have the same shape
    confidences = np.asarray(confidences).flatten()
    accuracies = np.asarray(accuracies).flatten()
    
    # Double check they have the same length
    assert len(confidences) == len(accuracies), f"Length mismatch: confidences ({len(confidences)}) != accuracies ({len(accuracies)})"
    
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
    
    # Print shapes for debugging
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Ensure targets has the right shape to match predictions
    if targets.shape != predictions.shape:
        if targets.shape[-1] == predictions.shape[-1]:  # If only last dimension matches
            # Try to broadcast targets to match predictions shape
            try:
                targets = np.broadcast_to(targets, predictions.shape)
                print(f"Broadcasted targets to shape: {targets.shape}")
            except ValueError:
                print("Could not broadcast targets to match predictions shape")
                raise
    
    # Calculate spread-based confidence more robustly
    prediction_std = np.std(predictions, axis=2)  # Calculate std over samples dimension
    prediction_iqr = np.percentile(predictions, 75, axis=2) - np.percentile(predictions, 25, axis=2)
    
    # Combine std and IQR for more robust spread estimate
    spread = (prediction_std + prediction_iqr) / 2
    max_spread = np.maximum(np.max(spread), 1e-8)
    confidences = np.clip(1 - (spread / max_spread), 0, 1)
    
    # Calculate accuracy using both absolute and relative errors
    prediction_means = np.mean(predictions, axis=2)  # Mean over samples dimension
    
    # Ensure targets and prediction_means have compatible shapes for comparison
    if targets.ndim != prediction_means.ndim:
        targets_for_comparison = np.mean(targets, axis=2)  # Average over samples dimension if needed
    else:
        targets_for_comparison = targets
    
    # Relative error with better scaling
    scale = np.maximum(np.abs(targets_for_comparison), np.abs(prediction_means))
    scale = np.maximum(scale, np.std(targets_for_comparison) / 10)  # Avoid too small scales
    relative_errors = np.abs(prediction_means - targets_for_comparison) / (scale + 1e-8)
    
    # Absolute error normalized by target std
    target_std = np.std(targets_for_comparison) + 1e-8
    abs_errors = np.abs(prediction_means - targets_for_comparison) / target_std
    
    # Combine both error metrics
    combined_errors = (relative_errors + abs_errors) / 2
    accuracies = (combined_errors <= tolerance).astype(float)
    
    # Make sure confidences and accuracies have the same shape
    print(f"Confidences shape: {confidences.shape}")
    print(f"Accuracies shape: {accuracies.shape}")
    
    # Flatten both arrays for calculate_ece
    confidences_flat = confidences.flatten()
    accuracies_flat = accuracies.flatten()
    
    return calculate_ece(
        confidences_flat,
        accuracies_flat,
        num_bins=num_bins,
        return_bins=return_bins
    )



def analyze_calibration(
    original_preds: Union[torch.Tensor, np.ndarray],
    calibrated_preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    tolerance: float = 0.1,
    num_bins: int = 15,
    config_list: List[Dict] = None  # ✅ Ensure perturbation configurations are passed
) -> pd.DataFrame:
    """
    Analyze calibration by calculating ECE **at the end for each perturbation configuration** instead of per series.

    Args:
        original_preds: Array of shape (num_series, batch, seq_len)
        calibrated_preds: Array of shape (num_series, batch, seq_len)
        targets: Array of shape (num_series, batch, seq_len)
        tolerance: Relative tolerance for accuracy
        num_bins: Number of bins for ECE calculation
        config_list: List of configuration dictionaries (one per perturbation config)
        
    Returns:
        Pandas DataFrame with ECE computed per perturbation setting, not per series.
    """

    # Ensure all arrays have the same shape
    assert original_preds.shape == calibrated_preds.shape == targets.shape, "Shape mismatch in predictions and targets!"

    results = []
    
    # Aggregate across **all time series before computing ECE**
    aggregated_original_preds = np.concatenate(original_preds, axis=0)  # (total_batch, seq_len)
    aggregated_calibrated_preds = np.concatenate(calibrated_preds, axis=0)  # (total_batch, seq_len)
    aggregated_targets = np.concatenate(targets, axis=0)  # (total_batch, seq_len)

    for i, noise_config in enumerate(config_list):
        # Compute **ECE on aggregated predictions**
        orig_ece, _ = compute_forecast_ece(
            aggregated_original_preds, aggregated_targets, tolerance, num_bins, return_bins=True
        )
        cal_ece, _ = compute_forecast_ece(
            aggregated_calibrated_preds, aggregated_targets, tolerance, num_bins, return_bins=True
        )

        # Compute **coverage improvement** based on model variance
        original_variance = np.var(aggregated_original_preds)
        calibrated_variance = np.var(aggregated_calibrated_preds)
        variance_ratio = calibrated_variance / original_variance if original_variance > 0 else 1.0

        ece_improvement = orig_ece - cal_ece
        ece_reduction_percent = (ece_improvement / orig_ece * 100) if orig_ece > 0 else 0

        # Store results for this perturbation config
        results.append({
            "dist": noise_config["dist"],
            "type": noise_config["type"],
            "strength": noise_config["strength"],
            "num_perturbations": noise_config["num_perturbations"],
            "original_ece": orig_ece,
            "calibrated_ece": cal_ece,
            "ece_reduction_percent": ece_reduction_percent,
            "coverage_improvement": variance_ratio  # ✅ Added coverage improvement metric
        })

    # Convert to DataFrame
    return pd.DataFrame(results)
