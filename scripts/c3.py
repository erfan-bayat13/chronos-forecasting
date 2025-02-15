import argparse
import pandas as pd
import torch
import numpy as np
from chronos import BaseChronosPipeline
from torch.nn.functional import softmax
import warnings
from pathlib import Path
import json
from calibration_metrics import evaluate_calibration
from ece import analyze_calibration
from ece import compute_forecast_ece
import matplotlib.pyplot as plt
from datasets import load_dataset
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def aggregate_predictions(all_predictions, all_logits=None, perturb_type="Logits"):
    """Aggregate predictions from multiple perturbations."""
    # Stack predictions properly preserving the sample dimension
    stacked_predictions = np.stack(all_predictions, axis=0)  # (n_perturbations, batch, n_samples, seq_len)
    
    # Average across perturbations while keeping sample dimension
    mean_predictions = np.mean(stacked_predictions, axis=0)  # (batch, n_samples, seq_len)
    
    if all_logits is not None and perturb_type == "Logits":
        stacked_logits = torch.stack(all_logits, dim=0)
        # Normalize logits
        max_val = stacked_logits.max()
        min_val = stacked_logits.min()
        stacked_logits = (stacked_logits - min_val) / (max_val - min_val + 1e-6)
        
        mean_logits = torch.mean(stacked_logits, dim=0)
        temperature = 1.0
        mean_logits = mean_logits / temperature
        
        probs = softmax(mean_logits, dim=-1)
        
        return mean_predictions, probs, mean_logits
    
    return mean_predictions, None, None  # If `perturb_type="Input"`, return `None` for probs and logits


def to_numpy(tensor_or_array):
    """Convert tensor to numpy array if it's a tensor."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return tensor_or_array

def plot_prediction_comparison(context, targets, original_forecast, calibrated_predictions, noise_config, perturb_type, save_path=None):
    """
    Plot original vs calibrated predictions with confidence intervals.
    """
    plt.figure(figsize=(12, 6))
    
    # Convert all inputs to numpy and ensure correct dimensions
    context = to_numpy(context).squeeze()
    targets = to_numpy(targets).squeeze()
    original_forecast = to_numpy(original_forecast)
    calibrated_predictions = to_numpy(calibrated_predictions)
    
    # Print shapes for debugging
    print("Shapes:")
    print(f"Context: {context.shape}")
    print(f"Targets: {targets.shape}")
    print(f"Original forecast: {original_forecast.shape}")
    print(f"Calibrated predictions: {calibrated_predictions.shape}")
    
    # Create x-axis points
    x_context = np.arange(len(context))
    x_forecast = np.arange(len(context), len(context) + len(targets))
    
    # Plot historical data
    plt.plot(x_context, context, color='gray', alpha=0.5, label='Historical')
    plt.plot(x_forecast, targets, color='black', linestyle='--', label='Actual')
    
    # Original forecast - take the first batch if multiple batches exist
    if original_forecast.ndim >= 3:
        original_mean = np.mean(original_forecast[0], axis=0)
        original_std = np.std(original_forecast[0], axis=0)
    else:
        original_mean = np.mean(original_forecast, axis=0)
        original_std = np.std(original_forecast, axis=0)
    
    plt.plot(x_forecast, original_mean, color='blue', label='Original Forecast')
    plt.fill_between(x_forecast, 
                    original_mean - 2*original_std,
                    original_mean + 2*original_std,
                    color='blue', alpha=0.2, label='Original 95% CI')
    
    # Calibrated forecast
    if calibrated_predictions.ndim == 4:  # (n_perturbations, batch, samples, seq_len)
        calibrated_mean = np.mean(calibrated_predictions, axis=(0, 1))[0]
        calibrated_std = np.std(calibrated_predictions, axis=(0, 1))[0]
    elif calibrated_predictions.ndim == 3:  # (batch, samples, seq_len)
        calibrated_mean = np.mean(calibrated_predictions, axis=1)[0]
        calibrated_std = np.std(calibrated_predictions, axis=1)[0]
    else:
        raise ValueError(f"Unexpected shape for calibrated predictions: {calibrated_predictions.shape}")
    
    plt.plot(x_forecast, calibrated_mean, color='red', label=f'Calibrated Forecast ({perturb_type})')
    plt.fill_between(x_forecast,
                    calibrated_mean - 2*calibrated_std,
                    calibrated_mean + 2*calibrated_std,
                    color='red', alpha=0.2, label=f'Calibrated 95% CI ({perturb_type})')
    
    plt.title(f'Prediction Comparison\n{noise_config["dist"].capitalize()} {noise_config["type"]} '
              f'(strength={noise_config["strength"]}, n={noise_config["num_perturbations"]})\n'
              f'Perturbation: {perturb_type}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
      # Ensure the directory exists
      save_path.parent.mkdir(parents=True, exist_ok=True)
      plt.savefig(save_path, bbox_inches='tight')
      print(f"Plot saved at: {save_path}")  # Debugging output
    
    plt.close()


def plot_perturbation_effects(final_ece_results, metrics_to_plot=None, save_path=None):
    """
    Plot how different metrics change with number of perturbations, using ECE aggregated across 5 time series.

    Args:
        final_ece_results: DataFrame containing final aggregated ECE results for all 5 series.
        metrics_to_plot: List of metrics to plot (default: ECE Reduction, Coverage Improvement, Variance Ratio).
        save_path: Optional path to save the plot.
    """
    if metrics_to_plot is None:
        metrics_to_plot = [
            ('ece_reduction_percent', 'ECE Reduction (%)'),
            ('coverage_improvement', 'Coverage Improvement'),
            ('model_variance_ratio', 'Variance Ratio')
        ]
    
    n_metrics = len(metrics_to_plot)
    plt.figure(figsize=(15, 4*n_metrics))
    
    for idx, (metric, title) in enumerate(metrics_to_plot, 1):
        plt.subplot(n_metrics, 1, idx)
        
        # Group by noise configurations
        configs = final_ece_results.groupby(['dist', 'type', 'strength'])
        
        for name, group in configs:
            dist, type_, strength = name
            label = f'{dist}-{type_}-{strength}'
            
            # Sort by number of perturbations and plot mean ECE with standard deviation
            sorted_group = group.sort_values('num_perturbations')
            plt.plot(sorted_group['num_perturbations'], 
                     sorted_group[f"{metric}_mean"], 
                     'o-', 
                     label=label)

            plt.fill_between(sorted_group['num_perturbations'], 
                             sorted_group[f"{metric}_mean"] - sorted_group[f"{metric}_std"], 
                             sorted_group[f"{metric}_mean"] + sorted_group[f"{metric}_std"], 
                             alpha=0.2)

        plt.title(f'{title} vs Number of Perturbations')
        plt.xlabel('Number of Perturbations')
        plt.ylabel(title)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Perturbation effects plot saved at: {save_path}")  # Debugging output
    plt.close()


def process_results_for_plotting(results):
    """Convert results dictionary to DataFrame with parsed configuration."""
    df = pd.DataFrame(results).T.reset_index()
    df = df.rename(columns={'index': 'config_key'})
    
    # Parse configuration from config_key
    df[['dist', 'type', 'strength', 'num_perturbations']] = df['config_key'].str.split('_', expand=True)
    df['strength'] = df['strength'].astype(float)
    df['num_perturbations'] = df['num_perturbations'].astype(int)
    
    return df


def run_calibration(args):
    """Run consistency calibration with given parameters across multiple time series."""
    warnings.filterwarnings("ignore")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Load pipeline and configure device
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model_name,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if args.use_bfloat16 else torch.float32,
    )

    # Load dataset
    ds = load_dataset("autogluon/chronos_datasets", "nn5")
    
    # Get all time series from the dataset
    all_time_series = ds["train"][args.target_column]
    
    # Randomly select 5 time series
    num_series = 5
    selected_indices = random.sample(range(len(all_time_series)), num_series)
    selected_time_series = [all_time_series[idx] for idx in selected_indices]
    
    print(f"Selected time series indices: {selected_indices}")

    # Parse parameters for different configurations
    perturbation_counts = [int(x) for x in args.num_perturbations.split(',')]
    noise_distributions = args.noise_dist.split(',')
    noise_types = args.noise_type.split(',')
    noise_strengths = [float(s) for s in args.noise_strength.split(',')]
    
    # Calculate total number of configurations
    total_configs = len(perturbation_counts) * len(noise_distributions) * len(noise_types) * len(noise_strengths)
    
    # Store results for each time series and configuration
    all_original_forecasts = []  # Will store original forecasts for each series
    all_calibrated_predictions = {i: [] for i in range(total_configs)}  # Dictionary to store calibrated predictions by config index
    all_test_targets = []
    all_configurations = []

    config_idx = 0  # Track configuration index
    
    # First, define all configurations
    for dist in noise_distributions:
        for noise_type in noise_types:
            for strength in noise_strengths:
                for num_pert in perturbation_counts:
                    noise_config = {
                        "dist": dist,
                        "type": noise_type,
                        "strength": strength,
                        "num_perturbations": num_pert
                    }
                    all_configurations.append(noise_config)

    for series_idx, full_data in enumerate(selected_time_series):
        print(f"\nProcessing time series {series_idx + 1}/{num_series} (dataset index: {selected_indices[series_idx]})")

        # Ensure data is a numpy array
        if isinstance(full_data, list):
            full_data = np.array(full_data)

        # Split data into context and test
        context_length = len(full_data) - args.prediction_length
        context = torch.tensor(full_data[:context_length], dtype=torch.float32)
        test_targets = torch.tensor(full_data[context_length:context_length + args.prediction_length], dtype=torch.float32)

        print(f"Using {len(context)} points for context and {len(test_targets)} points for evaluation")

        # Get original forecast without perturbation
        pipeline.model.config.use_cc = False
        with torch.no_grad():
            original_forecast = pipeline.predict(
                context=context,
                prediction_length=args.prediction_length,
                num_samples=args.num_samples,
                return_logits=False
            )
        original_forecast = original_forecast.type(torch.float32)

        # Ensure test_targets shape is consistent
        test_targets = test_targets.unsqueeze(0).unsqueeze(0)  # (1, 1, prediction_length)
        test_targets = test_targets.repeat(1, args.num_samples, 1)  # (1, num_samples, prediction_length)

        # Store for final ECE calculation
        all_original_forecasts.append(original_forecast.numpy())  
        all_test_targets.append(test_targets.numpy())  

        # Reset configuration index for each series
        config_idx = 0
        
        # Process each configuration
        for dist in noise_distributions:
            for noise_type in noise_types:
                for strength in noise_strengths:
                    for num_pert in perturbation_counts:
                        noise_config = {
                            "dist": dist,
                            "type": noise_type,
                            "strength": strength,
                            "num_perturbations": num_pert
                        }
                        print(f"\nTesting configuration: {noise_config}")

                        # Generate multiple perturbed predictions
                        all_predictions = []
                        with torch.no_grad():
                            for _ in range(num_pert):
                                prediction = pipeline.predict(
                                    context=context,
                                    prediction_length=args.prediction_length,
                                    num_samples=args.num_samples
                                )
                                all_predictions.append(prediction.cpu().numpy())

                        # Aggregate predictions
                        calibrated_predictions, _, _ = aggregate_predictions(all_predictions)

                        # Store calibrated predictions for this configuration
                        all_calibrated_predictions[config_idx].append(calibrated_predictions)
                        
                        # Increment configuration index
                        config_idx += 1

    # Convert lists to numpy arrays
    all_original_forecasts = np.array(all_original_forecasts)  # (5, 1, num_samples, prediction_length)
    all_test_targets = np.array(all_test_targets)  # (5, 1, num_samples, prediction_length)

    print(f"Original forecasts shape: {all_original_forecasts.shape}")
    print(f"Test targets shape: {all_test_targets.shape}")

    # Process each configuration separately for ECE calculation
    final_results = []
    
    for i, config in enumerate(all_configurations):
        # Stack calibrated predictions for this configuration
        calibrated_preds_for_config = np.array(all_calibrated_predictions[i])  # (5, 1, num_samples, prediction_length)
        
        print(f"Configuration {i+1}/{len(all_configurations)}: {config}")
        print(f"Calibrated predictions shape for this config: {calibrated_preds_for_config.shape}")
        
        # Calculate ECE for this configuration
        orig_ece, _ = compute_forecast_ece(
            all_original_forecasts, all_test_targets, 
            tolerance=args.ece_tolerance, num_bins=args.ece_bins, 
            return_bins=True
        )
        
        cal_ece, _ = compute_forecast_ece(
            calibrated_preds_for_config, all_test_targets,
            tolerance=args.ece_tolerance, num_bins=args.ece_bins,
            return_bins=True
        )
        
        # Compute coverage improvement based on model variance
        original_variance = np.var(all_original_forecasts)
        calibrated_variance = np.var(calibrated_preds_for_config)
        variance_ratio = calibrated_variance / original_variance if original_variance > 0 else 1.0
        
        ece_improvement = orig_ece - cal_ece
        ece_reduction_percent = (ece_improvement / orig_ece * 100) if orig_ece > 0 else 0
        
        # Store results
        final_results.append({
            "dist": config["dist"],
            "type": config["type"], 
            "strength": config["strength"],
            "num_perturbations": config["num_perturbations"],
            "original_ece": orig_ece,
            "calibrated_ece": cal_ece,
            "ece_reduction_percent": ece_reduction_percent,
            "ece_reduction_percent_mean": ece_reduction_percent,  # For compatibility with plotting
            "ece_reduction_percent_std": 0,                      # For compatibility with plotting
            "coverage_improvement": variance_ratio,
            "coverage_improvement_mean": variance_ratio,         # For compatibility with plotting
            "coverage_improvement_std": 0,                       # For compatibility with plotting
            "model_variance_ratio": variance_ratio,
            "model_variance_ratio_mean": variance_ratio,         # For compatibility with plotting
            "model_variance_ratio_std": 0                        # For compatibility with plotting
        })
    
    # Convert to DataFrame
    final_ece_results_df = pd.DataFrame(final_results)

    # Save results to CSV
    final_ece_results_df.to_csv(output_dir / "final_ece_results.csv", index=False)
    
    # Create plot directory
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Save selected indices
    with open(output_dir / "selected_indices.json", "w") as f:
        json.dump({"selected_indices": selected_indices}, f, indent=4)
    
    # Plot perturbation effects
    plot_perturbation_effects(
        final_ece_results_df,
        save_path=plot_dir / "perturbation_effects.png"
    )

    print("\nCalibration complete. Results saved to:", output_dir)

def main():
    parser = argparse.ArgumentParser(description="Consistency Calibration for Chronos (C3)")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="amazon/chronos-t5-small",
                      help="Name or path of the Chronos model")
    parser.add_argument("--use_bfloat16", action="store_true",
                      help="Use bfloat16 precision")
    
    # Data parameters
    parser.add_argument("--target_column", type=str, default="target",
                      help="Name of target column in dataset")
    
    # Prediction parameters
    parser.add_argument("--prediction_length", type=int, default=12,
                      help="Number of steps to predict")
    parser.add_argument("--num_samples", type=int, default=20,
                      help="Number of samples per prediction")
    parser.add_argument("--num_perturbations", type=str, default="8,16,32,64",
                      help="Comma-separated list of perturbation counts to test")
    
    # Noise parameters
    parser.add_argument("--perturb_type", type=str, default="logits",
                      help="flag for applying perturbation on input")

    parser.add_argument("--noise_dist", type=str, default="gaussian,uniform",
                      help="Comma-separated list of noise distributions to test")
    parser.add_argument("--noise_type", type=str, default="multiplicative,additive",
                      help="Comma-separated list of noise types to test")
    parser.add_argument("--noise_strength", type=str, default="0.01,0.05,0.1",
                      help="Comma-separated list of noise strengths to test")
    
    # ECE parameters
    parser.add_argument("--ece_bins", type=int, default=15,
                      help="Number of bins for ECE calculation")
    parser.add_argument("--ece_tolerance", type=float, default=0.1,
                      help="Tolerance for considering a prediction accurate in ECE calculation")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./c3_output",
                      help="Directory for output files")
    
    args = parser.parse_args()
    run_calibration(args)

if __name__ == "__main__":
    main()
