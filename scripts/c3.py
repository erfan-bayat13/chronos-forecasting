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
import matplotlib.pyplot as plt

torch.manual_seed(11)
np.random.seed(11)

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


def plot_perturbation_effects(results_df, metrics_to_plot=None, save_path=None):
    """
    Plot how different metrics change with number of perturbations.
    
    Args:
        results_df: DataFrame containing results for different configurations
        metrics_to_plot: List of metrics to plot (default: ECE and coverage)
        save_path: Optional path to save the plot
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
        
        # Group by configuration
        configs = results_df.groupby(['dist', 'type', 'strength'])
        
        for name, group in configs:
            dist, type_, strength = name
            label = f'{dist}-{type_}-{strength}'
            
            # Sort by number of perturbations and plot
            sorted_group = group.sort_values('num_perturbations')
            plt.plot(sorted_group['num_perturbations'], 
                    sorted_group[metric], 
                    'o-', 
                    label=label)
        
        plt.title(f'{title} vs Number of Perturbations')
        plt.xlabel('Number of Perturbations')
        plt.ylabel(title)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
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
    """Run consistency calibration with given parameters."""
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
    
    # Load and prepare data
    df = pd.read_csv(args.data_path)
    full_data = df[args.target_column].values
    
    # Split data into context and test
    context_length = len(full_data) - args.prediction_length
    
    if context_length <= 0:
        raise ValueError(f"Data length ({len(full_data)}) must be greater than prediction_length ({args.prediction_length})")
    
    context = torch.tensor(full_data[:context_length])
    test_targets = torch.tensor(full_data[context_length:context_length + args.prediction_length])
    
    print(f"Using {len(context)} points for context and {len(test_targets)} points for evaluation")
    
    # Get original forecast without perturbation
    pipeline.model.config.use_cc = False
    with torch.no_grad():
        original_forecast, original_logits = pipeline.predict(
            context=context,
            prediction_length=args.prediction_length,
            num_samples=args.num_samples,
            return_logits=True
        )
    original_forecast = original_forecast.type(torch.float32)
    
    # Parse parameters for different configurations
    perturbation_counts = [int(x) for x in args.num_perturbations.split(',')]
    noise_configs = []
    for dist in args.noise_dist.split(','):
        for noise_type in args.noise_type.split(','):
            for strength in [float(s) for s in args.noise_strength.split(',')]:
                for num_pert in perturbation_counts:
                    noise_configs.append({
                        "dist": dist,
                        "type": noise_type,
                        "strength": strength,
                        "num_perturbations": num_pert
                    })
    
    results = {}
    for noise_config in noise_configs:
        config_key = f"{noise_config['dist']}_{noise_config['type']}_{noise_config['strength']}_{noise_config['num_perturbations']}"
        print(f"\nTesting configuration: {config_key}")
        
        # Enable CC with current configuration
        pipeline.model.config.use_cc = True
        pipeline.model.config.cc_noise_dist = noise_config["dist"]
        pipeline.model.config.cc_noise_type = noise_config["type"]
        pipeline.model.config.cc_noise_strength = noise_config["strength"]
        
        # Generate multiple predictions with logit perturbation
        all_predictions = []
        all_logits = [] if args.perturb_type == "Logits" else None
        if(args.perturb_type=="Logits") : 
          with torch.no_grad():
              for _ in range(noise_config["num_perturbations"]):
                  prediction, logits = pipeline.predict(
                      context=context,
                      prediction_length=args.prediction_length,
                      num_samples=args.num_samples,
                      return_logits=True
                  )
                  all_predictions.append(prediction.cpu().numpy())
                  all_logits.append(logits)
        
          # Aggregate predictions and calculate probabilities
          calibrated_predictions, probs, mean_logits = aggregate_predictions(all_predictions, all_logits)


        elif (args.perturb_type=="Input"):
          with torch.no_grad():
              for _ in range(noise_config["num_perturbations"]):
                  prediction = pipeline.predict(
                      context=context,
                      prediction_length=args.prediction_length,
                      num_samples=args.num_samples
                  )
                  all_predictions.append(prediction.cpu().numpy())
        
          # Aggregate predictions and calculate probabilities
        
        
        calibrated_predictions, probs, mean_logits = aggregate_predictions(all_predictions, all_logits, args.perturb_type)

        # Calculate traditional calibration metrics
        calibration_metrics = evaluate_calibration(
            original_forecast,
            calibrated_predictions,
            test_targets,
            context=context,
            config_name=config_key,
            noise_config=noise_config
        )
        
        # Calculate ECE metrics
        ece_metrics = analyze_calibration(
            original_forecast,
            torch.from_numpy(calibrated_predictions),
            test_targets,
            tolerance=args.ece_tolerance,
            num_bins=args.ece_bins
        )
        
        combined_metrics = {
          **calibration_metrics,
          **ece_metrics,
          'num_perturbations': noise_config["num_perturbations"],
          'model_variance_ratio': float(calibrated_predictions.var().mean() / original_forecast.var().mean().item()),
        }

        # Only add logit-related metrics if mean_logits is not None (i.e., when perturb_type="Logits")
        if mean_logits is not None and probs is not None:
            combined_metrics.update({
                'mean_logit': float(mean_logits.mean().item()),
                'logit_std': float(mean_logits.std().item()),
                'max_prob': float(probs.max().item()),
                'min_prob': float(probs.min().item())
            })
        
        results[config_key] = combined_metrics
        
        # Print summary
        print(f"\nResults for {config_key}:")
        print("Coverage rate: {:.3f} -> {:.3f} (improvement: {:.3f})".format(
            calibration_metrics['original_coverage_rate'],
            calibration_metrics['calibrated_coverage_rate'],
            calibration_metrics['coverage_improvement']
        ))
        print("ECE: {:.3f} -> {:.3f} (reduction: {:.1f}%)".format(
            ece_metrics['original_ece'],
            ece_metrics['calibrated_ece'],
            ece_metrics['ece_reduction_percent']
        ))
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot predictions for each configuration
    for noise_config in noise_configs:
      config_key = f"{noise_config['dist']}_{noise_config['type']}_{noise_config['strength']}_{noise_config['num_perturbations']}"
      save_path = output_dir / f"prediction_plot_{args.perturb_type}.png"

      plot_prediction_comparison(
      context=context,
      targets=test_targets,
      original_forecast=original_forecast,
      calibrated_predictions=calibrated_predictions,
      noise_config={
          "dist": args.noise_dist,
          "type": args.noise_type,
          "strength": args.noise_strength,
          "num_perturbations": args.num_perturbations  # Ensure this key is included
      },
      perturb_type=args.perturb_type,
      save_path=save_path
      )

    # Plot perturbation effects
    results_df = process_results_for_plotting(results)
    print("Results DataFrame columns:", results_df.columns)  # Debugging output
    plot_perturbation_effects(
        results_df,
        save_path=plots_dir / "perturbation_effects.png"
    )
        
    # Save detailed results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Save metrics summary
    metrics_df = pd.DataFrame(results).T.reset_index()
    metrics_df = metrics_df.rename(columns={'index': 'config_key'})
    metrics_df.to_csv(output_dir / "calibration_metrics.csv", index=False)
    
    print("\nCalibration complete. Results saved to:", output_dir)
    print("Detailed metrics saved to:", output_dir / "calibration_metrics.csv")

def main():
    parser = argparse.ArgumentParser(description="Consistency Calibration for Chronos (C3)")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="amazon/chronos-t5-small",
                      help="Name or path of the Chronos model")
    parser.add_argument("--use_bfloat16", action="store_true",
                      help="Use bfloat16 precision")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                      help="Path to CSV data file")
    parser.add_argument("--target_column", type=str, required=True,
                      help="Name of target column in CSV")
    
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
