import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline
from torch.nn.functional import softmax
from matplotlib.colors import LogNorm
import warnings
from pathlib import Path
import json
from calibration_metrics import evaluate_calibration

def plot_calibrated_forecast(output_dir, context, forecast_index, original_forecast, 
                           calibrated_predictions, noise_config, logit_probs=None):
    """Create and save plot comparing original and calibrated forecasts."""
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    
    original_forecast_array = original_forecast[0, :, :].numpy()
    original_low, original_median, original_high = np.quantile(
        original_forecast_array, [0.1, 0.5, 0.9], axis=0
    )
    
    calibrated_low, calibrated_median, calibrated_high = np.quantile(
        calibrated_predictions[:, 0, :, :], [0.1, 0.5, 0.9], axis=(0, 1)
    )
    
    plt.plot(context.numpy(), color="royalblue", label="Historical data")
    plt.plot(forecast_index, original_median, '--', color="tomato", label="Original median forecast")
    plt.fill_between(forecast_index, original_low, original_high, color="tomato", alpha=0.2, label="Original 80% PI")
    plt.plot(forecast_index, calibrated_median, '-', color="green", label="Calibrated median forecast")
    plt.fill_between(forecast_index, calibrated_low, calibrated_high, color="green", alpha=0.2, label="Calibrated 80% PI")
    
    plt.legend()
    plt.grid(True)
    plt.title(f"{noise_config['dist'].capitalize()} {noise_config['type']} Noise (n={noise_config['num_perturbations']})")
    
    if logit_probs is not None:
        plt.subplot(1, 2, 2)
        mean_probs = logit_probs.mean(dim=1)
        
        nonzero_probs = (mean_probs > 1e-4).any(dim=0)
        token_indices = torch.where(nonzero_probs)[0]
        min_token = max(token_indices.min().item() - 100, 0)
        max_token = min(token_indices.max().item() + 100, mean_probs.shape[1])
        
        plt.imshow(mean_probs[:, min_token:max_token].cpu().numpy(),
                  aspect='auto',
                  cmap='viridis',
                  norm=LogNorm(vmin=1e-4))
        plt.colorbar(label='Log Probability')
        plt.title('Token Probabilities')
        plt.xlabel(f'Token ID ({min_token}-{max_token})')
        plt.ylabel('Prediction Step')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{noise_config['dist']}_{noise_config['type']}_strength_{noise_config['strength']}.png"
    plt.savefig(output_dir / plot_filename)
    plt.close()

def aggregate_predictions(all_predictions, all_logits=None):
    """Aggregate predictions from multiple perturbations."""
    stacked_predictions = np.stack(all_predictions, axis=0)
    
    if all_logits is not None:
        stacked_logits = torch.stack(all_logits, dim=0)
        max_val = stacked_logits.max()
        min_val = stacked_logits.min()
        stacked_logits = (stacked_logits - min_val) / (max_val - min_val + 1e-6)
        
        mean_logits = torch.mean(stacked_logits, dim=0)
        temperature = 1.0
        mean_logits = mean_logits / temperature
        
        probs = softmax(mean_logits + 1e-6, dim=-1)
        
        return stacked_predictions, probs, mean_logits
    
    return stacked_predictions

def run_calibration(args):
    """Run consistency calibration with given parameters."""
    warnings.filterwarnings("ignore")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for results
    plots_dir = output_dir / "plots"
    calibration_dir = output_dir / "calibration"
    for d in [plots_dir, calibration_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Load pipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model_name,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if args.use_bfloat16 else torch.float32,
    )
    
    # Load data
    df = pd.read_csv(args.data_path)
    full_data = df[args.target_column].values
    
    # Split data into context and test
    context_length = len(full_data) - args.prediction_length
    if context_length <= 0:
        raise ValueError(f"Data length ({len(full_data)}) must be greater than prediction_length ({args.prediction_length})")
    
    context = torch.tensor(full_data[:context_length])
    test_targets = torch.tensor(full_data[context_length:context_length + args.prediction_length])
    forecast_index = range(context_length, context_length + args.prediction_length)
    
    print(f"Using {len(context)} points for context and {len(test_targets)} points for evaluation")
    
    # Get original forecast
    pipeline.model.config.use_cc = False
    with torch.no_grad():
        original_forecast, original_logits = pipeline.predict(
            context=context,
            prediction_length=args.prediction_length,
            num_samples=args.num_samples,
            return_logits=True
        )
    original_forecast = original_forecast.type(torch.float32)
    
    # Parse perturbation counts
    perturbation_counts = [int(x) for x in args.num_perturbations.split(',')]
    
    # Create combinations of all parameters
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
        # Create config key
        config_key = f"{noise_config['dist']}_{noise_config['type']}_{noise_config['strength']}_{noise_config['num_perturbations']}"
        print(f"\nTesting {noise_config['dist']} {noise_config['type']} noise "
              f"(strength={noise_config['strength']}, perturbations={noise_config['num_perturbations']}):")
        
        # Create subdirectory for this perturbation count
        pert_dir = plots_dir / str(noise_config['num_perturbations'])
        pert_dir.mkdir(parents=True, exist_ok=True)
        
        # Enable CC with current configuration
        pipeline.model.config.use_cc = True
        pipeline.model.config.cc_noise_dist = noise_config["dist"]
        pipeline.model.config.cc_noise_type = noise_config["type"]
        pipeline.model.config.cc_noise_strength = noise_config["strength"]
        
        # Generate multiple predictions
        all_predictions = []
        all_logits = []
        
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
        
        # Aggregate predictions
        calibrated_predictions, probs, mean_logits = aggregate_predictions(all_predictions, all_logits)
        
        # Create visualizations and save results
        plot_calibrated_forecast(
            pert_dir,
            context,
            forecast_index,
            original_forecast,
            calibrated_predictions,
            noise_config,
            probs
        )
        
        # Calculate calibration metrics
        calibration_metrics = evaluate_calibration(
            original_forecast,
            calibrated_predictions,
            test_targets,
            context=context,
            output_dir=calibration_dir / str(noise_config["num_perturbations"]),
            config_name=config_key,
            noise_config=noise_config
        )
        
        # Add additional statistics
        metrics_with_stats = {
            **calibration_metrics,
            'num_perturbations': noise_config["num_perturbations"],
            'model_variance_ratio': float(calibrated_predictions.var().mean() / original_forecast.var().mean().item()),
            'mean_logit': float(mean_logits.mean().item()),
            'logit_std': float(mean_logits.std().item()),
            'max_prob': float(probs.max().item()),
            'min_prob': float(probs.min().item())
        }
        
        results[config_key] = metrics_with_stats
        
        print(f"\nResults for {config_key}:")
        print("Coverage rate: {:.3f} -> {:.3f} (improvement: {:.3f})".format(
            calibration_metrics['original_coverage_rate'],
            calibration_metrics['calibrated_coverage_rate'],
            calibration_metrics['coverage_improvement']
        ))
        print("Interval width: {:.3f} -> {:.3f} (reduction: {:.3f})".format(
            calibration_metrics['original_interval_width'],
            calibration_metrics['calibrated_interval_width'],
            calibration_metrics['width_reduction']
        ))
    
    # Save detailed results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Create DataFrame with results
    metrics_df = pd.DataFrame(results).T.reset_index()
    metrics_df = metrics_df.rename(columns={'index': 'config_key'})
    
    # Create perturbation analysis plots
    for metric, title in [
        ('coverage_improvement', 'Coverage Improvement'),
        ('width_reduction', 'Width Reduction'),
        ('error_reduction', 'Error Reduction')
    ]:
        plt.figure(figsize=(10, 6))
        for dist in args.noise_dist.split(','):
            for noise_type in args.noise_type.split(','):
                for strength in [float(s) for s in args.noise_strength.split(',')]:
                    mask = (metrics_df['noise_distribution'] == dist) & \
                          (metrics_df['noise_type'] == noise_type) & \
                          (metrics_df['noise_strength'] == strength)
                    
                    data = metrics_df[mask].sort_values('num_perturbations')
                    plt.plot(data['num_perturbations'], 
                            data[metric], 
                            marker='o',
                            label=f'{dist}-{noise_type}-{strength}')
        
        plt.title(f'{title} vs Number of Perturbations')
        plt.xlabel('Number of Perturbations')
        plt.ylabel(title)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(plots_dir / f'{metric}_vs_perturbations.png', bbox_inches='tight')
        plt.close()
    
    # Save metrics to CSV
    metrics_df.to_csv(output_dir / "calibration_metrics.csv", index=False)
    
    print("\nCalibration complete. Results saved to:", output_dir)
    print("Detailed metrics saved to:", output_dir / "calibration_metrics.csv")
    print("Analysis plots saved in:", plots_dir)

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
    parser.add_argument("--noise_dist", type=str, default="gaussian,uniform",
                      help="Comma-separated list of noise distributions to test")
    parser.add_argument("--noise_type", type=str, default="multiplicative,additive",
                      help="Comma-separated list of noise types to test")
    parser.add_argument("--noise_strength", type=str, default="0.01,0.05,0.1",
                      help="Comma-separated list of noise strengths to test")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./c3_output",
                      help="Directory for output files")
    
    args = parser.parse_args()
    run_calibration(args)

if __name__ == "__main__":
    main()