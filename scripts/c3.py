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

def plot_calibrated_forecast(output_dir, context, forecast_index, original_forecast, 
                           calibrated_predictions, noise_config, logit_probs=None):
    """Create and save plot comparing original and calibrated forecasts."""
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
    plt.title(f"{noise_config['dist'].capitalize()} {noise_config['type']} Noise")
    
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
    
    # Load pipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model_name,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if args.use_bfloat16 else torch.float32,
    )
    
    # Load data
    context = pd.read_csv(args.data_path)[args.target_column].values
    context = torch.tensor(context)
    forecast_index = range(len(context), len(context) + args.prediction_length)
    
    # Get original forecast
    pipeline.model.config.use_cc = False
    original_forecast, original_logits = pipeline.predict(
        context=context,
        prediction_length=args.prediction_length,
        num_samples=args.num_samples,
        return_logits=True
    )
    
    # Test different noise configurations
    noise_configs = []
    for dist in args.noise_dist.split(','):
        for noise_type in args.noise_type.split(','):
            for strength in [float(s) for s in args.noise_strength.split(',')]:
                noise_configs.append({
                    "dist": dist,
                    "type": noise_type,
                    "strength": strength
                })
    
    results = {}
    for noise_config in noise_configs:
        print(f"\nTesting {noise_config['dist']} {noise_config['type']} noise (strength={noise_config['strength']}):")
        
        # Enable CC with current configuration
        pipeline.model.config.use_cc = True
        pipeline.model.config.cc_noise_dist = noise_config["dist"]
        pipeline.model.config.cc_noise_type = noise_config["type"]
        pipeline.model.config.cc_noise_strength = noise_config["strength"]
        
        # Generate multiple predictions
        all_predictions = []
        all_logits = []
        
        for _ in range(args.num_perturbations):
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
        
        # Plot results
        plot_calibrated_forecast(
            output_dir,
            context,
            forecast_index,
            original_forecast,
            calibrated_predictions,
            noise_config,
            probs
        )
        
        # Collect statistics
        config_key = f"{noise_config['dist']}_{noise_config['type']}_{noise_config['strength']}"
        results[config_key] = {
            "original_variance": float(original_forecast.var().mean().item()),
            "calibrated_variance": float(calibrated_predictions.var().mean()),
            "mean_logit": float(mean_logits.mean().item()),
            "logit_std": float(mean_logits.std().item()),
            "max_prob": float(probs.max().item()),
            "min_prob": float(probs.min().item())
        }
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

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
    parser.add_argument("--num_perturbations", type=int, default=32,
                      help="Number of perturbations for calibration")
    
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