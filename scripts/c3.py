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

def aggregate_predictions(all_predictions, all_logits=None):
    """Aggregate predictions from multiple perturbations."""
    # Stack predictions properly preserving the sample dimension
    stacked_predictions = np.stack(all_predictions, axis=0)  # (n_perturbations, batch, n_samples, seq_len)
    
    # Average across perturbations while keeping sample dimension
    mean_predictions = np.mean(stacked_predictions, axis=0)  # (batch, n_samples, seq_len)
    
    if all_logits is not None:
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
    
    return mean_predictions

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
        
        # Aggregate predictions and calculate probabilities
        calibrated_predictions, probs, mean_logits = aggregate_predictions(all_predictions, all_logits)
        
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
        
        # Combine all metrics
        combined_metrics = {
            **calibration_metrics,
            **ece_metrics,
            'num_perturbations': noise_config["num_perturbations"],
            'model_variance_ratio': float(calibrated_predictions.var().mean() / original_forecast.var().mean().item()),
            'mean_logit': float(mean_logits.mean().item()),
            'logit_std': float(mean_logits.std().item()),
            'max_prob': float(probs.max().item()),
            'min_prob': float(probs.min().item())
        }
        
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