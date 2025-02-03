import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from chronos import BaseChronosPipeline
from torch.nn.functional import softmax

class ConsistencyCalibration:
    def __init__(self, num_perturbations=32, noise_type='gaussian', noise_strength=0.1):
        self.num_perturbations = num_perturbations
        self.noise_type = noise_type.lower()
        self.noise_strength = noise_strength

        if self.noise_type not in ['gaussian', 'uniform']:
            raise ValueError("noise_type must be either 'gaussian' or 'uniform'")

    def perturb_context(self, context):
        if isinstance(context, torch.Tensor):
            context = context.numpy()

        if self.noise_type == 'gaussian':
            noise = 1 + np.random.normal(0, self.noise_strength, size=context.shape)
        elif self.noise_type == "uniform":  # uniform
            noise = 1 + np.random.uniform(-self.noise_strength, self.noise_strength, size=context.shape)

        perturbed_context = context * noise
        return torch.tensor(perturbed_context)

    def perturb_context_additive(self, context):
        if isinstance(context, torch.Tensor):
            context = context.numpy()

        if self.noise_type == 'gaussian':
            noise = 1 + np.random.normal(0, self.noise_strength, size=context.shape)
        elif self.noise_type == "uniform":  # uniform
            noise = 1 + np.random.uniform(-self.noise_strength, self.noise_strength, size=context.shape)

        perturbed_context = context + noise
        return torch.tensor(perturbed_context)

    def aggregate_logits(self, logits_list):
      # Stack logits from all perturbations
      stacked_logits = torch.stack(logits_list, dim=0)

      # Normalize across all dimensions
      max_val = stacked_logits.max()
      min_val = stacked_logits.min()
      stacked_logits = (stacked_logits - min_val) / (max_val - min_val + 1e-6)

      # Average logits across perturbations
      mean_logits = torch.mean(stacked_logits, dim=0)

      # Apply gentler temperature scaling
      temperature = 1.0
      mean_logits = mean_logits / temperature

      # Add small epsilon to prevent numerical instability
      probs = softmax(mean_logits + 1e-6, dim=-1)

      return probs, mean_logits

    def __call__(self, pipeline, context, prediction_length):
        all_predictions = []
        all_logits = []

        for _ in range(self.num_perturbations):
            perturbed_context = self.perturb_context(context)
            forecast, logits = pipeline.predict(
                context=perturbed_context,
                prediction_length=prediction_length,
                return_logits=True
            )
            all_predictions.append(forecast[0].numpy())
            all_logits.append(logits)

        # Aggregate predictions and logits
        predictions = np.stack(all_predictions, axis=0)
        probs, mean_logits = self.aggregate_logits(all_logits)

        return predictions, probs, mean_logits

def plot_calibrated_forecast(context, forecast_index, original_forecast, calibrated_predictions, noise_type, logit_probs=None):
    """Create a single plot for one type of calibration"""
    plt.figure(figsize=(15, 8))

    # Create two subplots
    plt.subplot(1, 2, 1)

    # Calculate quantiles for predictions
    original_low, original_median, original_high = np.quantile(
        original_forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0
    )
    calibrated_low, calibrated_median, calibrated_high = np.quantile(
        calibrated_predictions, [0.1, 0.5, 0.9], axis=(0, 1)
    )

    # Plot historical data
    plt.plot(context.numpy(), color="royalblue", label="Historical data")

    # Plot original forecast
    plt.plot(forecast_index, original_median, '--', color="tomato", label="Original median forecast")
    plt.fill_between(forecast_index, original_low, original_high, color="tomato", alpha=0.2, label="Original 80% PI")

    # Plot calibrated forecast
    plt.plot(forecast_index, calibrated_median, '-', color="green", label="Calibrated median forecast")
    plt.fill_between(forecast_index, calibrated_low, calibrated_high, color="green", alpha=0.2, label="Calibrated 80% PI")

    plt.legend()
    plt.grid(True)
    plt.title(f"Air Passengers Forecast - {noise_type.capitalize()} Noise Calibration")

    # Add logit probabilities visualization if available
    if logit_probs is not None:
        plt.subplot(1, 2, 2)
        mean_probs = logit_probs.mean(dim=1)

        # Find token range with significant probabilities
        nonzero_probs = (mean_probs > 1e-4).any(dim=0)
        token_indices = torch.where(nonzero_probs)[0]
        min_token = max(token_indices.min().item() - 100, 0)  # Add some padding
        max_token = min(token_indices.max().item() + 100, mean_probs.shape[1])

        # Plot only the relevant token range
        plt.imshow(mean_probs[:, min_token:max_token].cpu().numpy(),
                  aspect='auto',
                  cmap='viridis',
                  norm=LogNorm(vmin=1e-4))
        plt.colorbar(label='Log Probability')
        plt.title('Token Probabilities Across Prediction Steps')
        plt.xlabel(f'Token ID (range {min_token}-{max_token})')
        plt.ylabel('Prediction Step')

    plt.tight_layout()
    plt.show()

def main():
    # Load pipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )

    # Load data
    df = pd.read_csv(
        "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    )
    context = torch.tensor(df["#Passengers"].values)
    prediction_length = 12
    forecast_index = range(len(context), len(context) + prediction_length)

    # Get original forecast with logits
    original_forecast, original_logits = pipeline.predict(
        context=context,
        prediction_length=prediction_length,
        return_logits=True
    )

    # Create calibrated forecasts with different noise types
    for noise_type in ['gaussian', 'uniform']:
        calibrator = ConsistencyCalibration(
            num_perturbations=50,
            noise_type=noise_type,
            noise_strength=0.01
        )
        calibrated_predictions, probs, mean_logits = calibrator(pipeline, context, prediction_length)

        # Create plot with both forecasts and logit probabilities
        plot_calibrated_forecast(
            context,
            forecast_index,
            original_forecast,
            calibrated_predictions,
            noise_type,
            probs
        )

        # Optional: Print some statistics about the logits
        print(f"\n{noise_type.capitalize()} Noise Calibration Statistics:")
        print(f"Mean logit value: {mean_logits.mean().item():.4f}")
        print(f"Logit std deviation: {mean_logits.std().item():.4f}")
        print(f"Max probability: {probs.max().item():.4f}")
        print(f"Min probability: {probs.min().item():.4f}")

if __name__ == "__main__":
    main()