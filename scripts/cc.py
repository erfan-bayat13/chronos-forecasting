import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline

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
        else:  # uniform
            noise = 1 + np.random.uniform(-self.noise_strength, self.noise_strength, size=context.shape)
            
        perturbed_context = context * noise
        return torch.tensor(perturbed_context)

    def __call__(self, pipeline, context, prediction_length):
        all_predictions = []
        for _ in range(self.num_perturbations):
            perturbed_context = self.perturb_context(context)
            forecast = pipeline.predict(
                context=perturbed_context,
                prediction_length=prediction_length
            )
            all_predictions.append(forecast[0].numpy())
            
        return np.stack(all_predictions, axis=0)

def plot_calibrated_forecast(context, forecast_index, original_forecast, calibrated_predictions, noise_type):
    """Create a single plot for one type of calibration"""
    plt.figure(figsize=(12, 6))
    
    # Calculate quantiles
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

    # Get original forecast
    original_forecast = pipeline.predict(
        context=context,
        prediction_length=prediction_length
    )

    # Create calibrated forecasts with different noise types
    for noise_type in ['gaussian', 'uniform']:
        calibrator = ConsistencyCalibration(
            num_perturbations=32, 
            noise_type=noise_type, 
            noise_strength=0.1
        )
        calibrated_predictions = calibrator(pipeline, context, prediction_length)
        
        # Create separate plot for each noise type
        plot_calibrated_forecast(
            context,
            forecast_index,
            original_forecast,
            calibrated_predictions,
            noise_type
        )

if __name__ == "__main__":
    main()