import logging
from pathlib import Path
from typing import Iterable, Optional

import datasets
import numpy as np
import pandas as pd
import torch
import typer
import yaml
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast, SampleForecast
from tqdm.auto import tqdm
import scipy.stats as stats

from chronos import (
    BaseChronosPipeline,
    ChronosBoltPipeline,
    ChronosPipeline,
    ForecastType,
)

app = typer.Typer(pretty_exceptions_enable=False)

# Taken from pandas._libs.tslibs.dtypes.OFFSET_TO_PERIOD_FREQSTR
offset_alias_to_period_alias = {
    "WEEKDAY": "D",
    "EOM": "M",
    "BME": "M",
    "SME": "M",
    "BQS": "Q",
    "QS": "Q",
    "BQE": "Q",
    "BQE-DEC": "Q",
    "BQE-JAN": "Q",
    "BQE-FEB": "Q",
    "BQE-MAR": "Q",
    "BQE-APR": "Q",
    "BQE-MAY": "Q",
    "BQE-JUN": "Q",
    "BQE-JUL": "Q",
    "BQE-AUG": "Q",
    "BQE-SEP": "Q",
    "BQE-OCT": "Q",
    "BQE-NOV": "Q",
    "MS": "M",
    "D": "D",
    "B": "B",
    "min": "min",
    "s": "s",
    "ms": "ms",
    "us": "us",
    "ns": "ns",
    "h": "h",
    "QE": "Q",
    "QE-DEC": "Q-DEC",
    "QE-JAN": "Q-JAN",
    "QE-FEB": "Q-FEB",
    "QE-MAR": "Q-MAR",
    "QE-APR": "Q-APR",
    "QE-MAY": "Q-MAY",
    "QE-JUN": "Q-JUN",
    "QE-JUL": "Q-JUL",
    "QE-AUG": "Q-AUG",
    "QE-SEP": "Q-SEP",
    "QE-OCT": "Q-OCT",
    "QE-NOV": "Q-NOV",
    "YE": "Y",
    "YE-DEC": "Y-DEC",
    "YE-JAN": "Y-JAN",
    "YE-FEB": "Y-FEB",
    "YE-MAR": "Y-MAR",
    "YE-APR": "Y-APR",
    "YE-MAY": "Y-MAY",
    "YE-JUN": "Y-JUN",
    "YE-JUL": "Y-JUL",
    "YE-AUG": "Y-AUG",
    "YE-SEP": "Y-SEP",
    "YE-OCT": "Y-OCT",
    "YE-NOV": "Y-NOV",
    "W": "W",
    "ME": "M",
    "Y": "Y",
    "BYE": "Y",
    "BYE-DEC": "Y",
    "BYE-JAN": "Y",
    "BYE-FEB": "Y",
    "BYE-MAR": "Y",
    "BYE-APR": "Y",
    "BYE-MAY": "Y",
    "BYE-JUN": "Y",
    "BYE-JUL": "Y",
    "BYE-AUG": "Y",
    "BYE-SEP": "Y",
    "BYE-OCT": "Y",
    "BYE-NOV": "Y",
    "YS": "Y",
    "BYS": "Y",
    "QS-JAN": "Q",
    "QS-FEB": "Q",
    "QS-MAR": "Q",
    "QS-APR": "Q",
    "QS-MAY": "Q",
    "QS-JUN": "Q",
    "QS-JUL": "Q",
    "QS-AUG": "Q",
    "QS-SEP": "Q",
    "QS-OCT": "Q",
    "QS-NOV": "Q",
    "QS-DEC": "Q",
    "BQS-JAN": "Q",
    "BQS-FEB": "Q",
    "BQS-MAR": "Q",
    "BQS-APR": "Q",
    "BQS-MAY": "Q",
    "BQS-JUN": "Q",
    "BQS-JUL": "Q",
    "BQS-AUG": "Q",
    "BQS-SEP": "Q",
    "BQS-OCT": "Q",
    "BQS-NOV": "Q",
    "BQS-DEC": "Q",
    "YS-JAN": "Y",
    "YS-FEB": "Y",
    "YS-MAR": "Y",
    "YS-APR": "Y",
    "YS-MAY": "Y",
    "YS-JUN": "Y",
    "YS-JUL": "Y",
    "YS-AUG": "Y",
    "YS-SEP": "Y",
    "YS-OCT": "Y",
    "YS-NOV": "Y",
    "YS-DEC": "Y",
    "BYS-JAN": "Y",
    "BYS-FEB": "Y",
    "BYS-MAR": "Y",
    "BYS-APR": "Y",
    "BYS-MAY": "Y",
    "BYS-JUN": "Y",
    "BYS-JUL": "Y",
    "BYS-AUG": "Y",
    "BYS-SEP": "Y",
    "BYS-OCT": "Y",
    "BYS-NOV": "Y",
    "BYS-DEC": "Y",
    "Y-JAN": "Y-JAN",
    "Y-FEB": "Y-FEB",
    "Y-MAR": "Y-MAR",
    "Y-APR": "Y-APR",
    "Y-MAY": "Y-MAY",
    "Y-JUN": "Y-JUN",
    "Y-JUL": "Y-JUL",
    "Y-AUG": "Y-AUG",
    "Y-SEP": "Y-SEP",
    "Y-OCT": "Y-OCT",
    "Y-NOV": "Y-NOV",
    "Y-DEC": "Y-DEC",
    "Q-JAN": "Q-JAN",
    "Q-FEB": "Q-FEB",
    "Q-MAR": "Q-MAR",
    "Q-APR": "Q-APR",
    "Q-MAY": "Q-MAY",
    "Q-JUN": "Q-JUN",
    "Q-JUL": "Q-JUL",
    "Q-AUG": "Q-AUG",
    "Q-SEP": "Q-SEP",
    "Q-OCT": "Q-OCT",
    "Q-NOV": "Q-NOV",
    "Q-DEC": "Q-DEC",
    "W-MON": "W-MON",
    "W-TUE": "W-TUE",
    "W-WED": "W-WED",
    "W-THU": "W-THU",
    "W-FRI": "W-FRI",
    "W-SAT": "W-SAT",
    "W-SUN": "W-SUN",
}

import numpy as np
import scipy.stats as stats

def compute_skewness_ratio(data):
    """
    Computes the skewness ratio for dynamic quantile selection.

    Parameters:
    - data (array-like): Input time series data.

    Returns:
    - skew_ratio (float): Skewness ratio (Mean - Median) / Std Dev.
    - skewness_value (float): Traditional skewness measure.
    """
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_dev = np.std(data, ddof=1)
    skewness_val = stats.skew(data)

    if std_dev == 0:  # Avoid division by zero
        skew_ratio = 0
    else:
        skew_ratio = (mean_val - median_val) / std_dev

    # ✅ Ensure skew ratio is within a reasonable range
    skew_ratio = np.clip(skew_ratio, -3, 3)

    # ✅ Ensure skew ratio is finite (not NaN or Inf)
    if not np.isfinite(skew_ratio):
        skew_ratio = 0

    return skew_ratio, skewness_val

def dynamic_quantile_selection(skew_ratio, alpha=1.5):
    """
    Dynamically selects the best quantile based on skewness ratio.

    Parameters:
    - skew_ratio (float): Computed skewness ratio.
    - alpha (float): Sensitivity factor for scaling.

    Returns:
    - best_quantile (float): Dynamically chosen quantile (between 0.1 and 0.9).
    """
    # ✅ Ensure skew ratio is within a reasonable range
    skew_ratio = np.clip(skew_ratio, -3, 3)

    # ✅ Compute dynamic quantile mapping
    quantile = 0.5 + 0.4 * np.tanh(alpha * skew_ratio)  

    # ✅ Final clamp to ensure it's strictly between 0.1 and 0.9
    quantile = np.clip(quantile, 0.1, 0.9)

    return round(quantile, 2)


def analyze_gluonts_dataset(trainset):
    """
    Extracts the target column from a GluonTS dataset and selects the best dynamic quantile.

    Parameters:
    - trainset (GluonTS dataset): The training dataset.

    Returns:
    - best_quantile (float): Dynamically selected quantile.
    - skew_ratio (float): Computed skewness ratio.
    """
    all_targets = []

    # Extract target values from all series
    for entry in trainset:
        all_targets.extend(entry["target"])

    all_targets = np.array(all_targets)

    # Compute skewness ratio
    skew_ratio, skewness_value = compute_skewness_ratio(all_targets)

    # Select the best quantile dynamically
    best_quantile = dynamic_quantile_selection(skew_ratio)

    return best_quantile, skew_ratio, skewness_value


def analyze_gluonts_dataset_volatility(trainset, window=30):
    """
    Extracts the target column from a GluonTS dataset and computes overall volatility level.

    Parameters:
    - trainset (GluonTS dataset): The training dataset.
    - window (int): Rolling window size for computing volatility.

    Returns:
    - overall_volatility_level (str): High, Moderate, or Low.
    - mean_cv (float): Mean Coefficient of Variation across all series.
    """

    all_targets = []

    # Extract target values from all series
    for entry in trainset:
        all_targets.extend(entry["target"])

    all_targets = np.array(all_targets)

    # ✅ Check for empty data
    if len(all_targets) == 0:
        print("⚠️ Warning: No data found in training set!")
        return "Unknown Volatility", np.nan  # Return unknown volatility if empty

    # ✅ Compute standard deviation & mean, handling edge cases
    std_dev = np.std(all_targets)

    if std_dev == 0 or not np.isfinite(std_dev):
        print("⚠️ Warning: Standard deviation is zero or invalid, setting CV to NaN")
        return "Unknown Volatility", np.nan

    # ✅ Clip extreme values
    std_dev = np.clip(std_dev, 0, 1)  # Since data is normalized

    # Determine overall volatility level based on dataset-wide standard deviation
    if std_dev > 0.2:  # Adjust this threshold based on dataset behavior
        overall_volatility_level = "High Volatility"
    elif std_dev > 0.05:
        overall_volatility_level = "Moderate Volatility"
    else:
        overall_volatility_level = "Low Volatility"

    return overall_volatility_level, std_dev


def apply_residual_alignment(forecasted, actual, volatility_level, window=30):
    """
    Applies residual alignment dynamically based on detected volatility level.

    Parameters:
    - forecasted (array-like): Model's forecasted values.
    - actual (array-like): Ground truth values.
    - volatility_level (str): "High", "Moderate", or "Low".
    - window (int): Rolling window size.

    Returns:
    - adjusted_forecast (array-like): Forecast after residual correction.
    """
    forecasted, actual = np.array(forecasted), np.array(actual)
    residuals = actual[-1] - forecasted  # Compute residuals

    if volatility_level == "High Volatility":
        # ❌ Skip residual alignment
        adjusted_forecast = forecasted

    elif volatility_level == "Moderate Volatility":
        # ✅ Use rolling window residual correction (smoother adjustment)
        rolling_residuals = pd.Series(residuals).rolling(window=window).mean()
        adjusted_forecast = forecasted + rolling_residuals.fillna(0).values  # Fill NaN with 0

    else:  # Low Volatility
        # ✅ Apply full residual alignment (adjust first forecasted point to match last observed)
        correction_factor = actual[-1] - forecasted[0]  # Align first predicted value
        adjusted_forecast = forecasted + correction_factor

    return adjusted_forecast



def to_gluonts_univariate(hf_dataset: datasets.Dataset):
    series_fields = [
        col
        for col in hf_dataset.features
        if isinstance(hf_dataset.features[col], datasets.Sequence)
    ]
    series_fields.remove("timestamp")
    dataset_length = hf_dataset.info.splits["train"].num_examples * len(series_fields)
    dataset_freq = pd.infer_freq(hf_dataset[0]["timestamp"])
    dataset_freq = offset_alias_to_period_alias.get(dataset_freq, dataset_freq)

    gts_dataset = []
    for hf_entry in hf_dataset:
        for field in series_fields:
            gts_dataset.append(
                {
                    "start": pd.Period(
                        hf_entry["timestamp"][0],
                        freq=dataset_freq,
                    ),
                    "target": hf_entry[field],
                }
            )
    assert len(gts_dataset) == dataset_length

    return gts_dataset


def load_and_split_dataset(backtest_config: dict):
    hf_repo = backtest_config["hf_repo"]
    dataset_name = backtest_config["name"]
    offset = backtest_config["offset"]
    prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]

    # This is needed because the datasets in autogluon/chronos_datasets_extra cannot
    # be distribued due to license restrictions and must be generated on the fly
    trust_remote_code = True if hf_repo == "autogluon/chronos_datasets_extra" else False

    ds = datasets.load_dataset(
        hf_repo, dataset_name, split="train", trust_remote_code=trust_remote_code
    )
    ds.set_format("numpy")

    gts_dataset = to_gluonts_univariate(ds)

    # Split dataset for evaluation
    train_data, test_template = split(gts_dataset, offset=offset)
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls)

    return (train_data,test_data)


def generate_forecasts(
    global_volatility_level,
    test_data_input: Iterable,
    pipeline: BaseChronosPipeline,
    prediction_length: int,
    batch_size: int,
    **predict_kwargs,
):
    

    # 🚀 Forecast Generation with Global Residual Alignment Strategy
    forecast_outputs = []
    for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
        context = [torch.tensor(entry["target"]) for entry in batch]
        
        # Pass updated `predict_kwargs` with global volatility level
        output = pipeline.predict(
            context,
            prediction_length=prediction_length,
            **predict_kwargs,  # Now contains global_volatility_level
        ).numpy()

        # Process each time series in the batch
        for i in range(len(context)):  
            actual_values = np.array(context[i])  # Get historical series
            forecasted_samples = output[i]  # All forecast samples for this series

            # Apply residual alignment based on **GLOBAL** volatility level
            aligned_samples = []
            for sample in forecasted_samples:  
                adjusted_sample = apply_residual_alignment(sample, actual_values, global_volatility_level)
                aligned_samples.append(adjusted_sample)

            forecast_outputs.append(np.array(aligned_samples))  

    # Convert list of arrays into final numpy array
    forecast_outputs = np.array(forecast_outputs)

    # Convert forecast samples into gluonts Forecast objects
    forecasts = []
    for item, ts in zip(forecast_outputs, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])

        if pipeline.forecast_type == ForecastType.SAMPLES:
            forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )
        elif pipeline.forecast_type == ForecastType.QUANTILES:
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, pipeline.quantiles)),
                    start_date=forecast_start_date,
                )
            )

    return forecasts
    

@app.command()
def main(
    config_path: Path,
    metrics_path: Path,
    chronos_model_id: str = "amazon/chronos-t5-small",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    batch_size: int = 32,
    num_samples: int = 20,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    """Evaluate Chronos models.

    Parameters
    ----------
    config_path : Path
        Path to the evaluation config. See ./configs/.
    metrics_path : Path
        Path to the CSV file where metrics will be saved.
    chronos_model_id : str, optional, default = "amazon/chronos-t5-small"
        HuggingFace ID of the Chronos model or local path
        Available models on HuggingFace:
        Chronos:
            - amazon/chronos-t5-tiny
            - amazon/chronos-t5-mini
            - amazon/chronos-t5-small
            - amazon/chronos-t5-base
            - amazon/chronos-t5-large
        Chronos-Bolt:
            - amazon/chronos-bolt-tiny
            - amazon/chronos-bolt-mini
            - amazon/chronos-bolt-small
            - amazon/chronos-bolt-base
    device : str, optional, default = "cuda"
        Device on which inference will be performed
    torch_dtype : str, optional
        Model's dtype, by default "bfloat16"
    batch_size : int, optional, default = 32
        Batch size for inference. For Chronos-Bolt models, significantly larger
        batch sizes can be used
    num_samples : int, optional, default = 20
        Number of samples to draw when using the original Chronos models
    temperature : Optional[float], optional, default = 1.0
        Softmax temperature to used for the original Chronos models
    top_k : Optional[int], optional, default = 50
        Top-K sampling, by default None
    top_p : Optional[float], optional, default = 1.0
        Top-p sampling, by default None
    """
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    # Load Chronos
    pipeline = BaseChronosPipeline.from_pretrained(
        chronos_model_id,
        device_map=device,
        torch_dtype=torch_dtype,
    )

    if isinstance(pipeline, ChronosPipeline):
        predict_kwargs = dict(
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    elif isinstance(pipeline, ChronosBoltPipeline):
        predict_kwargs = {}

    

    # Load backtest configs
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    result_rows = []
    for config in backtest_configs:
        dataset_name = config["name"]
        prediction_length = config["prediction_length"]

        logger.info(f"Loading {dataset_name}")

        train_data,test_data = load_and_split_dataset(backtest_config=config)
        # Analyze dataset and get best quantile
        best_q, sr, skew_value = analyze_gluonts_dataset(train_data)
        
        global_volatility_level, mean_cv = analyze_gluonts_dataset_volatility(train_data)
        logger.info(f"✅ Global Volatility Level: {global_volatility_level} (std: {mean_cv:.3f})")
        

        logger.info(
            f"Generating forecasts for {dataset_name} "
            f"({len(test_data.input)} time series)"
        )
        forecasts = generate_forecasts(
            global_volatility_level,
            test_data.input,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            **predict_kwargs,
        )

        logger.info(f"Evaluating forecasts for {dataset_name}")
        metrics = (
            evaluate_forecasts(
                forecasts,
                test_data=test_data,
                metrics=[
                    MASE(forecast_type=best_q),
                    MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                ],
                batch_size=5000,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        result_rows.append(
            {"dataset": dataset_name, "model": chronos_model_id, **metrics[0]}
        )


    # Convert result_rows to DataFrame
    results_df = pd.DataFrame(result_rows)

    # ✅ Find all columns that contain "MASE"
    mase_columns = [col for col in results_df.columns if "MASE" in col]

    # ✅ Create a new column "MASE" with the first non-null MASE value in each row
    results_df["MASE"] = results_df[mase_columns].bfill(axis=1).iloc[:, 0]

    # ✅ Rename WQL column
    results_df = results_df.rename(columns={"mean_weighted_sum_quantile_loss": "WQL"})

    # ✅ Drop the original MASE columns (optional)
    results_df = results_df.drop(columns=mase_columns)

    # ✅ Sort by dataset for cleaner output
    results_df = results_df.sort_values(by="dataset")

    # ✅ Save to CSV
    results_df.to_csv(metrics_path, index=False)

    logger.info(f"Metrics saved to {metrics_path}")



if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)
    app()
