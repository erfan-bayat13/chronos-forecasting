import random
import logging
from pathlib import Path
from typing import Iterable, Optional
from gluonts.dataset.arrow import ArrowFile
from multiprocessing import Pool
import multiprocessing

import datasets
import gluonts
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
from torch.nn.functional import cross_entropy
from torchmetrics.classification import MulticlassCalibrationError
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

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


def compute_arguments(num_splits, logits_list, n_perturbations, std, compute_naive):
    # Memory-optimized partitioning
    size = logits_list.shape[0]
    arguments = []

    # More even distribution with smaller chunks
    chunk_size = max(1, size // (num_splits * 2))

    for i in range(0, size, chunk_size):
        process_seed = random.randint(0, 1e9)
        end = min(i + chunk_size, size)
        chunk_id = i // chunk_size + 1
        arguments.append(
            (
                logits_list[i:end],
                n_perturbations,
                std,
                chunk_id,
                compute_naive,
                process_seed,
            )
        )

    return arguments


def compute_metrics_path(mode, model, std, n_pert):
    assert mode in [
        "naive",
        "consistency",
    ], "The metrics paths mode should be 'naive' or 'consistency'"
    model_name = model.split("/")[-1]
    return (
        f"Results/{model_name}_std_{std:.2f}_npert_{n_pert}_{mode}".replace(".", "_")
        + ".csv"
    )


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def compute_probabilities(
    logits_list,
    n_perturbations=10,
    std=0.1,
    instance=1,
    compute_naive=True,
    process_seed=None,
):
    naive_probs = []
    consistency_probs = []
    print(" ", end="", flush=True)
    desc = f"Parallel process {instance}"

    # Process in smaller batches with better memory management
    batch_size = 250  # Even smaller batch size for less memory pressure

    if process_seed:
        random.seed(process_seed)
        np.random.seed(process_seed)
        torch.manual_seed(process_seed)

    for i in range(0, len(logits_list), batch_size):
        batch = logits_list[i : i + batch_size]
        batch_naive = []
        batch_consistency = []

        for idx, logits in enumerate(batch):
            if idx % 25 == 0:  # Less frequent progress updates
                print(f"\r{desc}: {i+idx}/{len(logits_list)}", end="", flush=True)

            # Process one sample at a time
            naive_probs_sample = []
            consistency_probs_sample = []

            for logit in logits:
                # Calculate softmax once and reuse
                if compute_naive:
                    naive_prob = softmax(logit)
                    naive_probs_sample.append(naive_prob)

                # Use more memory-efficient approach for consistency
                consistency = np.zeros_like(
                    logit, dtype=np.float32
                )  # Specify dtype for memory efficiency
                for _ in range(int(n_perturbations)):
                    # Generate perturbation directly into pre-allocated array
                    perturb = np.random.normal(0, std, size=logit.shape)
                    logit_perturb = logit + perturb
                    max_index = np.argmax(logit_perturb)
                    consistency[max_index] += 1
                    # Clean up temporary arrays
                    del perturb
                    del logit_perturb

                # Normalize in-place
                consistency /= n_perturbations
                consistency_probs_sample.append(consistency)
            if compute_naive:
                batch_naive.append(naive_probs_sample)
            batch_consistency.append(consistency_probs_sample)

            # Clear sample variables explicitly
            del naive_probs_sample
            del consistency_probs_sample

        # Convert batch results to arrays and extend results
        if compute_naive:
            naive_probs.extend(batch_naive)
        consistency_probs.extend(batch_consistency)

        # Explicit cleanup
        del batch
        del batch_naive
        del batch_consistency

        # Force garbage collection
        import gc

        gc.collect()

    print(f"\r{desc}: Completed {len(logits_list)} samples", flush=True)

    # Convert to arrays at the end to minimize intermediate memory usage
    return np.array(naive_probs, dtype=np.float32), np.array(
        consistency_probs, dtype=np.float32
    )


def get_sample_tokens(p_total, num_samples):
    x = np.arange(p_total.shape[-1], dtype=np.int32)
    prediction_tokens = []

    # Process in batches
    batch_size = 500
    for i in range(0, len(p_total), batch_size):
        batch = p_total[i : i + batch_size]
        batch_tokens = []

        for p_series in batch:
            series_tokens = []
            for p_step in p_series:
                series_tokens.append(np.random.choice(x, num_samples, p=p_step))
            batch_tokens.append(series_tokens)

        prediction_tokens.extend(batch_tokens)

        # Clean up
        del batch
        del batch_tokens
        import gc

        gc.collect()

    return np.array(prediction_tokens, dtype=np.int32)


def plot_time_series(
    data_naive,
    data_consistency,
    dataset_name,
    quantile=95,
    ground_truth=None,
    max_preceding=256,
):
    """
    Plots each time series in the input data as a separate plot.

    Args:
        data_naive (np.ndarray): Array of naive forecast samples (num_series, prediction_length, num_samples).
        data_consistency (np.ndarray): Array of consistency model forecast samples (num_series, prediction_length, num_samples).
        dataset_name (str): Name of the dataset for plot titles.
        quantile (int): The quantile level for the confidence intervals (default: 95).
        ground_truth (object, optional): Object containing ground truth data. Defaults to None.
                                        Assumed to have attributes 'dataset' and 'input.test_data.dataset'.
        max_preceding (int): Maximum number of preceding timesteps to plot (default: 256).
    """
    num_series = data_naive.shape[0]
    prediction_length = data_naive.shape[1]
    timesteps = np.arange(prediction_length)

    for i in range(num_series):
        # for i in range(10):
        plt.figure(figsize=(10, 6))  # Create a new figure for each plot

        # Calculate statistics for naive and consistency data for the current series
        mean_series_naive = data_naive[i].mean(axis=1)
        mean_series_cons = data_consistency[i].mean(axis=1)
        std_series_cons = data_consistency[i].std(axis=1)

        lower_bound_naive = np.percentile(data_naive[i], (100 - quantile) / 2, axis=1)
        upper_bound_naive = np.percentile(
            data_naive[i], quantile + (100 - quantile) / 2, axis=1
        )

        lower_bound_cons = np.percentile(
            data_consistency[i], (100 - quantile) / 2, axis=1
        )
        upper_bound_cons = np.percentile(
            data_consistency[i], quantile + (100 - quantile) / 2, axis=1
        )

        if ground_truth is not None:
            if max_preceding > 0:
                preceding_series = ground_truth.input.test_data.dataset[i][
                    "target"
                ]  # Extract time series
                n = min(
                    len(preceding_series) - prediction_length, max_preceding
                )  # Determine the length dynamically
                preceding_timesteps = np.arange(
                    -n, 0
                )  # Negative indices for proper alignment

                plt.plot(
                    preceding_timesteps,
                    preceding_series[-n - prediction_length : -prediction_length],
                    label="Preceding Data",
                    color="green",
                    linestyle="dashed",
                )

            real_ts = ground_truth.dataset[i]["target"][-prediction_length:]
            plt.plot(timesteps, real_ts, label="Ground Truth", color="black")

        # Plot the main time series and shaded quantile ranges
        plt.plot(timesteps, mean_series_naive, label="Naive", color="blue")
        plt.fill_between(
            timesteps, lower_bound_naive, upper_bound_naive, color="blue", alpha=0.3
        )

        plt.plot(timesteps, mean_series_cons, label="Consistency", color="red")
        plt.fill_between(
            timesteps, lower_bound_cons, upper_bound_cons, color="red", alpha=0.3
        )

        plt.title(f"Time Series {i+1} - Dataset {dataset_name}")
        plt.xlabel("Timesteps")
        plt.ylabel("Value")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.show()


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


def load_and_split_dataset(backtest_config: dict, max_series: int):
    hf_repo = backtest_config["hf_repo"]
    dataset_name = backtest_config["name"]
    offset = backtest_config["offset"]
    prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]
    freq = backtest_config.get("freq", "B")

    # This is needed because the datasets in autogluon/chronos_datasets_extra cannot
    # be distribued due to license restrictions and must be generated on the fly
    trust_remote_code = hf_repo == "autogluon/chronos_datasets_extra"

    if hf_repo == "local":
        ds = ArrowFile(dataset_name)
        gts_dataset = [
            {"start": pd.Period(entry["start"], freq=freq), "target": entry["target"]}
            for _, entry in enumerate(ds)
        ]
        gts_dataset = list(
            filter(lambda x: len(x["target"]) > prediction_length, gts_dataset)
        )
    else:
        ds = datasets.load_dataset(
            hf_repo, dataset_name, split="train", trust_remote_code=trust_remote_code
        )
        ds.set_format("numpy")
        n_samples = ds.shape[0]
        if max_series is not None and n_samples > max_series:
            print(
                f"Sampling only {max_series} at random from the dataset (original size: {n_samples})"
            )
            sample_indices = np.random.choice(n_samples, max_series, replace=False)
            ds = ds.select(sample_indices)
            ds.info.splits["train"].num_examples = max_series

    gts_dataset = to_gluonts_univariate(ds)

    # Split dataset for evaluation
    _, test_template = split(gts_dataset, offset=offset)
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls)

    return test_data


def compute_data_for_cc(
    test_data_input: Iterable,
    pipeline: BaseChronosPipeline,
    prediction_length: int,
    batch_size: int,
    test_targets,
    **predict_kwargs,
):
    # Generate forecasts
    pipeline.tokenizer.config.prediction_length = prediction_length
    forecast_outputs = []
    correct_tokens = []
    logits = []
    predictions = []
    test_targets = test_targets.dataset
    scales = []
    for batch in tqdm(
        batcher(zip(test_data_input, test_targets), batch_size=batch_size)
    ):
        context = [torch.tensor(entry[0]["target"]) for entry in batch]
        tgt = [torch.tensor(entry[1]["target"][-prediction_length:]) for entry in batch]
        _, original_logits, scale, predicted_tokens = pipeline.predict(
            context=context,
            prediction_length=prediction_length,
            return_logits=True,
            **predict_kwargs,
        )
        if len(context) == 1:
            continue
        tgt = torch.stack(tgt, dim=0)
        # pl_pipeline = pipeline.tokenizer.config.prediction_length
        # pipeline.tokenizer.config.prediction_length = prediction_length
        correct_batch_tokens, _ = pipeline.tokenizer.label_input_transform(tgt, scale)
        correct_tokens.append(correct_batch_tokens)
        logits.append(original_logits)
        predictions.append(predicted_tokens)
        scales.append(scale)

    predictions = [pred.squeeze() for pred in predictions]
    correct_tokens = [tok[:, :-1] for tok in correct_tokens]

    scales = torch.cat(scales)
    predictions = torch.cat(predictions)
    correct_tokens = torch.cat(correct_tokens)
    logits = torch.cat(logits, axis=1)

    logits = logits.swapaxes(0, 1)

    return predictions.cpu(), correct_tokens.cpu(), logits.cpu(), scales.cpu()


def compute_probability_metrics(naive_probs, consistency_probs):
    cross_entropies = []
    for naives, consistencies in zip(naive_probs, consistency_probs):
        for naive, cons in zip(naives, consistencies):
            naive_tensor = torch.from_numpy(naive)
            cons_tensor = torch.from_numpy(cons)
            cross_entropies.append(cross_entropy(naive_tensor, cons_tensor))

    mean_cross_entropy = np.mean(cross_entropies)
    mean_abs_difference = np.mean(np.abs(naive_probs - consistency_probs))

    print("Mean cross entropy between naive and consistency: ", mean_cross_entropy)
    print(
        "Mean absolute difference between naive and consistency: ", mean_abs_difference
    )

    return mean_cross_entropy, mean_abs_difference


def generate_forecasts(
    test_data_input: Iterable,
    pipeline: BaseChronosPipeline,
    prediction_length: int,
    batch_size: int,
    **predict_kwargs,
):
    # Generate forecasts
    forecast_outputs = []
    for batch in batcher(tqdm(test_data_input, batch_size=batch_size)):
        context = [torch.tensor(entry["target"]) for entry in batch]
        forecast_outputs.append(
            pipeline.predict(
                context,
                prediction_length=prediction_length,
                **predict_kwargs,
            ).numpy()
        )
    forecast_outputs = np.concatenate(forecast_outputs)

    print("Forecast outputs shape:", forecast_outputs.shape)
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


def get_forecasts_cc(
    test_data_input, predictions, pipeline, scales
):  # predictions should be of shape (num_series, pred_length, num_samples)
    forecasts = []
    predicted_ts = []

    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)

    assert isinstance(
        predictions, torch.Tensor
    ), "The predictions array is neither a tensor nor a numpy ndarray"

    for i, samples in enumerate(predictions):
        scale = scales[i]
        predicted_ts.append(
            pipeline.tokenizer.output_transform(samples.to(scale.device), scale)
        )

    predicted_ts = torch.stack(predicted_ts, dim=0)
    forecast_outputs = torch.swapaxes(predicted_ts, 1, 2).numpy()

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
    return predicted_ts, forecasts


def group_logits_and_labels(logits, correct_tokens, group_size=10):
    num_classes = logits.shape[-1]
    num_new_classes = num_classes // group_size  # Reduce the number of classes

    # Trim logits if not divisible by group_size
    trimmed_size = num_new_classes * group_size
    logits_trimmed = logits[..., :trimmed_size]  # Remove excess classes if needed

    # Reshape and sum within groups
    grouped_logits = logits_trimmed.reshape(
        *logits.shape[:-1], num_new_classes, group_size
    ).sum(axis=-1)

    grouped_tokens = correct_tokens // group_size
    return grouped_logits, grouped_tokens


def compute_metrics(forecasts_naive, forecasts_cons, test_data, batch_size=5000):
    metrics_naive = None
    if forecasts_naive:
        metrics_naive = (
            evaluate_forecasts(
                forecasts_naive,
                test_data=test_data,
                metrics=[MASE(), MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1))],
                batch_size=batch_size,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )

    metrics_cons = (
        evaluate_forecasts(
            forecasts_cons,
            test_data=test_data,
            metrics=[
                MASE(),
                MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
            ],
            batch_size=5000,
        )
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

    return metrics_naive, metrics_cons


@app.command()
def main(
    config_path: Path,
    metrics_path_naive: Optional[Path] = None,
    metrics_path_consistency: Optional[Path] = None,
    chronos_model_id: str = "amazon/chronos-t5-small",
    std: float = 4,
    n_perturbations: int = 16,
    n_bins: int = 10,
    n_samples: int = 100,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    batch_size: int = 32,
    num_samples: int = 20,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    max_series: Optional[int] = 24_000,
    compute_naive: Optional[bool] = False,
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

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    num_cores = multiprocessing.cpu_count()
    # Load Chronos
    pipeline = BaseChronosPipeline.from_pretrained(
        chronos_model_id,
        device_map=device,
        torch_dtype=torch_dtype,
    )

    # Compute metrics paths
    if metrics_path_naive is None:
        metrics_path_naive = compute_metrics_path(
            mode="naive", model=chronos_model_id, std=std, n_pert=n_perturbations
        )
    if metrics_path_consistency is None:
        metrics_path_consistency = compute_metrics_path(
            mode="consistency", model=chronos_model_id, std=std, n_pert=n_perturbations
        )
    assert isinstance(
        pipeline, ChronosPipeline
    ), "The model needs to be of type ChronosPipeline"

    predict_kwargs = dict(
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Load backtest configs
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    result_rows_naive = []
    result_rows_cons = []
    all_naive_ece = []
    all_calibrated_ece = []
    for config in backtest_configs:
        starting_time = datetime.now()
        dataset_name = config["name"]
        prediction_length = config["prediction_length"]

        # logger.info(f"Loading {dataset_name}")
        print(f"Loading {dataset_name}")
        test_data = load_and_split_dataset(
            backtest_config=config, max_series=max_series
        )

        # logger.info(
        #     f"Generating forecasts for {dataset_name} "
        #     f"({len(test_data.input)} time series)"
        # )
        print(
            f"Generating forecasts for {dataset_name} "
            f"({len(test_data.input)} time series)"
        )

        if isinstance(pipeline, ChronosPipeline):
            predict_kwargs = dict(
                num_samples=1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        predictions, correct_tokens, logits, scales = compute_data_for_cc(
            test_data.input,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            test_targets=test_data,
            **predict_kwargs,
        )

        if len(test_data.input) > 100000:  # Adjust threshold based on your data
            num_processes = min(3, num_cores)  # Limit processes for large datasets
        else:
            num_processes = min(len(test_data.input), num_cores)
        arguments = compute_arguments(
            num_processes, logits, n_perturbations, std, compute_naive
        )
        # print("logits.shape:", logits.shape)
        with Pool(num_processes) as pool:
            results = pool.starmap(compute_probabilities, arguments)
            pool.close()
            pool.join()
            naive_probs, consistency_probs = zip(*results)
            results = None
            import gc

            gc.collect()
            if compute_naive:
                naive_probs = np.concatenate(naive_probs)
                naive_sample_tokens = get_sample_tokens(naive_probs, n_samples)
                predicted_ts_naive, forecasts_naive = get_forecasts_cc(
                    test_data.input, naive_sample_tokens, pipeline, scales
                )
                naive_probs = torch.from_numpy(naive_probs).flatten(
                    start_dim=0, end_dim=1
                )

            consistency_probs = np.concatenate(consistency_probs)
            # naive_probs, consistency_probs = compute_probabilities(logits, n_perturbations=n_perturbations, std = std)

        cons_sample_tokens = get_sample_tokens(consistency_probs, n_samples)

        predicted_ts_cons, forecasts_cons = get_forecasts_cc(
            test_data.input, cons_sample_tokens, pipeline, scales
        )

        # get_plots
        # plot_time_series(naive_sample_tokens, cons_sample_tokens, dataset_name)
        # plot_time_series(predicted_ts_naive,
        #                  predicted_ts_cons,
        #                  dataset_name,
        #                  ground_truth = test_data,
        #                  max_preceding=100)

        # reshape arrays for compatibility reasons with ECE library implementation
        consistency_probs = torch.from_numpy(consistency_probs).flatten(
            start_dim=0, end_dim=1
        )
        correct_tokens = correct_tokens.flatten(start_dim=0, end_dim=1)

        # ------ CODE FOR HISTOGRAMS ------
        # # Get max confidence per sample
        # confidences_cons = consistency_probs.max(dim=-1).values
        # confidences_naive = naive_probs.max(dim=-1).values
        # Plot histogram
        # plt.hist(confidences_cons.numpy(), bins=n_bins, range=(0, 1), alpha=0.5, color='blue', edgecolor='black', label='Consistency')
        # plt.hist(confidences_naive.numpy(), bins=n_bins, range=(0, 1), alpha=0.5, color='orange', edgecolor='black', label='Naive')

        # plt.xlabel("Confidence")
        # plt.ylabel("Count")
        # plt.title(f"Distribution of Consistency Probabilities std={std}")
        # plt.legend()
        # plt.show()

        ECE = MulticlassCalibrationError(
            num_classes=consistency_probs.shape[-1], n_bins=n_bins
        )
        if compute_naive:
            ece_naive = ECE(naive_probs, correct_tokens)
            all_naive_ece.append(ece_naive)
            print("Naive probs ECE: ", ece_naive)

        ece_consistency = ECE(consistency_probs, correct_tokens)
        all_calibrated_ece.append(ece_consistency)

        print("Consistency probs ECE: ", ece_consistency)

        # logger.info(f"Evaluating forecasts for {dataset_name}")
        print(f"Evaluating forecasts for {dataset_name}")

        if not compute_naive:
            forecasts_naive = None
        metrics_naive, metrics_cons = compute_metrics(
            forecasts_naive, forecasts_cons, test_data
        )

        print("Naive metrics: ", metrics_naive)
        print("Cons metrics: ", metrics_cons)

        if compute_naive:
            metrics_naive_dict = metrics_naive[0]
            metrics_naive_dict["ECE"] = ece_naive.item()

        metrics_cons_dict = metrics_cons[0]
        metrics_cons_dict["ECE"] = ece_consistency.item()

        if compute_naive:
            result_rows_naive.append(
                {
                    "dataset": dataset_name,
                    "model": chronos_model_id,
                    **metrics_naive_dict,
                }
            )
        result_rows_cons.append(
            {"dataset": dataset_name, "model": chronos_model_id, **metrics_cons_dict}
        )
        elapsed_time = datetime.now() - starting_time
        print("Elapsed time: ", elapsed_time.total_seconds())
        print("Time/series: ", elapsed_time.total_seconds() / len(test_data.input))
        os.makedirs("Results", exist_ok=True)
        # Save results to CSV files
        if compute_naive:
            results_df_naive = (
                pd.DataFrame(result_rows_naive)
                .rename(
                    {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
                    axis="columns",
                )
                .sort_values(by="dataset")
            )
            results_df_naive.to_csv(metrics_path_naive, index=False)

        results_df_cons = (
            pd.DataFrame(result_rows_cons)
            .rename(
                {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
                axis="columns",
            )
            .sort_values(by="dataset")
        )
        results_df_cons.to_csv(metrics_path_consistency, index=False)

    print("all naive probs:", all_naive_ece)
    print("all calibrated ece", all_calibrated_ece)
    print("naive mean", np.mean(all_naive_ece))
    print("calibrated mean", np.mean(all_calibrated_ece))

    os.makedirs("Results", exist_ok=True)
    # Save results to CSV files
    if compute_naive:
        results_df_naive = (
            pd.DataFrame(result_rows_naive)
            .rename(
                {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
                axis="columns",
            )
            .sort_values(by="dataset")
        )
        results_df_naive.to_csv(metrics_path_naive, index=False)

    results_df_cons = (
        pd.DataFrame(result_rows_cons)
        .rename(
            {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
            axis="columns",
        )
        .sort_values(by="dataset")
    )
    results_df_cons.to_csv(metrics_path_consistency, index=False)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Evaluation")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    app()
