import logging
from pathlib import Path
from typing import Iterable, Optional
from gluonts.dataset.arrow import ArrowFile

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

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def compute_probabilities(logits_list, n_perturbations=10, epsilon = 0.1):
    naive_probs = []
    consistency_probs = []
    for logits in tqdm(logits_list):
      naive_probs_per_logit = []
      consistency_probs_per_logit = []
      for logit in logits:
          naive_probs_per_logit.append(softmax(logit))
          consistency = np.zeros_like(logit) 
          for i in range(n_perturbations):
              logit_perturb = logit + np.random.normal(0, epsilon, size = logit.shape)
              max_index = np.argmax(logit_perturb)
              consistency[max_index.item()] += 1
          consistency /= n_perturbations
          consistency_probs_per_logit.append(consistency)
      naive_probs.append(naive_probs_per_logit)
      consistency_probs.append(consistency_probs_per_logit)
            
    return np.array(naive_probs), np.array(consistency_probs)


def ECE(predictions, correct_tokens, probs, n_bins=10):
    correct_predictions_mask = predictions==correct_tokens
    correct_predictions = predictions[correct_predictions_mask]
    correct_probs = probs[correct_predictions_mask]
    epsilon = 1e-9
    bins = np.linspace(0,1+epsilon,n_bins+1)
    bin_item_count = np.zeros((n_bins,))
    bin_accuracies = np.zeros((n_bins,))
    bin_confidences = np.zeros((n_bins,))
    for i in range(1,n_bins+1):
        bin_accuracy = 0
        probs_bin_map = np.logical_and(probs < bins[i], probs >= bins[i-1])
        n_bin_predictions = probs_bin_map.sum()
        bin_item_count[i-1] = n_bin_predictions
        print(n_bin_predictions)
        for j,pred in enumerate(correct_predictions):
            if correct_probs[j,pred]<bins[i] and correct_probs[j,pred]>=bins[i-1]:
                bin_accuracy += 1
        bin_accuracy = bin_accuracy / (n_bin_predictions + epsilon)
        bin_confidence = probs[probs_bin_map]/probs_bin_map.sum()
        bin_confidence = bin_confidence.sum()
        bin_accuracies[i-1] = bin_accuracy
        bin_confidences[i-1] = bin_confidence

    ece = np.abs(bin_accuracies - bin_confidences)*bin_item_count
    ece = ece.sum()/predictions.numel()
return ece


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
    freq = backtest_config.get('freq', 'B')

    # This is needed because the datasets in autogluon/chronos_datasets_extra cannot
    # be distribued due to license restrictions and must be generated on the fly
    trust_remote_code = True if hf_repo == "autogluon/chronos_datasets_extra" else False

    if hf_repo == 'local':
        ds = ArrowFile(dataset_name)
        gts_dataset = [{'start': pd.Period(entry['start'], freq=freq), 'target': entry['target']} for _, entry in enumerate(ds)]
        gts_dataset = list(filter(lambda x: len(x['target'])>prediction_length, gts_dataset))
    else:
        ds = datasets.load_dataset(
            hf_repo, dataset_name, split="train", trust_remote_code=trust_remote_code
        )
        ds.set_format("numpy")

        gts_dataset = to_gluonts_univariate(ds)

    # Split dataset for evaluation
    _, test_template = split(gts_dataset, offset=offset)
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls)

    return test_data


def compute_data_for_cc( 
    test_data_input: Iterable,
    pipeline: BaseChronosPipeline,
    prediction_length:int,
    batch_size: int,
    test_targets,
    **predict_kwargs
    ):
    # Generate forecasts
    pipeline.tokenizer.config.prediction_length = prediction_length
    forecast_outputs = []
    correct_tokens = []
    logits = []
    predictions = []
    test_targets = test_targets.dataset
    for batch in tqdm(batcher(zip(test_data_input, test_targets) , batch_size=batch_size)):
        context = [torch.tensor(entry[0]["target"]) for entry in batch]
        tgt = [torch.tensor(entry[1]["target"][-prediction_length:]) for entry in batch]
        _, original_logits, scale, predicted_tokens = pipeline.predict(
            context=context,
            prediction_length=prediction_length,
            return_logits=True,
            **predict_kwargs
        )
        tgt = torch.stack(tgt, dim=0)
        # pl_pipeline = pipeline.tokenizer.config.prediction_length
        # pipeline.tokenizer.config.prediction_length = prediction_length
        correct_batch_tokens, _ = pipeline.tokenizer.label_input_transform(tgt, scale)
        correct_tokens.append(correct_batch_tokens)
        logits.append(original_logits)
        predictions.append(predicted_tokens)
    
    predictions = [pred.squeeze() for pred in predictions]
    correct_tokens = [tok[:,:-1] for tok in correct_tokens]

    predictions=torch.cat(predictions)
    correct_tokens=torch.cat(correct_tokens)
    logits=torch.cat(logits, axis=1)

    logits = logits.swapaxes(0,1)

    return predictions, correct_tokens, logits


def compute_probability_metrics(naive_probs, consistency_probs):
    cross_entropies = []
    for naives, consistencies in zip(naive_probs, consistency_probs):
        for naive, cons in zip(naives, consistencies):
            naive_tensor = torch.from_numpy(naive)
            cons_tensor = torch.from_numpy(cons)
            cross_entropies.append(cross_entropy(naive_tensor, cons_tensor)) 

    mean_cross_entropy = np.mean(cross_entropies)
    mean_abs_difference = np.mean(np.abs(naive_probs-consistency_probs))

    print("Mean cross entropy between naive and consistency: ", mean_cross_entropy)
    print("Mean absolute difference between naive and consistency: ", mean_abs_difference)

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
    for batch in tqdm(batcher(test_data_input , batch_size=batch_size)):
        context = [torch.tensor(entry["target"]) for entry in batch]
        forecast_outputs.append(
            pipeline.predict(
                context,
                prediction_length=prediction_length,
                **predict_kwargs,
            ).numpy()
        )
    forecast_outputs = np.concatenate(forecast_outputs)

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
        test_data = load_and_split_dataset(backtest_config=config)

        logger.info(
            f"Generating forecasts for {dataset_name} "
            f"({len(test_data.input)} time series)"
        )
        forecasts = generate_forecasts(
            test_data.input,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            **predict_kwargs,
        )
        if isinstance(pipeline, ChronosPipeline):
            predict_kwargs = dict(
                num_samples=1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        predictions, correct_tokens, logits = compute_data_for_cc(test_data.input,
                                                                  pipeline=pipeline,
                                                                  prediction_length=prediction_length,
                                                                  batch_size=batch_size,
                                                                  test_targets=test_data,
                                                                  **predict_kwargs)
        naive_probs, consistency_probs = compute_probabilities(logits)
        compute_probability_metrics(naive_probs, consistency_probs)
        ece_naive = ECE(predictions, correct_tokens, naive_probs, n_bins=10)
        ece_consistency = ECE(predictions, correct_tokens, consistency_probs, n_bins=10)

        print("Naive probs ECE: ", ece_naive)
        print("Consistency probs ECE: ", ece_consistency)


        logger.info(f"Evaluating forecasts for {dataset_name}")
        metrics = (
            evaluate_forecasts(
                forecasts,
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
        result_rows.append(
            {"dataset": dataset_name, "model": chronos_model_id, **metrics[0]}
        )

    # Save results to a CSV file
    results_df = (
        pd.DataFrame(result_rows)
        .rename(
            {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
            axis="columns",
        )
        .sort_values(by="dataset")
    )
    results_df.to_csv(metrics_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)
    app()

