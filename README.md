<div align="center">
<img src="figures/chronos-logo.png" width="60%">
</div>


<div align="center">

# Chronos: Learning the Language of Time Series

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2403.07815&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2403.07815)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-FFD21E)](https://huggingface.co/datasets/autogluon/chronos_datasets)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444)
[![faq](https://img.shields.io/badge/FAQ-Questions%3F-blue)](https://github.com/amazon-science/chronos-forecasting/issues?q=is%3Aissue+label%3AFAQ)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)



## ✨ Introduction

Chronos is a family of **pretrained time series forecasting models** based on language model architectures. A time series is transformed into a sequence of tokens via scaling and quantization, and a language model is trained on these tokens using the cross-entropy loss. Once trained, probabilistic forecasts are obtained by sampling multiple future trajectories given the historical context. Chronos models have been trained on a large corpus of publicly available time series data, as well as synthetic data generated using Gaussian processes.

For details on Chronos models, training data and procedures, and experimental results, please refer to the paper [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815).

<p align="center">
  <img src="figures/main-figure.png" width="100%">
  <br />
  <span>
    Fig. 1: High-level depiction of Chronos. (<b>Left</b>) The input time series is scaled and quantized to obtain a sequence of tokens. (<b>Center</b>) The tokens are fed into a language model which may either be an encoder-decoder or a decoder-only model. The model is trained using the cross-entropy loss. (<b>Right</b>) During inference, we autoregressively sample tokens from the model and map them back to numerical values. Multiple trajectories are sampled to obtain a predictive distribution.
  </span>
</p>

### Architecture

The models in this repository are based on the [T5 architecture](https://arxiv.org/abs/1910.10683). The only difference is in the vocabulary size: Chronos-T5 models use 4096 different tokens, compared to 32128 of the original T5 models, resulting in fewer parameters.

<div align="center">

| Model                                                                  | Parameters | Based on                                                               |
| ---------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------- |
| [**chronos-t5-tiny**](https://huggingface.co/amazon/chronos-t5-tiny)   | 8M         | [t5-efficient-tiny](https://huggingface.co/google/t5-efficient-tiny)   |
| [**chronos-t5-mini**](https://huggingface.co/amazon/chronos-t5-mini)   | 20M        | [t5-efficient-mini](https://huggingface.co/google/t5-efficient-mini)   |
| [**chronos-t5-small**](https://huggingface.co/amazon/chronos-t5-small) | 46M        | [t5-efficient-small](https://huggingface.co/google/t5-efficient-small) |
| [**chronos-t5-base**](https://huggingface.co/amazon/chronos-t5-base)   | 200M       | [t5-efficient-base](https://huggingface.co/google/t5-efficient-base)   |
| [**chronos-t5-large**](https://huggingface.co/amazon/chronos-t5-large) | 710M       | [t5-efficient-large](https://huggingface.co/google/t5-efficient-large) |
| [**chronos-bolt-tiny**](https://huggingface.co/amazon/chronos-bolt-tiny)   | 9M         | [t5-efficient-tiny](https://huggingface.co/google/t5-efficient-tiny)   |
| [**chronos-bolt-mini**](https://huggingface.co/amazon/chronos-bolt-mini)   | 21M        | [t5-efficient-mini](https://huggingface.co/google/t5-efficient-mini)   |
| [**chronos-bolt-small**](https://huggingface.co/amazon/chronos-bolt-small) | 48M        | [t5-efficient-small](https://huggingface.co/google/t5-efficient-small) |
| [**chronos-bolt-base**](https://huggingface.co/amazon/chronos-bolt-base)   | 205M       | [t5-efficient-base](https://huggingface.co/google/t5-efficient-base)   |

</div>

### Zero-Shot Results

The following figure showcases the remarkable **zero-shot** performance of Chronos and Chronos-Bolt models on 27 datasets against local models, task-specific models and other pretrained models. For details on the evaluation setup and other results, please refer to [the paper](https://arxiv.org/abs/2403.07815). 

<p align="center">
  <img src="figures/zero_shot-agg_scaled_score.svg" width="100%">
  <br />
  <span>
    Fig. 2: Performance of different models on Benchmark II, comprising 27 datasets <b>not seen</b> by Chronos and Chronos-Bolt models during training. This benchmark provides insights into the zero-shot performance of Chronos and Chronos-Bolt models against local statistical models, which fit parameters individually for each time series, task-specific models <i>trained on each task</i>, and pretrained models trained on a large corpus of time series. Pretrained Models (Other) indicates that some (or all) of the datasets in Benchmark II may have been in the training corpus of these models. The probabilistic (WQL) and point (MASE) forecasting metrics were normalized using the scores of the Seasonal Naive baseline and aggregated through a geometric mean to obtain the Agg. Relative WQL and MASE, respectively.
  </span>
</p>

# Extending Chronos: Fine-Tuning and Consistency Calibration

This repository contains extensions to the Chronos time series forecasting model. Two main extensions are provided:

1. [TODO] Fine-tuning Chronos for financial forecasting using Geometric Brownian Motion (GBM) for synthetic data
2. Consistency Calibration for Chronos (C3) - A method to improve forecast reliability through input perturbation and aggregation

## C3: Consistency Calibration for Chronos

C3 improves the reliability of Chronos forecasts by applying controlled perturbations to the input data and aggregating multiple predictions. This helps quantify model uncertainty and produces more robust forecasts.

### Features

- Multiple perturbation strategies (additive and multiplicative noise)
- Support for both Gaussian and uniform noise distributions
- Configurable noise strength and number of perturbations
- Comprehensive evaluation metrics for calibration quality
- Visualization tools for comparing original and calibrated forecasts

### Installation

```bash
# Clone the repository
git clone [https://github.com/erfan-bayat13/chronos-forecasting]

# Install in editable mode with extra training-related dependencies
cd chronos-forecasting && pip install --editable ".[training]"
```

### Usage

The main script for running C3 is `c3.py`. Here's an example of how to use it:

```bash
python c3.py \
    --model_name "amazon/chronos-t5-small" \
    --data_path "path/to/your/data.csv" \
    --target_column "value" \
    --prediction_length 14 \
    --num_samples 20 \
    --num_perturbations "8,16,32,64" \
    --noise_dist "gaussian,uniform" \
    --noise_type "multiplicative,additive" \
    --noise_strength "0.01,0.05,0.1" \
    --output_dir "./c3_output"
```

### Parameters

- **Model Parameters:**
  - `--model_name`: Name or path of the Chronos model (default: "amazon/chronos-t5-small")
  - `--use_bfloat16`: Use bfloat16 precision (flag)

- **Data Parameters:**
  - `--data_path`: Path to CSV data file (required)
  - `--target_column`: Name of target column in CSV (required)

- **Prediction Parameters:**
  - `--prediction_length`: Number of steps to predict (default: 12)
  - `--num_samples`: Number of samples per prediction (default: 20)
  - `--num_perturbations`: Comma-separated list of perturbation counts to test (default: "8,16,32,64")

- **Noise Parameters:**
  - `--noise_dist`: Comma-separated list of noise distributions (default: "gaussian,uniform")
  - `--noise_type`: Comma-separated list of noise types (default: "multiplicative,additive")
  - `--noise_strength`: Comma-separated list of noise strengths (default: "0.01,0.05,0.1")

- **Output Parameters:**
  - `--output_dir`: Directory for output files (default: "./c3_output")

### Output Structure

The script creates the following output structure:

```
c3_output/
├── config.json                    # Experiment configuration
├── results.json                   # Detailed results
├── calibration_metrics.csv        # Summary metrics
├── plots/
│   ├── coverage_improvement_vs_perturbations.png
│   ├── width_reduction_vs_perturbations.png
│   ├── error_reduction_vs_perturbations.png
│   └── [perturbation_count]/     # Individual experiment plots
└── calibration/
    └── [perturbation_count]/     # Calibration metrics by configuration
```

### Calibration Configurations

Three preset configurations are provided:

1. Conservative:
   - Gaussian distribution
   - Additive noise
   - Noise strength: 0.1
   - 16 perturbations

2. Moderate:
   - Gaussian distribution
   - Additive noise
   - Noise strength: 0.3
   - 32 perturbations

3. Aggressive:
   - Uniform distribution
   - Multiplicative noise
   - Noise strength: 0.01
   - 64 perturbations

### Evaluation Metrics

The calibration quality is evaluated using several metrics:

- Coverage rate: Percentage of true values falling within prediction intervals
- Interval width: Average width of prediction intervals
- Prediction error: Median absolute error of point forecasts
- Consistency score: Standard deviation of predictions across perturbations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

## Acknowledgments

This work builds upon the Chronos model developed by Amazon. The implementation draws inspiration from various consistency calibration techniques in machine learning literature.
