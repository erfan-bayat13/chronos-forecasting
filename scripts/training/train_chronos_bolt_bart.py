import logging
import os
from pathlib import Path
import yaml
import typer
from typer_config import use_yaml_config
import torch
import torch.distributed as dist
from transformers import (
    AutoConfig,
    BartConfig,
    Trainer,
    TrainingArguments,
)
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic

from chronos.chronos_bolt_bart import ChronosBartForForecasting
from train import ChronosDataset, Filter, has_enough_observations, get_next_path

app = typer.Typer(pretty_exceptions_enable=False)

def is_main_process() -> bool:
    """Check if we're on the main process."""
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0

def log_on_main(msg: str, logger: logging.Logger, log_level: int = logging.INFO):
    """Log message only on main process."""
    if is_main_process():
        logger.log(log_level, msg)

@app.command()
@use_yaml_config(param_name="config")
def main(
    # Data paths and mixing probabilities
    training_data_paths: str,
    probability: str = None,
    
    # Model configuration
    context_length: int = 512,
    prediction_length: int = 64,
    input_patch_size: int = 16,
    input_patch_stride: int = 8,
    quantiles: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    use_reg_token: bool = False,
    
    # Training parameters
    min_past: int = 64,
    max_steps: int = 200_000,
    save_steps: int = 50_000,
    log_steps: int = 500,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 1e-3,
    gradient_accumulation_steps: int = 2,
    
    # Model architecture
    model_id: str = "facebook/bart-base",
    random_init: bool = True,
    output_dir: str = "./output/",
    
    # Hardware settings
    tf32: bool = True,
    torch_compile: bool = True,
    dataloader_num_workers: int = 1,
    max_missing_prop: float = 0.9,
    
    # Optional parameters
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
):
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Process data paths
    import ast
    training_data_paths = ast.literal_eval(training_data_paths)
    if probability is None:
        probability = [1.0 / len(training_data_paths)] * len(training_data_paths)
    else:
        probability = ast.literal_eval(probability)

    # Initialize BART config
    bart_config = BartConfig.from_pretrained(model_id)
    
    # Add Chronos-BART specific config
    bart_config.chronos_config = {
        "context_length": context_length,
        "prediction_length": prediction_length,
        "input_patch_size": input_patch_size,
        "input_patch_stride": input_patch_stride,
        "quantiles": quantiles,
        "use_reg_token": use_reg_token
    }

    # Get output directory
    output_dir = get_next_path("run", base_dir=Path(output_dir), file_type="")
    log_on_main(f"Logging dir: {output_dir}", logger)

    # Initialize Chronos-BART model
    model = ChronosBartForForecasting(bart_config)
    if not random_init:
        log_on_main("Using pretrained initialization", logger)
    else:
        log_on_main("Using random initialization", logger)
        model._init_weights(model)

    # Create datasets
    train_datasets = [
        Filter(
            lambda x: has_enough_observations(
                x,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in training_data_paths
    ]

    # Create training dataset
    train_dataset = ChronosDataset(
        datasets=train_datasets,
        probabilities=probability,
        tokenizer=None,  # BART doesn't use external tokenizer
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        model_type="seq2seq",  # BART is encoder-decoder
        mode="training",
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=log_steps,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        save_strategy="steps",
        report_to=["tensorboard"],
        tf32=tf32,
        torch_compile=torch_compile,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=dataloader_num_workers
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args, 
        train_dataset=train_dataset,
    )

    # Train
    log_on_main("Starting training", logger)
    trainer.train()

    # Save final model
    if is_main_process():
        model.save_pretrained(output_dir / "checkpoint-final")


if __name__ == "__main__":
    app()