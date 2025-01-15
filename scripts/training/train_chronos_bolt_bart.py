import logging
import os
from pathlib import Path
import yaml
import typer
import torch
import torch.distributed as dist
from transformers import (
    BartConfig,
    Trainer,
    TrainingArguments,
)
from gluonts.dataset.common import FileDataset
from chronos.chronos_bolt_bart import ChronosBartForForecasting
from chronos.chronos_bart_dataset import ChronosBartDataset # Import new dataset class
from train import Filter, has_enough_observations, get_next_path

app = typer.Typer(pretty_exceptions_enable=False)

def is_main_process() -> bool:
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0

def log_on_main(msg: str, logger: logging.Logger, log_level: int = logging.INFO):
    if is_main_process():
        logger.log(log_level, msg)

@app.command()
def main(config_path: str):
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    log_on_main(f"Loaded config from {config_path}", logger)

    # Initialize BART config
    bart_config = BartConfig.from_pretrained(config["model_id"])
    
    # Add Chronos-BART specific config
    bart_config.chronos_config = {
        "context_length": config["context_length"],
        "prediction_length": config["prediction_length"],
        "input_patch_size": config["input_patch_size"],
        "input_patch_stride": config["input_patch_stride"],
        "quantiles": config["quantiles"],
        "use_reg_token": config["use_reg_token"]
    }

    # Get output directory
    output_dir = get_next_path("run", base_dir=Path(config["output_dir"]), file_type="")
    log_on_main(f"Logging dir: {output_dir}", logger)

    # Initialize Chronos-BART model
    model = ChronosBartForForecasting(bart_config)
    if not config["random_init"]:
        log_on_main("Using pretrained initialization", logger)
    else:
        log_on_main("Using random initialization", logger)
        model._init_weights(model)

    # Create datasets
    train_datasets = [
        Filter(
            lambda x: has_enough_observations(
                x,
                min_length=config["min_past"] + config["prediction_length"],
                max_missing_prop=config["max_missing_prop"],
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in config["training_data_paths"]
    ]

    # Create training dataset using new dataset class
    train_dataset = ChronosBartDataset(
        datasets=train_datasets,
        probabilities=config["probability"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        input_patch_size=config["input_patch_size"],
        input_patch_stride=config["input_patch_stride"],
        min_past=config["min_past"],
        mode="training"
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        save_steps=config["save_steps"],
        logging_steps=config["log_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        save_strategy="steps",
        report_to=["tensorboard"],
        tf32=config["tf32"],
        torch_compile=config["torch_compile"],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=config["dataloader_num_workers"]
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