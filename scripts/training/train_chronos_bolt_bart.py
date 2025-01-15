import logging
import os
from pathlib import Path
import yaml
import torch
from transformers import Trainer, TrainingArguments
from chronos.chronos_bolt_bart import ChronosBartForForecasting
from train import ChronosDataset
from train_chronos_bolt_bart import ChronosBartConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load config
    with open("chronos_bolt_bart_config.yaml") as f:
        config = yaml.safe_load(f)

    # Create model config
    model_config = ChronosBartConfig(
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        input_patch_size=config["input_patch_size"],
        input_patch_stride=config["input_patch_stride"],
        quantiles=config["quantiles"],
        use_reg_token=config["use_reg_token"]
    )

    # Initialize model
    model = ChronosBartForForecasting(model_config)

    # Create dataset
    train_dataset = ChronosDataset(
        data_paths=config["training_data_paths"],
        probabilities=config["probability"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        min_past=config["min_past"],
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        save_steps=config["save_steps"],
        logging_steps=config["log_steps"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        tf32=config["tf32"],
        torch_compile=config["torch_compile"],
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train
    trainer.train()

if __name__ == "__main__":
    main()