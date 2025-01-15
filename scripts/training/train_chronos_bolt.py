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
    T5Config,
    Trainer,
    TrainingArguments,
)
from gluonts.dataset.common import FileDataset

from chronos.chronos_bolt import ChronosBoltModelForForecasting
from train import ChronosDataset, Filter, has_enough_observations

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
@use_yaml_config(param_name="config")
def main(
    config_path: str,
):
    # Load config
    config = yaml.safe_load(open(config_path))
    
    # Initialize T5 config
    t5_config = T5Config.from_pretrained(config["model_id"])
    
    # Add Chronos-Bolt specific config
    t5_config.chronos_config = {
        "context_length": config["context_length"],
        "prediction_length": config["prediction_length"],
        "input_patch_size": config["input_patch_size"],
        "input_patch_stride": config["input_patch_stride"],
        "quantiles": config["quantiles"],
        "use_reg_token": config["use_reg_token"]
    }

    # Initialize Chronos-Bolt model with T5 base
    model = ChronosBoltModelForForecasting(t5_config)
    if not config["random_init"]:
        # Load pretrained T5 weights
        model._init_weights(model)
        model.tie_weights()

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

    # Create training dataset
    train_dataset = ChronosDataset(
        datasets=train_datasets,
        probabilities=config["probability"],
        tokenizer=config.create_tokenizer(),
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        min_past=config["min_past"],
        model_type="seq2seq",  # Bolt always uses seq2seq architecture
        mode="training",
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        save_steps=config["save_steps"],
        logging_steps=config["log_steps"],
        tf32=config["tf32"],
        torch_compile=config["torch_compile"],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train
    trainer.train()

if __name__ == "__main__":
    app()