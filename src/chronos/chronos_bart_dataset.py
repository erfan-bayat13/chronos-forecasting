import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from typing import Iterator, List, Dict, Any, Optional
import itertools
from gluonts.transform import (
    FilterTransformation,
    TestSplitSampler, 
    ValidationSplitSampler,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
)
from gluonts.itertools import Cyclic

class Patch(nn.Module):
    """Time series patching layer"""
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        
        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(
                size=padding_size, 
                fill_value=torch.nan,
                dtype=x.dtype,
                device=x.device
            )
            x = torch.concat((padding, x), dim=-1)
            
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x

class InstanceNorm(nn.Module):
    """Instance normalization layer"""
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num(
                torch.nanmean((x - loc).square(), dim=-1, keepdim=True).sqrt(), 
                nan=1.0
            )
            scale = torch.where(scale == 0, torch.abs(loc) + self.eps, scale)
        else:
            loc, scale = loc_scale

        return (x - loc) / scale, (loc, scale)

    def inverse(
        self, 
        x: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        loc, scale = loc_scale
        return x * scale + loc


class ChronosBartDataset(IterableDataset):
    def __init__(
        self,
        datasets: list,
        probabilities: List[float],
        context_length: int = 512,
        prediction_length: int = 64,
        input_patch_size: int = 16,
        input_patch_stride: int = 8,
        min_past: Optional[int] = None,
        mode: str = "training",
        np_dtype=np.float32,
    ) -> None:
        super().__init__()

        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")

        self.datasets = datasets
        self.probabilities = probabilities
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.min_past = min_past or prediction_length
        self.mode = mode
        self.np_dtype = np_dtype
        
        # Initialize patching and normalization layers
        self.patch = Patch(input_patch_size, input_patch_stride)
        self.instance_norm = InstanceNorm()

    def preprocess_entry(self, entry: Dict[str, Any], mode: str) -> Dict[str, Any]:
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        assert entry["target"].ndim == 1, f"got {entry['target'].ndim=}, expected 1"
        return entry

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "test", "validation"]

        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    def create_training_data(self, data):
        data = Cyclic(data)
        split_transform = self._create_instance_splitter(
            "training"
        ) + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        data = split_transform.apply(data, is_train=True)
        return data

    def create_test_data(self, data):
        data = self._create_instance_splitter("test").apply(data, is_train=False)
        return data

    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_bart_format(self, entry: Dict) -> Dict:
        # Convert to tensor and add batch dimension
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        attention_mask = ~torch.isnan(past_target)

        # Apply instance normalization
        past_target, (loc, scale) = self.instance_norm(past_target)
        
        # Apply patching
        patched_context = self.patch(past_target)
        patched_mask = torch.nan_to_num(self.patch(attention_mask.float()), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        
        # Concatenate context and mask
        input_embeds = torch.cat([patched_context, patched_mask], dim=-1)
        
        # Create attention mask - 1 if at least one item in patch is observed
        attention_mask = (patched_mask.sum(dim=-1) > 0)

        # Get future target and normalize
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        future_target, _ = self.instance_norm(future_target, (loc, scale))
        future_target_mask = ~torch.isnan(future_target)
        future_target[~future_target_mask] = 0.0

        return {
            "input_embeds": input_embeds.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "target": future_target.squeeze(0), 
            "target_mask": future_target_mask.squeeze(0)
        }

    def __iter__(self) -> Iterator:
        preprocessed_datasets = [
            itertools.starmap(
                lambda i, x: self.preprocess_entry(x, mode=self.mode),
                enumerate(dataset),
            )
            for dataset in self.datasets
        ]

        if self.mode == "training":
            iterables = [
                self.create_training_data(dataset) for dataset in preprocessed_datasets
            ]
        elif self.mode == "test":
            iterables = [
                self.create_test_data(dataset) for dataset in preprocessed_datasets
            ]
        else:
            iterables = [
                self.create_validation_data(dataset)
                for dataset in preprocessed_datasets
            ]

        iterators = list(map(iter, iterables))
        probs = list(self.probabilities)

        if self.mode == "training":
            while True:
                idx = np.random.choice(range(len(iterators)), p=probs)
                try:
                    yield self.to_bart_format(next(iterators[idx]))
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
        else:
            for entry in itertools.chain(*iterators):
                yield self.to_bart_format(entry)