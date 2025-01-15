import copy
import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    BartConfig, 
    BartPreTrainedModel,
    BartModel
)
from transformers.models.bart.modeling_bart import (
    BartEncoder, 
    BartDecoder,
    shift_tokens_right
)
from transformers.utils import ModelOutput

from chronos.base import BaseChronosPipeline, ForecastType

logger = logging.getLogger(__name__)

@dataclass 
class ChronosBartConfig:
    """Configuration for ChronosBolt BART model"""
    context_length: int
    prediction_length: int
    input_patch_size: int  
    input_patch_stride: int
    quantiles: List[float]
    use_reg_token: bool = False
    
@dataclass
class ChronosBartOutput(ModelOutput):
    """Output type for ChronosBolt BART model"""
    loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None 
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class Patch(nn.Module):
    """Convert time series into patches"""
    def __init__(self, patch_size: int, patch_stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        
        if length % self.patch_size != 0:
            padding_size = (*x.shape[:-1], self.patch_size - (length % self.patch_size))
            padding = torch.full(
                size=padding_size,
                fill_value=torch.nan,
                dtype=x.dtype,
                device=x.device
            )
            x = torch.concat((padding, x), dim=-1)
            
        return x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)

class InstanceNorm(nn.Module):
    """Instance normalization for time series data"""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(
        self, 
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
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
        loc_scale: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        loc, scale = loc_scale
        return x * scale + loc

class ResidualBlock(nn.Module):
    """Residual block for patch embeddings"""
    def __init__(
        self,
        in_dim: int,
        h_dim: int, 
        out_dim: int,
        dropout_p: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_dim)
            
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual_layer(x)
        
        out = self.hidden_layer(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.output_layer(out)
        out = out + identity
        
        if self.use_layer_norm:
            out = self.layer_norm(out)
            
        return out
    
class ChronosBartForForecasting(BartPreTrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"
        self.chronos_config = ChronosBartConfig(**config.chronos_config)
        
        # Initialize BART
        self.model = BartModel(config)
        
        # Patch embedding
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride
        )
        
        # Instance normalization
        self.instance_norm = InstanceNorm()
        
        # Input embedding
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.chronos_config.input_patch_size * 2,
            h_dim=config.encoder_ffn_dim,
            out_dim=config.d_model,
            dropout_p=config.dropout
        )
        
        # Quantile prediction head
        self.num_quantiles = len(self.chronos_config.quantiles)
        self.quantiles = nn.Parameter(
            torch.tensor(self.chronos_config.quantiles),
            requires_grad=False
        )
        
        # Output projection
        self.output_projection = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.decoder_ffn_dim,
            out_dim=self.num_quantiles * self.chronos_config.prediction_length,
            dropout_p=config.dropout
        )
        
        # Initialize weights
        self.post_init()
        
    def get_encoder(self):
        return self.model.get_encoder()
        
    def get_decoder(self):
        return self.model.get_decoder()
        
    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ChronosBartOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Create attention mask for context
        mask = (
            mask.to(context.dtype) 
            if mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )
        
        batch_size = context.shape[0]
        
        # Handle context length
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length:]
            mask = mask[..., -self.chronos_config.context_length:]
            
        # Scale input
        context, loc_scale = self.instance_norm(context)
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)
        
        # Create patches
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0) 
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        
        # Concatenate context and mask
        patched_input = torch.cat([patched_context, patched_mask], dim=-1)
        
        # Create attention mask for patches
        attention_mask = (patched_mask.sum(dim=-1) > 0).float()
        
        # Get input embeddings
        input_embeds = self.input_patch_embedding(patched_input)
        
        # Add REG token if configured
        if self.chronos_config.use_reg_token:
            reg_embeds = self.model.shared(
                torch.full((batch_size, 1), 1, device=input_embeds.device)
            )
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(attention_mask[:,:1])],
                dim=1
            )
            
        # Create decoder input ids
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=input_embeds.device
        )
        
        # Pass through BART
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # Get predictions from last hidden state
        sequence_output = outputs.last_hidden_state[:,-1:]
        quantile_preds = self.output_projection(sequence_output)
        quantile_preds = quantile_preds.view(
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length
        )
        
        # Calculate loss if target provided
        loss = None
        if target is not None:
            # Scale target
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)
            
            # Create target mask
            target_mask = (
                target_mask.unsqueeze(1).to(target.device)
                if target_mask is not None
                else ~torch.isnan(target) 
            )
            target[~target_mask] = 0.0
            
            # Pad if needed
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1]
                )
                target = torch.cat(
                    [target, torch.zeros(padding_shape).to(target)],
                    dim=-1
                )
                target_mask = torch.cat(
                    [target_mask, torch.zeros(padding_shape).to(target_mask)],
                    dim=-1
                )
                
            # Compute quantile loss
            loss = 2 * torch.abs(
                (target - quantile_preds) *
                (
                    (target <= quantile_preds).float() - 
                    self.quantiles.view(1, -1, 1)
                )
            ) * target_mask.float()
            
            loss = loss.mean(dim=-2)  # Mean over prediction horizon
            loss = loss.sum(dim=-1)   # Sum over quantiles
            loss = loss.mean()        # Mean over batch
            
        # Unscale predictions
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale
        ).view(
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length
        )
        
        if not return_dict:
            output = (quantile_preds,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        return ChronosBartOutput(
            loss=loss,
            quantile_preds=quantile_preds,
            encoder_attentions=outputs.encoder_attentions,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_hidden_states=outputs.encoder_hidden_states,
            decoder_hidden_states=outputs.decoder_hidden_states
        )
        
class ChronosBartPipeline(BaseChronosPipeline):
    """Pipeline for ChronosBolt BART model"""
    forecast_type = ForecastType.QUANTILES
    
    def __init__(self, model: ChronosBartForForecasting):
        super().__init__(inner_model=model)
        self.model = model
        
    @property
    def quantiles(self) -> List[float]:
        return self.model.config.chronos_config["quantiles"]
        
    def predict(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = False,
    ) -> torch.Tensor:
        context_tensor = self._prepare_and_validate_context(context)
        
        model_prediction_length = self.model.chronos_config.prediction_length
        if prediction_length is None:
            prediction_length = model_prediction_length
            
        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend prediction length <= {model_prediction_length}. "
                "Longer predictions may degrade in quality."
            )
            if limit_prediction_length:
                msg += " Set limit_prediction_length=False to override."
                raise ValueError(msg)
            warnings.warn(msg)
            
        predictions = []
        remaining = prediction_length
        
        # Handle long context
        if context_tensor.shape[-1] > self.model.chronos_config.context_length:
            context_tensor = context_tensor[
                ..., 
                -self.model.chronos_config.context_length:
            ]
            
        while remaining > 0:
            with torch.no_grad():
                prediction = self.model(
                    context=context_tensor.to(
                        device=self.model.device,
                        dtype=torch.float32,
                    ),
                ).quantile_preds.to(context_tensor)
                
            predictions.append(prediction)
            remaining -= prediction.shape[-1]
            
            if remaining <= 0:
                break
                
            # Use median prediction for next step
            central_idx = torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]
            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)
            
        return torch.cat(predictions, dim=-1)[..., :prediction_length]
        
    def predict_quantiles(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **predict_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get quantile forecasts and mean predictions"""
        
        predictions = self.predict(
            context, 
            prediction_length=prediction_length,
            **predict_kwargs
        ).detach().swapaxes(1, 2)
        
        training_quantiles = self.quantiles
        
        # Handle requested quantiles
        if set(quantile_levels).issubset(set(training_quantiles)):
            # Directly map to trained quantiles
            quantiles = predictions[
                ..., 
                [training_quantiles.index(q) for q in quantile_levels]
            ]
        else:
            # Need to interpolate quantiles
            if min(quantile_levels) < min(training_quantiles) or max(quantile_levels) > max(training_quantiles):
                logger.warning(
                    f"Requested quantiles {quantile_levels} outside training range {training_quantiles}. "
                    "Results will be capped at training quantile range."
                )
                
            # Add boundary quantiles for interpolation
            augmented_predictions = torch.cat(
                [
                    predictions[..., [0]], 
                    predictions,
                    predictions[..., [-1]]
                ],
                dim=-1
            )
            
            # Interpolate to requested quantiles
            quantiles = torch.quantile(
                augmented_predictions,
                q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype),
                dim=-1
            ).permute(1, 2, 0)
            
        # Use median as mean prediction
        mean = predictions[:, :, training_quantiles.index(0.5)]
        
        return quantiles, mean
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load pretrained model from local path or HuggingFace Hub"""
        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"
        
        architecture = config.architectures[0]
        class_ = globals().get(architecture)
        
        if class_ is None:
            logger.warning(
                f"Unknown architecture: {architecture}, defaulting to ChronosBartForForecasting"
            )
            class_ = ChronosBartForForecasting
            
        model = class_.from_pretrained(*args, **kwargs)
        return cls(model=model)