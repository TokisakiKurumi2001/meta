from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import PreTrainedModel
import torch.nn as nn
import torch
from .separate_mlp import SeparateMLP, SeparateMLPConfig, SeparateMLPPretrainedModel

logger = logging.get_logger(__name__)


class NormalMLP(SeparateMLPPretrainedModel):
    def __init__(self, config: SeparateMLPConfig):
        super().__init__(config)
        self.w1 = nn.Parameter(torch.normal(0, 0.02, (config.input_dim, config.hidden_dim)))
        self.w2 = nn.Parameter(torch.normal(0, 0.02, (config.hidden_dim, config.output_dim)))
        if config.act_fn == 'relu':
            self.act_fn = nn.ReLU()

    def forward(self, x):
        y_hat = x @ self.w1
        y_hat = self.act_fn(y_hat)
        return y_hat @ self.w2

class InvAdapterConfig(PretrainedConfig):
    model_type = "invertible_adapter"
    
    def __init__(self, input_dim: int=4096, output_dim: int=2048, hidden_dim: int=1024, act_fn: str='relu', pad_token_id=0, bos_token_id=1, eos_token_id=2,**kwargs,):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.act_fn = act_fn
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

class InvAdapterPretrainedModel(PreTrainedModel):
    config_class = InvAdapterConfig
    base_model_prefix = "invertible_adapter"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class InvAdapter(InvAdapterPretrainedModel):
    def __init__(self, config: InvAdapterConfig):
        super().__init__(config)
        self.F = NormalMLP(
            SeparateMLPConfig(
                input_dim=config.input_dim // 2,
                output_dim=config.output_dim // 2,
                hidden_dim=config.hidden_dim // 2,
                act_fn=config.act_fn))
        self.G = NormalMLP(
            SeparateMLPConfig(
                input_dim=config.output_dim // 2,
                output_dim=config.output_dim // 2,
                hidden_dim=config.hidden_dim // 2))
        self.adapter_dim_down = config.input_dim // 2
        self.adapter_dim_up = config.output_dim // 2
        self.S = nn.Parameter(torch.normal(0, 0.02, (self.adapter_dim_down, self.adapter_dim_up)))
        self.S_inv = nn.Parameter(torch.normal(0, 0.02, (self.adapter_dim_up, self.adapter_dim_down)))
   
    def down_scale(self, x):
        e_1, e_2 = x[:, :, :self.adapter_dim_down], x[:, :, self.adapter_dim_down:]
        o_1 = self.F(e_2) + e_1 @ self.S
        o_2 = self.G(o_1) + e_2 @ self.S
        return torch.cat((o_1, o_2), dim=-1)

    def up_scale(self, x):
        o_1, o_2 = x[:, :, :self.adapter_dim_up], x[:, :, self.adapter_dim_up:]
        e_2 = (o_2 - self.G(o_1)) @ self.S_inv
        e_1 = (o_1 - self.F(e_2)) @ self.S_inv
        return torch.cat((e_1, e_2), dim=-1)

    def forward(self, x, direction):
        if direction == "up":
            return self.up_scale(x)
        else:
            return self.down_scale(x)
