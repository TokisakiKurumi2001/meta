from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import PreTrainedModel
import torch.nn as nn
import torch

logger = logging.get_logger(__name__)

class ReversibleMLPConfig(PretrainedConfig):
    model_type = "reversible_mlp"
    
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

class ReversibleMLPPretrainedModel(PreTrainedModel):
    config_class = ReversibleMLPConfig
    base_model_prefix = "reversible_mlp"
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

class ReversibleMLP(ReversibleMLPPretrainedModel):
    def __init__(self, config: ReversibleMLPConfig):
        super().__init__(config)
        self.w1 = nn.Parameter(torch.normal(0, 0.02, (config.input_dim, config.hidden_dim)))
        self.w2 = nn.Parameter(torch.normal(0, 0.02, (config.hidden_dim, config.output_dim)))
        if config.act_fn == 'relu':
            self.act_fn = nn.ReLU()
   
    def down_scale(self, x):
        y_hat = x @ self.w1
        y_hat = self.act_fn(y_hat)
        return y_hat @ self.w2

    def up_scale(self, x):
        y_hat = x @ torch.transpose(self.w2, 0, 1)
        y_hat = self.act_fn(y_hat)
        return y_hat @ torch.transpose(self.w1, 0, 1)

    def forward(self, x, direction):
        if direction == "up":
            return self.up_scale(x)
        else:
            return self.down_scale(x)
