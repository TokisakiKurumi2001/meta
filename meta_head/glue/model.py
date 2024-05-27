from transformers import PreTrainedModel
from .config import DimensionGlueConfig
import torch.nn as nn
import torch

class DimensionGluePretrainedModel(PreTrainedModel):
    config_class = DimensionGlueConfig
    base_model_prefix = "dimension_glue"
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

class DimensionGlue(DimensionGluePretrainedModel):
    def __init__(self, config: DimensionGlueConfig):
        super().__init__(config)
        self.w = nn.Parameter(torch.normal(0, 0.02, (config.input_dim, config.output_dim)))
    
    def down_scale(self, x):
        return x @ self.w

    def up_scale(self, x):
        return x @ torch.transpose(self.w, 0, 1)

    def forward(self, x, direction):
        if direction == "up":
            return self.up_scale(x)
        else:
            return self.down_scale(x)