from transformers import PreTrainedModel
from .config import MetaheadConfig
import torch
import torch.nn as nn
from .glue import DimensionGlueConfig, DimensionGlue

class MetaheadPreTrainedModel(PreTrainedModel):
    config_class = MetaheadConfig
    base_model_prefix = "meta_head"
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

class MetaheadModel(MetaheadPreTrainedModel):
    def __init__(self, config: MetaheadConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dimension_glue = DimensionGlue(DimensionGlueConfig())

        self.gradient_checkpointing = False
        self.force_gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def save_train_weight_only(self, path: str):
        self.dimension_glue.save_pretrained(path)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def embedding(self, x):
        return self.dimension_glue(self.embed_tokens(x), "down")

    def predicting(self, x):
        return self.lm_head(self.dimension_glue(x, "up"))

    def forward(self, x, direction):
        if direction == 'input':
            return self.embedding(x)
        else:
            return self.predicting(x)