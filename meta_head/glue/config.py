from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class DimensionGlueConfig(PretrainedConfig):
    model_type = "dimension_glue"
    
    def __init__(self, input_dim: int=4096, output_dim: int=2048, pad_token_id=0, bos_token_id=1, eos_token_id=2,**kwargs,):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
