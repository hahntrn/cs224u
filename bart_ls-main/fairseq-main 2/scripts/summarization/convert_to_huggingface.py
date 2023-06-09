import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .bart import BARTModel

class MyConfig(PretrainedConfig):
    model_type = 'bartls_govreport'
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

class MyModel(PreTrainedModel):
    config_class = MyConfig
    def __init__(self, config, model):
        super().__init__(config)
        self.config = config
        self.model = model
    def forward(self, input):
        return self.model(input) 