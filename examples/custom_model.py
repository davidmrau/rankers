from transformers import PretrainedConfig, PreTrainedModel
import torch

class CustomModelConfig(PretrainedConfig):
    model_type = "CustomModel"

class CustomModel(PreTrainedModel):

    config_class = CustomModelConfig
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.embedding = torch.nn.Embedding(30522, 768)
        self.classifier = torch.nn.Linear(768, 2) 
 
    def forward(self, **kwargs):
        input_ids = kwargs['input_ids']
        return self.classifier(self.embedding(input_ids))
