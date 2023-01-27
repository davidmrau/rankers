import torch
import numpy as np
from transformers import PreTrainedModel,PretrainedConfig, AutoModelForMaskedLM


class SpladeConfig(PretrainedConfig):
    model_type = "Splade"

class Splade(PreTrainedModel):

    config_class = SpladeConfig
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bert_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

    def forward(self, **kwargs):
        out = self.bert_model(input_ids=kwargs['input_ids'])["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        out, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
        return out

