import torch
import numpy as np

from transformers import AutoModelForMaskedLM

class Splade(torch.nn.Module):
        def __init__(self, model_type_or_dir):
                super().__init__()
                self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)

        def forward(self, **kwargs):
                out = self.transformer(input_ids=kwargs['input_ids'])["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
                out, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
                return out

