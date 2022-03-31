import torch

from transformers import AutoModelForMaskedLM

class Splade(torch.nn.Module):
        def __init__(self, model_type_or_dir, agg="max"):
            super().__init__()
            self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
            assert agg in ("sum", "max")
            self.agg = agg

        def forward(self, **kwargs):
            out = self.transformer(**kwargs)["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
            if self.agg == "max":
                values, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
                return values
# 0 masking also works with max because all activations are positive
            else:
                return torch.sum(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
