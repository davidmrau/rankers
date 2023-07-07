import torch
from torch import nn
from transformers import BertModel, BertConfig, PretrainedConfig, PreTrainedModel, AutoModelForMaskedLM, AutoTokenizer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from models.biencoder_base import BiEncoderBase

class SpladeCocondenserEnsembleDistil(BiEncoderBase):
    
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}

        config = SpladeConfig(base_model='naver/splade-cocondenser-ensembledistil')
        self.model = Splade(config)

    def forward(self, **kwargs):
        return self.model(**kwargs)


class SpladeCocondenser(BiEncoderBase):
    
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}

        model = 'Luyu/co-condenser-marco'
        config = SpladeConfig(base_model=model)
        self.model = Splade(config)

class SpladeConfig(PretrainedConfig):
    model_type = "Splade"
    base_model = 'bert-base-uncased'

class Splade(PreTrainedModel):

    config_class = SpladeConfig
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.bert_model = AutoModelForMaskedLM.from_pretrained(cfg.base_model)

    
    def forward(self, **kwargs):
        out = self.bert_model(**kwargs)["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        out, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
        return out
