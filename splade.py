import torch
import numpy as np
from transformers import PreTrainedModel,PretrainedConfig, AutoModelForMaskedLM, AutoModelForSequenceClassification, BertModel


class SpladeConfig(PretrainedConfig):
    model_type = "Splade"
    base_model = 'bert-base-uncased'

class Splade(PreTrainedModel):

    config_class = SpladeConfig
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.bert_model = AutoModelForMaskedLM.from_pretrained(cfg.base_model)
        #self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    
    def forward(self, **kwargs):
        out = self.bert_model(**kwargs)["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        out, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
        return out

#    def forward(self, **kwargs):
#        out = self.bert_model(**kwargs)["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
#        #out = self.bert_model(**kwargs, output_hidden_states=True)['last_hidden_state'] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
#        #out = out[:,0]
#        #exact_match = torch.nn.functional.one_hot(kwargs['input_ids'], num_classes=self.bert_model.config.vocab_size)
#        #out += exact_match
#        out, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
#        return out

