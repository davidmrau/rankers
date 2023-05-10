
from transformers import BertTokenizer, AutoModelForSequenceClassification
from models.crossencoder_base import CrossEncoderBase
from transformers.models.bert.modeling_bert import BertEmbeddings

import torch


class BertScaleEmbeddings(CrossEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask, token_type_ids, scores):
        embs = self.model.module.bert.embeddings(input_ids)
        outmap_min, _ = torch.min(scores, dim=1, keepdim=True)
        outmap_max, _ = torch.max(scores, dim=1, keepdim=True)
        normed_scores = (scores - outmap_min) / (outmap_max - outmap_min)
        mask = token_type_ids == 0
        #mask = torch.ones_like(scores, requires_grad=False)
        embs = embs * (normed_scores * mask).unsqueeze(-1)
        #embs = embs *  (mask).unsqueeze(-1)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


    def get_scores(self, features, index):
        encoded_input = features['encoded_input'][index]
        out_raw = self.forward(**encoded_input.to('cuda'), scores=features['scores'][index].to('cuda'))
        scores = out_raw.logits[:, 1]
        return_dict = {}
        return_dict['scores'] = scores
        if self.kwargs['save_last_hidden']:
            return_dict['last_hidden'] = out_raw['hidden_states'][-1][:,0,:]
        return return_dict


