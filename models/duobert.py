from transformers import AutoTokenizer,AutoModel, PreTrainedModel,PretrainedConfig
from typing import Dict
import torch
from torch import nn as nn
import time


class DUOBert():

    def __init__(self, kwargs):
        model_name = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        self.model = AutoModel.from_pretrained(model_name)
        self.type = 'bi'
        self.kwargs = kwargs
    
    def get_scores(self, features, index):
        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
        emb_queries = self.model(**encoded_queries.to('cuda')).last_hidden_state[:,0,:].squeeze(0)
        emb_docs = self.model(**encoded_docs.to('cuda')).last_hidden_state[:,0,:].squeeze(0)
        scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
        return_dict = {}
        return_dict['scores'] = scores
        return return_dict

