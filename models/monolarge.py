from transformers import AutoTokenizer,AutoModel, PreTrainedModel,PretrainedConfig
from typing import Dict
import torch
from torch import nn as nn
import time


class MonoLarge():

    def __init__(self, kwargs):
        self.type = 'bi'
        model_name = 'castorini/monobert-large-msmarco-finetune-only' 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.kwargs = kwargs
    
    def get_scores(self, features, index):

        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
        emb_queries = self.model(**encoded_queries.to('cuda')).last_hidden_state[:,0,:]
        emb_docs = self.model(**encoded_docs.to('cuda')).last_hidden_state[:,0,:]
        scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
        #finish = (time.time() - start)
        return_dict = {}
        return_dict['scores'] = scores
        #return_dict['time'] = finish
        return return_dict

