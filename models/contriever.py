from transformers import AutoTokenizer,AutoModel, PreTrainedModel,PretrainedConfig
from typing import Dict
import torch
from torch import nn as nn
import time


class Contriever():

    def __init__(self, kwargs):
        model_name = 'facebook/contriever-msmarco'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.type = 'bi'
        self.kwargs = kwargs
    
    def get_scores(self, features, index):
        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
        emb_queries = self.model(**encoded_queries.to('cuda'))
        emb_docs = self.model(**encoded_docs.to('cuda'))
        emb_queries_av = mean_pooling(emb_queries[0], encoded_queries['attention_mask'])
        emb_docs_av = mean_pooling(emb_docs[0], encoded_docs['attention_mask'])
        scores = torch.bmm(emb_queries_av.unsqueeze(1), emb_docs_av.unsqueeze(-1)).squeeze()
        return_dict = {}
        return_dict['scores'] = scores
        return return_dict

