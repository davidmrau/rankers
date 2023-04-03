from transformers import AutoTokenizer,AutoModel, PreTrainedModel,PretrainedConfig
from typing import Dict
import torch
from torch import nn as nn
import time


class SentenceBert(): 
    def __init__(self, kwargs):

        self.kwargs = kwargs
        model_name = "sentence-transformers/msmarco-bert-base-dot-v5"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.type = 'bi'

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, encoded_input):
        emb = self.model(**encoded_input.to('cuda'))
        emb_av = self.mean_pooling(emb, encoded_input['attention_mask'])
        return emb_av

    def get_scores(self, features, index):

        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
        emb_queries = self.model(**encoded_queries.to('cuda'))
        emb_docs = self.model(**encoded_docs.to('cuda'))
        emb_queries_av = self.mean_pooling(emb_queries, encoded_queries['attention_mask'])
        emb_docs_av = self.mean_pooling(emb_docs, encoded_docs['attention_mask'])
        scores = torch.bmm(emb_queries_av.unsqueeze(1), emb_docs_av.unsqueeze(-1)).squeeze()
        return_dict = {}
        return_dict['scores'] = scores
        return return_dict


