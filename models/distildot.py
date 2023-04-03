from transformers import AutoTokenizer,AutoModel, PreTrainedModel,PretrainedConfig
from typing import Dict
import torch
from torch import nn as nn
import time


class DistilDot()


    def __init__(self, kwargs):

	self.kwargs = kwargs
        model_name = 'sebastian-hofstaetter/distilbert-dot-margin_mse-T2-msmarco'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.type = 'bi'

    def get_scores(self, features, index):
	encoded_queries = features['encoded_queries']
	encoded_docs = features['encoded_docs'][index]

	emb_queries = self.model(**encoded_queries.to('cuda'))[0][:,0,:].squeeze(0)
	emb_docs = self.model(**encoded_docs.to('cuda'))[0][:,0,:].squeeze(0)
	scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
	return_dict = {}
	return_dict['scores'] = scores
	return return_dict


