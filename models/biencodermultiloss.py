
from transformers import AutoTokenizer, AutoModel
from models.biencoder_base import BiEncoderBase

import torch
class BiEncoderMultiLoss(BiEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        model_name = "Luyu/co-condenser-marco"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    
    def forward(self, **args):
        out = self.model(**args, output_hidden_states=True)
        return [layer[:, 0, :] for layer in out.hidden_states[9:]]
    def get_scores(self, features, index):
        return_dict = {}
        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
        emb_queries = self.forward(**encoded_queries.to('cuda'))
        emb_docs = self.forward(**encoded_docs.to('cuda'))
        if len(features['encoded_docs']) == 1:
            emb_queries = [emb_queries[-1]]
            emb_docs = [emb_docs[-1]]
        scores = list()
        for q, d in  zip(emb_queries, emb_docs):
            scores.append(torch.bmm(q.unsqueeze(1), d.unsqueeze(-1)).squeeze())
        scores = torch.cat(scores, dim=0)
        return_dict['scores'] = scores
        return return_dict
