import numpy as np 
import torch

class BiEncoderBase(): 

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.type = 'bi'


    def decode(self, encoded_input):
        logits = self.model(**encoded_input.to('cuda')).cpu().detach().numpy()
        weight_dicts = self.get_weight_dicts(logits)
        return weight_dicts
       
    def get_weight_dicts(self, batch_aggregated_logits):
        to_return = []
        for aggregated_logits in batch_aggregated_logits:
            col = np.nonzero(aggregated_logits)[0]
            weights = aggregated_logits[col]
            d = {self.reverse_voc[k]: float(v) for k, v in zip(list(col), list(weights))}
            to_return.append(d)
        return to_return

    def encode(self, encoded_input):
        emb = self.model(**encoded_input.to('cuda'))
        return emb

    def get_scores(self, features, index):
        return_dict = {}
        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
        emb_queries = self.model(**encoded_queries.to('cuda'))
        emb_docs = self.model(**encoded_docs.to('cuda'))
        def l1(batch_rep):
            return torch.sum(torch.abs(batch_rep), dim=-1).mean()

        def flops(batch_rep):
            return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)
        def l0(batch_rep):
            return torch.count_nonzero(batch_rep, dim=-1).float().mean()

        def used_dims(batch_rep):
            return torch.count_nonzero(batch_rep, dim=0).float().mean()
            
        return_dict['l1_queries'] = flops(emb_queries)
        return_dict['l1_docs'] = flops(emb_docs)
        scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
        return_dict['scores'] = scores
        return_dict['l0_docs'] = l0(emb_docs)
        return_dict['used_dims'] = used_dims(emb_docs)
        return return_dict
