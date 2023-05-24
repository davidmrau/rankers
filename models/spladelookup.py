import torch
from torch import nn
from transformers import BertModel, BertConfig, PretrainedConfig, PreTrainedModel, AutoModelForMaskedLM, AutoTokenizer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from models.biencoder_base import BiEncoderBase

class SpladeLookup(BiEncoderBase):
    
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        model = 'naver/splade-cocondenser-ensembledistil'
        #model = 'Luyu/co-condenser-marco'
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}

        config = SpladeConfig(base_model=model)
        self.model = Splade(config)

    def get_scores(self, features, index):
        return_dict = {}
        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
        emb_queries = self.model(**encoded_queries.to('cuda'), is_query=True)
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

class SpladeConfig(PretrainedConfig):
    model_type = "Splade"
    base_model = 'bert-base-uncased'

class Splade(PreTrainedModel):

    config_class = SpladeConfig
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.bert_model = AutoModelForMaskedLM.from_pretrained(cfg.base_model)
        self.vocab_size = self.bert_model.config.vocab_size

    def one_hot(self, input_ids):
        one_hot_matrix = torch.nn.functional.one_hot(input_ids, num_classes=self.vocab_size)
        return one_hot_matrix


    def forward(self, **kwargs):
        if 'is_query' in kwargs:
            binary_emb = self.one_hot(kwargs['input_ids']).float().sum(1)
            return binary_emb
        out = self.bert_model(**kwargs)["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        out, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
        return out
