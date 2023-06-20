import torch
from torch import nn
from transformers import BertModel, BertConfig, PretrainedConfig, PreTrainedModel, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.biencoder_base import BiEncoderBase
import copy
class SparseBertMLM(BiEncoderBase):
    
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained('naver/splade-cocondenser-ensembledistil')
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}
        config = SparseBertMLMModelConfig()
        self.model = SparseBertMLMModel(config)
    
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
            
        return_dict['reg_queries'] = flops(emb_queries)
        return_dict['reg_docs'] = flops(emb_docs)
        scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
        return_dict['scores'] = scores
        return_dict['l0_docs'] = l0(emb_docs)
        return_dict['used_dims'] = used_dims(emb_docs)
        return return_dict

class SparseBertMLMModelConfig(PretrainedConfig):
        model_type = "SparseBertMLMModel"

class SparseBertMLMModel(PreTrainedModel):
    config_class = SparseBertMLMModelConfig 
    def __init__(self, cfg):
        super().__init__(cfg)
        
        pretrained_model = BertForMaskedLM.from_pretrained('naver/splade-cocondenser-ensembledistil')
        self.bert = copy.deepcopy(pretrained_model.bert)
        self.head = copy.deepcopy(pretrained_model.cls)
        del pretrained_model

    @staticmethod
    def from_config(config):
        cfg = SparseBertMLMModelConfig()
        return SparseBertMLMModel(cfg)


    def forward(self, **kwargs):

        out = self.bert(input_ids=kwargs['input_ids'], output_hidden_states=True) # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        #out = self.head(out[0])
        #out, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
        #take cls
        out = out.last_hidden_state[:,0]
        out = torch.log(1 + torch.relu(self.head(out)))
        return out

