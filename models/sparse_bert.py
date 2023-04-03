import torch
from torch import nn
from transformers import BertModel, BertConfig, PretrainedConfig, PreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.biencoder_base import BiEncoderBase

class SparseBert(BiEncoderBase):
    
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}
        config = SparseBertModelConfig(sparse_dim=kwargs['sparse_dim'])
        self.model = SparseBertModel(config)
    

class SparseBertModelConfig(PretrainedConfig):
	model_type = "SparseBert"
	sparse_dim: int

class SparseBertModel(PreTrainedModel):
    config_class = SparseBertModelConfig 
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.bert = BertModel.from_pretrained('bert-base-uncased', add_pooling_layer=False)
        #self.bert = BertModel(BertConfig(num_hidden_layers=12))
        #self.bert.embeddings = BertModel.from_pretrained('bert-base-uncased', add_pooling_layer=False).embeddings
        
        #self.bert.embeddings.position_embeddings.weight.data = torch.zeros_like(self.bert.embeddings.position_embeddings.weight.data)
        #self.bert.embeddings.token_type_embeddings.weight.data = torch.zeros_like(self.bert.embeddings.token_type_embeddings.weight.data)
        #self.bert.embeddings.position_embeddings.requires_grad = False
        #self.bert.embeddings.token_type_embeddings.requires_grad = False

        #self.bert.config.vocab_size = cfg.sparse_dim
        #self.head = BertOnlyMLMHead(config=self.bert.config)
        self.head = nn.Linear(self.bert.config.hidden_size, cfg.sparse_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg.sparse_dim))
        self.head.bias = self.bias
        print(cfg)

    @staticmethod
    def from_config(config):
        cfg = SparseBertModelConfig()
        cfg.sparse_dim = config["sparse_dim"]
        return SparseBertModel(cfg)


    def forward(self, **kwargs):

        out = self.bert(input_ids=kwargs['input_ids'], output_hidden_states=True) # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        #out = self.head(out[0])
        #out, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
        #take cls
        out = out.last_hidden_state[:,0]
        #out = torch.log(1 + torch.relu(self.head(out.last_hidden_state[:,0])))
        return out
