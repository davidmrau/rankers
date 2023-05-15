
from transformers import BertModel, BertTokenizer, PreTrainedModel, PretrainedConfig
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.biencoder_base import BiEncoderBase


class BiSelector(BiEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side=kwargs['truncation_side'])
        self.model = SelectorModel(SelectorModelConfig(num_terms=kwargs.num_terms)) 

    def get_scores(self, features, index):
        return_dict = {}
        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
        emb_queries = self.model(encoded_queries['input_ids'].to('cuda'))
        emb_docs = self.model(encoded_docs['input_ids'].to('cuda'), reduce_input=True)

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
        
 

class SelectorModelConfig(PretrainedConfig):
    def __init__(self, num_terms, **kwargs):
        super().__init__(**kwargs)
	    model_type = 'SelectorModel'
	    num_terms = num_terms


class SelectorModel(PreTrainedModel):
    config_class  = SelectorModelConfig
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bert = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
        self.selector = CNNModel.from_pretrained('/scratch/drau/models/extractor_passage/num_steps_1000_bs_2048_fs_15_filters_768_lr_3e-05_pat_3_max_len_256/')
    

    def forward(self, input_ids, reduce_input=False):
        if reduce_input:
            input_ids, embeds = self.selector.get_tokens(input_ids, self.model.config.num_terms)
        emb_docs = self.bert(input_ids).logits
        return emb_docs

class CNNModelConfig(PretrainedConfig):
    model_type = 'CNNModel'
    filter_sizes = None
    num_filters = None

class CNNModel(PreTrainedModel):
    config_class  = CNNModelConfig

    @staticmethod
    def from_config(config):
        cfg = CNNModelConfig()
        cfg.filter_sizes         = config['filter_sizes']
        cfg.num_filters          = config['num_filters']
        return CNNModel(cfg)


    def __init__(self, cfg):
        super().__init__(cfg)
       
        num_filters = cfg.num_filters
        filter_sizes = cfg.filter_sizes
        output_dim = 1
        dropout = .5

        self.word_embeddings = BertModel.from_pretrained('huggingface/bert-base-uncased/').embeddings.word_embeddings
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels=self.word_embeddings.embedding_dim, 
                      out_channels=num_filters, 
                      kernel_size=fs,
                      padding=(fs)//2)  # ensure padding is same for all filter sizes
            for fs in filter_sizes
        ])
        self.fc = torch.nn.Conv1d(in_channels=len(filter_sizes)*num_filters, out_channels=1, kernel_size=output_dim)
        #self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
    
    def get_tokens(self, x, topk):
        scores, embeds = self.forward(x)
        scores[x==self.tokenizer.pad_token_id] = -1000
        indices = torch.topk(scores, topk, dim=1)[1].sort()[0]
        embeds_selected_tokens = embeds.permute(0,2,1).gather(dim=1, index=indices.unsqueeze(2).repeat(1,1,embeds.shape[1]))
        return x.gather(dim=1, index=indices), embeds_selected_tokens

    def forward(self, x):
        embds = self.word_embeddings(x)
        # x: [batch size, seq length, embedding dim]
        embds = embds.permute(0, 2, 1)  # [batch size, embedding dim, seq length]
        # Apply convolutional layers
        conved = [F.relu(conv(embds)) for conv in self.convs]
        # Concatenate pooled features and apply dropout
        cat = self.dropout(torch.cat(conved, dim=1))

        # Fully connected layer
        output = self.fc(cat)
        #output = F.relu(output)
        #output = torch.sigmoid(output)
        #output = torch.softmax(output, dim=1)
        return output.squeeze(1), embds
