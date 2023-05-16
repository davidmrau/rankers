
from transformers import BertModel, BertTokenizer, PreTrainedModel, PretrainedConfig
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.crossencoder_base import CrossEncoderBase
from transformers import BatchEncoding

class CrossSelector(CrossEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side=kwargs['truncation_side'])
        print(kwargs)
        self.model = SelectorModel(SelectorModelConfig(num_terms=kwargs['num_terms'])) 
        self.type = 'cross-selector'


    def get_scores(self, features, index):
        encoded_docs = features['encoded_docs'][index].to('cuda') 
        encoded_queries = features['encoded_queries'].to('cuda')
        out_raw = self.model(encoded_queries['input_ids'], encoded_docs['input_ids'])
        scores = out_raw.logits[:, 1]
        return_dict = {}
        return_dict['scores'] = scores
        if self.kwargs['save_last_hidden']:
            return_dict['last_hidden'] = out_raw['hidden_states'][-1][:,0,:]
        return return_dict
 

class SelectorModelConfig(PretrainedConfig):
    model_type = 'SelectorModel'
    num_terms = None 
   # def __init__(self, num_terms, **kwargs):
   #     super().__init__(**kwargs)

    def to_dict(self):
        """Converts the configuration to a dictionary."""
        config_dict = super().to_dict()
        config_dict['num_terms'] = self.num_terms
        return config_dict


class SelectorModel(PreTrainedModel):
    config_class  = SelectorModelConfig
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bert = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
        #self.selector = CNNModel.from_pretrained('/scratch/drau/models/extractor_passage/num_steps_1000_bs_2048_fs_15_filters_768_lr_3e-05_pat_3_max_len_256/')
        self.selector = CNNModel.from_pretrained('/scratch/drau/models/extractor_passage/num_steps_1000_bs_2048_fs_1_15_31_filters_768_lr_3e-05_pat_3_max_len_256')
        cnn_config = CNNModelConfig(filter_sizes=[31], num_filters=768)
        self.selector = CNNModel(cnn_config)
        self.num_terms = cfg.num_terms 
    def merge_inp(self, inp_ids_queries, inp_ids_docs):
        
        # input ids
        
        merged_input_ids = torch.cat((inp_ids_queries, inp_ids_docs ), dim=1)
        device = inp_ids_queries.device
        # attention_mask
        attention_mask = torch.ones_like(merged_input_ids, device=device)
        # token_type_ids
        token_type_ids_q = torch.zeros((attention_mask.shape[0], (attention_mask.shape[1]//2) + 1), device=device)
        token_type_ids_d = torch.ones((attention_mask.shape[0], (attention_mask.shape[1]//2)-1), device=device)
        token_type_ids = torch.cat((token_type_ids_q, token_type_ids_d), dim=1)
         
        return BatchEncoding(data={'input_ids': merged_input_ids.int(), 'attention_mask': attention_mask.int(), 'token_type_ids': token_type_ids.int()})


    def forward(self, inp_ids_queries, inp_ids_docs):

        reduced_inp_ids_docs, _ = self.selector.get_tokens(inp_ids_docs, self.num_terms)
        merged_input = self.merge_inp(inp_ids_queries, reduced_inp_ids_docs)
        out_raw = self.bert(**merged_input)
        return out_raw

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
        scores[x == 0] = -1000
        indices = torch.topk(scores, topk, dim=1)[1]#.sort()[0]
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
