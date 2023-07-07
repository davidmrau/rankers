from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, PretrainedConfig, modeling_utils
from models.biencoder_base import BiEncoderBase
import torch.nn as nn
import torch



class BiEncoderAttentionPool(BiEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        model_name = "Luyu/co-condenser-marco-retriever"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        cfg = SelfAttentionPoolBertConfig(model_name=model_name)
        self.model = SelfAttentionPoolBert(cfg)
    
class SelfAttentionPoolBertConfig(PretrainedConfig):
    def __init__(self, model_name='bert-base-uncased'):
        self.model_type = "bert"
        self.model_name = model_name

class SelfAttentionPoolBert(PreTrainedModel):
    config_class = SelfAttentionPoolBertConfig
    
    def __init__(self, cfg):
        super().__init__(cfg)
        model_name = cfg.model_name
        self.self_attention_pool = AttentionPooling(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        
    def forward(self, **kwargs):
        input_ids, attention_mask = kwargs['input_ids'], kwargs['attention_mask']
        aggregated_embedding = self.self_attention_pool(input_ids, attention_mask)
        outputs = self.bert_model(inputs_embeds=aggregated_embedding)
        return outputs

class AttentionPooling(nn.Module):
    def __init__(self, model_name):
        super(AttentionPooling, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.get_extended_attention_mask = self.bert.get_extended_attention_mask
        self.embeddings = self.bert.embeddings.word_embeddings
        self.num_attention_heads = self.bert.config.num_attention_heads
        self.encoder = self.bert.encoder.layer[0].attention.self

        del self.bert

    def forward(self, input_ids, attention_mask):
        embeddings = self.embeddings(input_ids)  # Get word embeddings
        input_shape = embeddings.size()[:-1]
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        self_attention_weights = self.encoder(hidden_states=embeddings, attention_mask=extended_attention_mask, output_attentions=True)[1].mean(2)  # Pass through the first layer
        # Multiply attention weights with hidden states and sum across sequence length
        pooled_representation = torch.bmm(self_attention_weights, embeddings).squeeze(1)
        
        return pooled_representation

