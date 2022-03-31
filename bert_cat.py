from transformers import AutoTokenizer,AutoModel, PreTrainedModel,PretrainedConfig
from typing import Dict
import torch

class BERT_Cat_Config(PretrainedConfig):
    model_type = "BERT_Cat"
    bert_model: str
    trainable: bool = True

class BERT_Cat(PreTrainedModel):
    """
    The vanilla/mono BERT concatenated (we lovingly refer to as BERT_Cat) architecture 
    -> requires input concatenation before model, so that batched input is possible
    """
    config_class = BERT_Cat_Config
    base_model_prefix = "bert_model"

    def __init__(self,
                 cfg) -> None:
        super().__init__(cfg)
        
        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)

        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable

        self._classification_layer = torch.nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self,
                query_n_doc_sequence):

        vecs = self.bert_model(**query_n_doc_sequence)[0][:,0,:] # assuming a distilbert model here
        score = self._classification_layer(vecs)
        return score
