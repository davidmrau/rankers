from transformers import AutoModelForSequenceClassification, PreTrainedModel, PretrainedConfig
import torch

class BERT_TF_Config(PretrainedConfig):
    model_type = "BERT_TF"
    bert_model: str
    trainable: bool = True

class BERT_TF(PreTrainedModel):
    def __init__(self,
                 cfg) -> None:
        super().__init__(cfg)
        
        self.bert_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable

        self.embeddings = self.bert_model.bert.embeddings

    def forward(self, input_ids, attention_mask, token_type_ids):
        inputs_embeds = self.embeddings(input_ids)
        out = self.bert_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return out
