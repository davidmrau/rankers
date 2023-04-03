
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.crossencoder_base import CrossEncoderBase
class CrossEncoder2(CrossEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs    
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side=kwargs['truncation_side'])
        self.model = AutoModelForSequenceClassification.from_config(BertConfig(num_hidden_layers=2))
        
        # copy embeddings from bert-base-uncased
        #model_pre = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
        #model.bert.embeddings = model_pre.bert.embeddings
        #model.bert.encoder.layer[0] = model_pre.bert.encoder.layer[0]
        #del model_pre

