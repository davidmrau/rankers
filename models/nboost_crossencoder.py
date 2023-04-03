
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.crossencoder_base import CrossEncoderBase
class nboostCrossEncoder(CrossEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        model_name = "nboost/pt-bert-base-uncased-msmarco"
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side=kwargs['truncation_side'])
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)


