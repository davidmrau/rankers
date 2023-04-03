
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.crossencoder_base import CrossEncoderBase
class CrossEncoder(CrossEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side=kwargs['truncation_side'])
        self.model = AutoModelForSequenceClassification.from_pretrained('dmrau/crossencoder-msmarco')


