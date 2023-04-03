from transformers import AutoTokenizer, AutoModelForSequenceClassification
from crossencoder_base import CrossEncoderBase
class BowBert(CrossEncoderBase):

    def __init__(self, kwargs):

        self.kwargs = kwargs 
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side=kwargs['truncation_side'])
        self.model = AutoModelForSequenceClassification.from_pretrained('dmrau/bow-bert')

