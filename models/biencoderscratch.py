
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from models.biencoder_base import BiEncoderBase


class BiEncoderScratch(BiEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification(BertConfig())
        
