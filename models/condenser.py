
from transformers import AutoTokenizer, AutoModel
from models.biencoder_base import BiEncoderBase


class Condenser(BiEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        #model_name = "bert-base-uncased"
        model_name = "Luyu/co-condenser-marco"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    

