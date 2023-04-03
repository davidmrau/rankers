
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Longformer():

    def __init__(self, kwargs):

        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096')
        self.type = 'cross' 

    def get_scores(self,features, index):
        encoded_input = features['encoded_input'][index]
        out_raw = self.model(**encoded_input.to('cuda'))
        scores = out_raw.logits[:, 0]
        return_dict = {}
        return_dict['scores'] = scores
        return return_dict

