
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TinyBert():

    def __init__(self, kwargs):

        model_name = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.type = 'cross'


    def get_scores(self, features, index):
        encoded_input = features['encoded_input'][index]
        out_raw = self.model(**encoded_input.to('cuda'))
        scores = out_raw.logits[:, 0]
        return_dict = {}
        return_dict['scores'] = scores
        return return_dict

