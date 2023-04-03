
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Bigbird():

    def __init__(self, kwargs):
        tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained("google/bigbird-roberta-base")
        self.kwargs = kwargs
        self.type = 'cross'

    def get_scores(self,features, index):
        encoded_input = features['encoded_input'][index]
        out_raw = self.model(**encoded_input.to('cuda'))
        scores = out_raw.logits[:, 0]
        return_dict = {}
        return_dict['scores'] = scores
        return return_dict

