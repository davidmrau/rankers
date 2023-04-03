
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class LongformerQA():

    def __init__(self, kwargs):

        self.kwargs = kwargs
        self.model = AutoModelForSequenceClassification.from_pretrained("aware-ai/longformer-QA")
        self.tokenizer = AutoTokenizer.from_pretrained("aware-ai/longformer-QA")
        self.type = 'cross' 

    def get_scores(self,features, index):
        encoded_input = features['encoded_input'][index]
        out_raw = self.model(**encoded_input.to('cuda'))
        scores = out_raw.logits[:, 0]
        return_dict = {}
        return_dict['scores'] = scores
        return return_dict

