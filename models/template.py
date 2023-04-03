from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Template():
    
    def __init__(self, kwargs):

        self.kwargs = kwargs 
        tokenizer = NotImplemented() 
        tokenizer = NotImplemented() 
        self.type= NotImplemented() # either 'cross'  or 'bi'

    def get_scores(self, features, index):
        encoded_input = features['encoded_input'][index]
        scores = NotImplemented() # model inference. for example self.model(**encoded_input.to('cuda'))
        return_dict = {}
        return_dict['scores'] = scores
        return return_dict

