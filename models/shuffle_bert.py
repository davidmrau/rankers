from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ShuffleBert():

    def __init__(self, kwargs):

        self.kwargs = kwargs 
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side=kwargs['truncation_side'])
        self.model = AutoModelForSequenceClassification.from_pretrained('/project/draugpu/experiments_cikm/bert/bz_128_lr_3e-06shuffle/model_30/')
        self.type = 'cross' 

    def get_scores(self, features, index):
        encoded_input = features['encoded_input'][index]
        out_raw = self.model(**encoded_input.to('cuda'), output_hidden_states=self.kwargs['save_hidden_states'])
        scores = out_raw.logits[:, 1]
        return_dict = {}
        return_dict['scores'] = scores
        if save_hidden_states:
            return_dict['last_hidden'] = out_raw['hidden_states'][-1][:,0,:]
        return return_dict


