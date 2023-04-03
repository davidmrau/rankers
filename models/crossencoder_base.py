
class CrossEncoderBase():
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.type = 'cross'

    def get_scores(self, features, index):
        encoded_input = features['encoded_input'][index]
        out_raw = self.model(**encoded_input.to('cuda'))
        scores = out_raw.logits[:, 1]
        return_dict = {}
        return_dict['scores'] = scores
        if self.kwargs['save_last_hidden']:
            return_dict['last_hidden'] = out_raw['hidden_states'][-1][:,0,:]
        return return_dict

