
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SortBert():

    def __init__(self, kwargs):

        self.kwargs = kwargs 
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side=kwargs['truncation_side'])
        self.model = torch.load('/project/draugpu/experiments_bert_model/experiments_msmarco/model_bz_64_lr_3e-06_do_0.2_sr__cls_1000fine_tune_shuffle/model.epfinal.pth')
        self.model = model.module
        self.model.to('cpu')
        self.type = 'cross' 

    def get_scores(self, features, index):
        encoded_input = features['encoded_input'][index]
        out_raw = self.model(**encoded_input.to('cuda'))
        scores = out_raw
        return_dict = {}
        return_dict['scores'] = scores
        return return_dict

