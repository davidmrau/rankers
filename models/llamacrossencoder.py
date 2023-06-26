
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.crossencoder_base import CrossEncoderBase
class LlamaCrossEncoder(CrossEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        model_name = 'huggyllama/llama-7b'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        for param in self.model.parameters():
                param.requires_grad = False

        for name, param in self.model.named_parameters():
            if 'layers' in name:
                if int(name.split('.')[2]) > 30:
                        param.requires_grad = True
    def get_scores(self, features, index):
        encoded_input = features['encoded_input'][index]
        out_raw = self.model(encoded_input.to('cuda')['input_ids'])
        scores = out_raw.logits[:, 1]
        return_dict = {}
        return_dict['scores'] = scores
        if self.kwargs['save_last_hidden']:
            return_dict['last_hidden'] = out_raw['hidden_states'][-1][:,0,:]
        return return_dict




