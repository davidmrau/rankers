
from transformers import AutoTokenizer, LlamaModel
from models.biencoder_base import BiEncoderBase


class LlamaBiEncoder(BiEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        model_name = 'huggyllama/llama-7b'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = LlamaModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.model.config.eos_token_id
    

        for param in self.model.parameters():
                param.requires_grad = False

        for name, param in self.model.named_parameters():
            if 'layers' in name:
                if int(name.split('.')[1]) > 29:
                        param.requires_grad = True
    def forward(self, **kwargs):
        out = self.model(kwargs['input_ids']).last_hidden_state[:, -1, :]
        return out

