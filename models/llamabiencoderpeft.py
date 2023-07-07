
from transformers import AutoTokenizer, LlamaModel
from models.biencoder_base import BiEncoderBase

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftType,
    TaskType,
    PeftModelForSequenceClassification
)


import torch
class LlamaBiEncoderPeft(BiEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        model_name = 'huggyllama/llama-7b'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = LlamaModel.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.float16)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.model.config.eos_token_id
    
        self.model = prepare_model_for_int8_training(self.model)
        config = LoraConfig(
                r=8,
                lora_alpha=16,
                inference_mode=False,
                lora_dropout=0.1,
                bias="lora_only",
                target_modules =  [
                "q_proj",
                "v_proj",
            ],
                task_type=TaskType.SEQ_CLS,
            )

        self.model = get_peft_model(self.model, config)
        print(self.model.print_trainable_parameters())

    def forward(self, **kwargs):
        del kwargs['token_type_ids']
        out = self.model(**kwargs).last_hidden_state[:, -1, :]
        return out

