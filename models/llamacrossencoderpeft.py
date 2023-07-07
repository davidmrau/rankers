
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.crossencoder_base import CrossEncoderBase
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
class LlamaCrossEncoderPeft(CrossEncoderBase):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        model_name = 'huggyllama/llama-7b'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.float16)

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

    def get_scores(self, features, index):
        encoded_input = features['encoded_input'][index]
        del encoded_input['token_type_ids']
        out_raw = self.model(**encoded_input.to('cuda'))
        scores = out_raw.logits[:, 1]
        return_dict = {}
        return_dict['scores'] = scores
        return return_dict




