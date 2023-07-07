
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from models.crossencoder_base import CrossEncoderBase
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftType,
    TaskType,
    PeftModelForSequenceClassification,
    PromptTuningInit, PromptTuningConfig, TaskType, PeftType
)
import torch 

class LlamaCausalLM():
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.type = 'causallm'
        model_name = 'huggyllama/llama-7b'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        #self.model = AutoModelForSequenceClassification.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.float16)
        self.task_prompt = 'The task is to determine the relevance of the document to the given query: '
        #task_prompt = 'Provided with a query and a document, the task is to determine the relevance of the document to the given query. Consider the meaning, context, and relevant information to determine the document\'s relevance to the query.'
        self.model = prepare_model_for_int8_training(self.model)
        #config = LoraConfig(
        #        r=8,
        #        lora_alpha=16,
        #        inference_mode=False,
        #        lora_dropout=0.1,
        #        bias="lora_only",
        #        target_modules =  [
        #        "q_proj",
        #        "v_proj",
        #    ],
        #        task_type="CAUSAL_LM",
        #    )
        config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text=self.task_prompt,
    tokenizer_name_or_path=model_name,
)

        self.model = get_peft_model(self.model, config)
        print(self.model.print_trainable_parameters())
        self.true_tokenid = self.tokenizer.encode('true', add_special_tokens=True)[-1]
    def get_scores(self, features, index):
        encoded_input = features['encoded'][index]
        encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}
        if 'labels' in features['encoded'][index]:
            out_raw = self.model(**encoded_input)
            return  {'loss': out_raw.loss}
        else:
            outputs = self.model(input_ids=encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"])
            scores = torch.softmax(outputs.logits[:, -2, self.true_tokenid] / .8, dim=-1)
            return {'scores': scores}




