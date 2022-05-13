# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert RoBERTa checkpoint."""


import argparse
import pathlib

import fairseq
import torch
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
from fairseq.modules import TransformerSentenceEncoderLayer
from packaging import version

from transformers import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification
from transformers.models.bert.modeling_bert import (
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
)
from torch import nn
from transformers.utils import logging


if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def convert_roberta_checkpoint_to_pytorch(
    roberta_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool
):
    """
    Copy/paste/tweak roberta's weights to our BERT structure.
    """
    #roberta = FairseqRobertaModel.from_pretrained(roberta_checkpoint_path, checkpoint_file='model.pt')
    roberta = torch.load(roberta_checkpoint_path)
    #roberta.eval()  # disable dropout
    #roberta_sent_encoder = roberta.model.encoder.sentence_encoder
    roberta_sent_encoder = roberta['model']
    config = RobertaConfig(
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=514,
        type_vocab_size=2,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
    )
    if classification_head:
        config.num_labels = roberta.model.classification_heads["mnli"].out_proj.weight.shape[0]
    print("Our BERT config:", config)

    model = RobertaForSequenceClassification(config) if classification_head else RobertaForMaskedLM(config)
    model.eval()

    # Now let's copy all the weights.
    # Embeddings
    model.roberta.embeddings.word_embeddings.weight.data = roberta_sent_encoder['encoder.sentence_encoder.embed_tokens.weight'].data

    # position embeddings
    #model.roberta.embeddings.position_embeddings.weight.data = nn.Embedding(2, model.config.hidden_size)

    #model.roberta.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros((2, config.hidden_size)) # just zero them out b/c RoBERTa doesn't use them.
    model.roberta.embeddings.LayerNorm.weight.data = roberta_sent_encoder['encoder.sentence_encoder.emb_layer_norm.weight'].data
    model.roberta.embeddings.LayerNorm.bias.data = roberta_sent_encoder['encoder.sentence_encoder.emb_layer_norm.bias'].data
    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        #layer: BertLayer = model.roberta.encoder.layer[i]
        #roberta_layer: TransformerSentenceEncoderLayer = roberta_sent_encoder.layers[i]

        layer: BertLayer = model.roberta.encoder.layer[i]
        roberta_layer = 'encoder.sentence_encoder.layers.{}'.format(i)
        
        # self attention
        self_attn: BertSelfAttention = layer.attention.self
        assert (
           roberta_sent_encoder[f'{roberta_layer}.self_attn.k_proj.weight'].data.shape
            == self_attn.query.weight.data.shape
            == roberta_sent_encoder[f'{roberta_layer}.self_attn.v_proj.weight'].data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )

        self_attn.query.weight.data = roberta_sent_encoder[f'{roberta_layer}.self_attn.q_proj.weight'].data
        self_attn.query.bias.data = roberta_sent_encoder[f'{roberta_layer}.self_attn.q_proj.bias'].data
        self_attn.key.weight.data =roberta_sent_encoder[f'{roberta_layer}.self_attn.k_proj.weight'].data
        self_attn.key.bias.data = roberta_sent_encoder[f'{roberta_layer}.self_attn.k_proj.bias'].data
        self_attn.value.weight.data = roberta_sent_encoder[f'{roberta_layer}.self_attn.v_proj.weight'].data
        self_attn.value.bias.data = roberta_sent_encoder[f'{roberta_layer}.self_attn.v_proj.bias'].data

        # self-attention output
        self_output: BertSelfOutput = layer.attention.output
        assert self_output.dense.weight.shape == roberta_sent_encoder[f'{roberta_layer}.self_attn.out_proj.weight'].shape
        self_output.dense.weight.data = roberta_sent_encoder[f'{roberta_layer}.self_attn.out_proj.weight'].data
        self_output.dense.bias.data = roberta_sent_encoder[f'{roberta_layer}.self_attn.out_proj.bias'].data
        self_output.LayerNorm.weight.data = roberta_sent_encoder[f'{roberta_layer}.self_attn_layer_norm.weight'].data
        self_output.LayerNorm.bias.data = roberta_sent_encoder[f'{roberta_layer}.self_attn_layer_norm.bias'].data

        # intermediate
        intermediate: BertIntermediate = layer.intermediate
        assert intermediate.dense.weight.shape == roberta_sent_encoder[f'{roberta_layer}.fc1.weight'].shape
        intermediate.dense.weight.data = roberta_sent_encoder[f'{roberta_layer}.fc1.weight'].data
        intermediate.dense.bias.data = roberta_sent_encoder[f'{roberta_layer}.fc1.bias'].data
        # output
        bert_output: BertOutput = layer.output
        assert bert_output.dense.weight.shape == roberta_sent_encoder[f'{roberta_layer}.fc2.weight'].shape
        bert_output.dense.weight.data =  roberta_sent_encoder[f'{roberta_layer}.fc2.weight'].data
        bert_output.dense.bias.data =  roberta_sent_encoder[f'{roberta_layer}.fc2.bias'].data
        bert_output.LayerNorm.weight.data =  roberta_sent_encoder[f'{roberta_layer}.final_layer_norm.weight'].data
        bert_output.LayerNorm.bias.data =  roberta_sent_encoder[f'{roberta_layer}.final_layer_norm.bias'].data
        # end of layer

    #if classification_head:
    #    model.classifier.dense.weight = roberta.model.classification_heads["mnli"].dense.weight
    #    model.classifier.dense.bias = roberta.model.classification_heads["mnli"].dense.bias
    #    model.classifier.out_proj.weight = roberta.model.classification_heads["mnli"].out_proj.weight
    #    model.classifier.out_proj.bias = roberta.model.classification_heads["mnli"].out_proj.bias
    #else:
    #    # LM Head
    #    model.lm_head.dense.weight = roberta.model.encoder.lm_head.dense.weight
    #    model.lm_head.dense.bias = roberta.model.encoder.lm_head.dense.bias
    #    model.lm_head.layer_norm.weight = roberta.model.encoder.lm_head.layer_norm.weight
    #    model.lm_head.layer_norm.bias = roberta.model.encoder.lm_head.layer_norm.bias
    #    model.lm_head.decoder.weight = roberta.model.encoder.lm_head.weight
    #    model.lm_head.decoder.bias = roberta.model.encoder.lm_head.bias

    # Let's check that we get the same results.
    #input_ids: torch.Tensor = roberta.encode(SAMPLE_TEXT).unsqueeze(0)  # batch of size 1

    #our_output = model(input_ids)[0]
    #if classification_head:
     #   their_output = roberta.model.classification_heads["mnli"](roberta.extract_features(input_ids))
    #else:
    #    their_output = roberta.model(input_ids)[0]
    #print(our_output.shape, their_output.shape)
    #max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    #print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    #success = torch.allclose(our_output, their_output, atol=1e-3)
    #print("Do both models output the same tensors?", "🔥" if success else "💩")
    #if not success:
    #    raise Exception("Something went wRoNg")

    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--roberta_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    args = parser.parse_args()
    convert_roberta_checkpoint_to_pytorch(
        args.roberta_checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
