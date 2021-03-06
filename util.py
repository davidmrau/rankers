
from bert_cat import BERT_Cat
from splade import Splade
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForSeq2SeqLM, T5Tokenizer, AutoModelForMaskedLM, RobertaTokenizer, BertTokenizer, BertForSequenceClassification, BertConfig
from fairseq.models.roberta import RobertaModel
import torch.nn as nn
import torch
from bert_tf import BERT_TF, BERT_TF_Config
import time 

class MarginMSELoss(nn.Module):
    def __init__(self):
        super(MarginMSELoss, self).__init__()

    def forward(self, scores_pos, scores_neg, label_pos, label_neg):
        """
        A Margin-MSE loss, receiving 2 scores and 2 labels and it computes the MSE of the respective margins.
        All inputs should be tensors of equal size
        """
        loss = torch.mean(torch.pow((scores_pos - scores_neg) - (label_pos - label_neg),2))
        return loss



def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def get_model(model_name, checkpoint=None, **kwargs):
    prepend_type = False
    # instanitae model

    if 'no_pos' == model_name:

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side=kwargs['truncation_side'])
        model = AutoModelForSequenceClassification.from_pretrained('/project/draugpu/experiments_cikm/bert/bz_128_lr_3e-06_no_pos_emb/model_20/')
        encoding = 'cross'
       
        def get_scores(model, features, index, save_hidden_states=False):
            encoded_input = features['encoded_input'][index]
            out_raw = model(**encoded_input.to('cuda'), output_hidden_states=save_hidden_states)
            scores = out_raw.logits[:, 1]
            return_dict = {}
            return_dict['scores'] = scores
            if save_hidden_states:
                return_dict['last_hidden'] = out_raw['hidden_states'][-1][:,0,:]
            return return_dict

    elif 'shuffle' == model_name:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side=kwargs['truncation_side'])
        model = AutoModelForSequenceClassification.from_pretrained('/project/draugpu/experiments_cikm/bert/bz_128_lr_3e-06shuffle/model_30/')
        encoding = 'cross' 
        def get_scores(model, features, index, save_hidden_states=False):
            encoded_input = features['encoded_input'][index]
            out_raw = model(**encoded_input.to('cuda'), output_hidden_states=save_hidden_states)
            scores = out_raw.logits[:, 1]
            return_dict = {}
            return_dict['scores'] = scores
            if save_hidden_states:
                return_dict['last_hidden'] = out_raw['hidden_states'][-1][:,0,:]
            return return_dict

    elif 'sort' == model_name:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side=kwargs['truncation_side'])
        model = torch.load('/project/draugpu/experiments_bert_model/experiments_msmarco/model_bz_64_lr_3e-06_do_0.2_sr__cls_1000fine_tune_shuffle/model.epfinal.pth')
        model = model.module
        model.to('cpu')
        encoding = 'cross' 
        def get_scores(model, features, index, save_hidden_states=False):
            encoded_input = features['encoded_input'][index]
            out_raw = model(**encoded_input.to('cuda'))
            scores = out_raw
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict

    elif 'crossencoder' == model_name:
        model_name = "nboost/pt-bert-base-uncased-msmarco"
        tokenizer = AutoTokenizer.from_pretrained(model_name,truncation_side=kwargs['truncation_side'] )
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        encoding = 'cross'

        def get_scores(model, features, index, save_hidden_states=False):
            encoded_input = features['encoded_input'][index]

            start = time.time()
            out_raw = model(**encoded_input.to('cuda'), output_hidden_states=save_hidden_states)
            finish = (time.time() - start)
            scores = out_raw.logits[:, 1]
            return_dict = {}
            return_dict['scores'] = scores
            return_dict['time'] = finish
            if save_hidden_states:
                return_dict['last_hidden'] = out_raw['hidden_states'][-1][:,0,:]
            return return_dict

    elif 'crossencoder_2' == model_name:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side=kwargs['truncation_side'])
        model = AutoModelForSequenceClassification.from_config(BertConfig(num_hidden_layers=2))
        model_pre = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
        model.bert.embeddings = model_pre.bert.embeddings
        model.bert.encoder.layer[0] = model_pre.bert.encoder.layer[0]
        del model_pre
        encoding = 'cross'

        def get_scores(model, features, index ):
            encoded_input = features['encoded_input'][index]
            out_raw = model(**encoded_input.to('cuda'))
            scores = out_raw.logits[:, 1]
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict

    elif 'electra' == model_name:
        model_name = 'google/electra-small-discriminator'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        encoding = 'cross'

        def get_scores(model, features, index):
            encoded_input = features['encoded_input'][index]
            out_raw = model(**encoded_input.to('cuda'))
            scores = out_raw.logits[:, 1]
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict
            return scores

    elif 'roberta.shuffle' in model_name or 'roberta.base.orig' == model_name:

        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        model = AutoModelForSequenceClassification.from_pretrained(f'models/{model_name}')
        encoding = 'cross'

        def get_scores(model, features, index):
            encoded_input = features['encoded_input'][index]
            out_raw = model(**encoded_input.to('cuda'))
            scores = out_raw.logits[:, 1]
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict
    elif 'roberta.base.nopos' in model_name :
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = AutoModelForSequenceClassification.from_pretrained(f'models/{model_name}')
        encoding = 'cross'

        def get_scores(model, features, index):
            encoded_input = features['encoded_input'][index]
            out_raw = model(**encoded_input.to('cuda'))
            scores = out_raw.logits[:, 1]
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict
        

    elif 'bert' == model_name:
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        encoding = 'cross'

        def get_scores(model, features, index, save_hidden_states=False ):
            return_dict = {}
            encoded_input = features['encoded_input'][index]
            out_raw = model(**encoded_input.to('cuda'), output_hidden_states=save_hidden_states)
            return_dict['scores'] = out_raw.logits[:, 1]
            if save_hidden_states:
                return_dict['last_hidden'] = out_raw['hidden_states'][-1][:,0,:]
            return return_dict


    elif 'bert_tf' == model_name:
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        #model = BERT_TF(BERT_TF_Config())
        #model = BERT_TF()
        encoding = 'cross'

        def get_scores(model, features, index):
            encoded_input = features['encoded_input'][index]
            out_raw = model(**encoded_input.to('cuda'), position_embeds=features['tf_embeds'][index].to('cuda'))
            scores = out_raw.logits[:, 1]
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict


    elif 'minilm12' == model_name:
        model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        encoding = 'cross'

        def get_scores(model, features, index):
            encoded_input = features['encoded_input'][index]
            out_raw = model(**encoded_input.to('cuda'))
            scores = out_raw.logits[:, 0]
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict
    elif 'tinybert' == model_name:
        model_name = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
        tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        encoding = 'cross'


        def get_scores(model, features, index):
            encoded_input = features['encoded_input'][index]
            out_raw = model(**encoded_input.to('cuda'))
            scores = out_raw.logits[:, 0]
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict

    elif 'duobert' == model_name:
        model_name = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side=kwargs['truncation_side'])
        model = AutoModel.from_pretrained(model_name)
        encoding = 'bi'

        def get_scores(model, features, index, save_hidden_states=False):
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][index]

            emb_queries = model(**encoded_queries.to('cuda'))[0][:,0,:].squeeze(0)
            emb_docs = model(**encoded_docs.to('cuda'))[0][:,0,:].squeeze(0)
            scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict

    elif 'contriever' == model_name:
        model_name = 'facebook/contriever-msmarco'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        encoding = 'bi'

        def get_scores(model, features, index):
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][index]
            emb_queries = model(**encoded_queries.to('cuda'))
            emb_docs = model(**encoded_docs.to('cuda'))
            emb_queries_av = mean_pooling(emb_queries[0], encoded_queries['attention_mask'])
            emb_docs_av = mean_pooling(emb_docs[0], encoded_docs['attention_mask'])
            scores = torch.bmm(emb_queries_av.unsqueeze(1), emb_docs_av.unsqueeze(-1)).squeeze()
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict


    elif 'tctcolbert' == model_name:
        model_name = "castorini/tct_colbert-msmarco"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoding = 'bi'
        prepend_type = True

        def get_scores(model, features, index):
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][index]
            emb_queries = model(**encoded_queries.to('cuda'))
            emb_docs = model(**encoded_docs.to('cuda'))
            emb_queries_mean = mean_pooling(emb_queries["last_hidden_state"], encoded_queries['attention_mask'])
            emb_docs_mean = mean_pooling(emb_docs["last_hidden_state"][:, 4:, :], encoded_docs['attention_mask'][:, 4:])
            scores = torch.bmm(emb_queries_mean.unsqueeze(1), emb_docs_mean.unsqueeze(-1)).squeeze()
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict

        encoding = 'bi'


    elif 'splade' == model_name:
        model_name = 'splade_max'
        encoding = 'bi'
        model = Splade(model_name, agg='max')
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def get_scores(model, features, index):
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][index]
            emb_queries = model(**encoded_queries.to('cuda'))
            emb_docs = model(**encoded_docs.to('cuda'))
            scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict

    elif 'monolarge' == model_name:
        model_name = 'castorini/monobert-large-msmarco-finetune-only' 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        encoding = 'bi'

        def get_scores(model, features, index):
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][index]
            emb_queries = model(**encoded_queries.to('cuda')).last_hidden_state[:,0,:]
            emb_docs = model(**encoded_docs.to('cuda')).last_hidden_state[:,0,:]
            #emb_queries_av = mean_pooling(emb_queries, encoded_queries['attention_mask'])
            #emb_docs_av = mean_pooling(emb_docs, encoded_docs['attention_mask'])
            #scores = torch.bmm(emb_queries_av.unsqueeze(1), emb_docs_av.unsqueeze(-1)).squeeze()
            scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict


    elif 'cocondenser' == model_name:
        model_name = 'Luyu/co-condenser-marco-retriever' 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        encoding = 'bi'

        def get_scores(model, features, index):

            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][index]
            emb_queries = model(**encoded_queries.to('cuda')).last_hidden_state[:,0,:]
            emb_docs = model(**encoded_docs.to('cuda')).last_hidden_state[:,0,:]
            #emb_queries_av = mean_pooling(emb_queries, encoded_queries['attention_mask'])
            #emb_docs_av = mean_pooling(emb_docs, encoded_docs['attention_mask'])
            #scores = torch.bmm(emb_queries_av.unsqueeze(1), emb_docs_av.unsqueeze(-1)).squeeze()
            scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict

            return scores

    elif 'tasb' == model_name:
        model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        encoding = 'bi'

        def get_scores(model, features, index):
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][index]
            emb_queries = model(**encoded_queries.to('cuda'))[0][:,0,:].squeeze(0)
            emb_docs = model(**encoded_docs.to('cuda'))[0][:,0,:].squeeze(0)
            scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict

    elif 'distilldot' == model_name:
        model_name = 'sebastian-hofstaetter/distilbert-dot-margin_mse-T2-msmarco'

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        encoding = 'bi'


        def get_scores(model, features, index):
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][index]

            emb_queries = model(**encoded_queries.to('cuda'))[0][:,0,:].squeeze(0)
            emb_docs = model(**encoded_docs.to('cuda'))[0][:,0,:].squeeze(0)
            scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict


    elif 'sentencebert' == model_name:
        model_name = "sentence-transformers/msmarco-bert-base-dot-v5"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        encoding = 'bi'

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


        def get_scores(model, features, index):
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][index]
            emb_queries = model(**encoded_queries.to('cuda'))
            emb_docs = model(**encoded_docs.to('cuda'))
            emb_queries_av = mean_pooling(emb_queries, encoded_queries['attention_mask'])
            emb_docs_av = mean_pooling(emb_docs, encoded_docs['attention_mask'])
            scores = torch.bmm(emb_queries_av.unsqueeze(1), emb_docs_av.unsqueeze(-1)).squeeze()
            return_dict = {}
            return_dict['scores'] = scores
            return return_dict

    if checkpoint != None:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    return model, tokenizer, get_scores, encoding, prepend_type
