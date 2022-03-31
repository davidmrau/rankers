import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


import os.path
import os
import datetime
import numpy as np
import shutil
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from file_interface import File
from metrics import Trec
from data_reader import DataReader
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForSeq2SeqLM, T5Tokenizer, AutoModelForMaskedLM

from bert_cat import BERT_Cat
from splade import Splade

parser = argparse.ArgumentParser()
parser.add_argument("--add_to_dir", type=str, default='')
parser.add_argument("--model", type=str, required=True )
parser.add_argument("--experiment_folder", type=str, required=True )
parser.add_argument("--dataset", type=str, required=True )
parser.add_argument("--mb_size", type=int, default=1024)
parser.add_argument("--single_gpu", action='store_true')
parser.add_argument("--eval_metric", default='ndcg_cut_10')
parser.add_argument("--max_rank", default='1000')
parser.add_argument("--max_doc_len", default=512, type=int)
parser.add_argument("--max_query_len", default=None, type=int)


args = parser.parse_args()
print(args)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

MAX_QUERY_TERMS = args.max_query_len
MAX_DOC_TERMS = args.max_doc_len
MB_SIZE = args.mb_size



#experiments_path = 'project/draugpu/experiments_rank_model/'
MODEL_DIR = "/".join(args.model.split('/')[:-1])


if args.dataset == '2020':
#2020
    QRELS_TEST = "data/msmarco/2020qrels-pass.txt"
    DATA_FILE_TEST = "data/msmarco/msmarco-passagetest2020-top1000_ranking_results_style_54.tsv" 
    ID2Q_TEST = "data/msmarco/msmarco-test2020-queries.tsv"
    ID2DOC = 'data/msmarco/collection.tsv'

elif args.dataset == '2019':
#2019
    QRELS_TEST = "data/msmarco/2019qrels-pass.txt"
    DATA_FILE_TEST = "data/msmarco/msmarco-passagetest2019-top1000_43_ranking_results_style.tsv"
    ID2Q_TEST = "data/msmarco/msmarco-test2019-queries_43.tsv"
    ID2DOC = 'data/msmarco/collection.tsv'


# load id2q 
id2q_test = File(ID2Q_TEST, encoded=False)
id2d = File(ID2DOC, encoded=False)
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


prepend_type = False
# instanitae model

if 'crossencoder' == args.model:
    model_name = "nboost/pt-bert-base-uncased-msmarco"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    bi_encoding = False

    def get_scores(model, features):
        with torch.no_grad():
            encoded_input = features['encoded_input'][0]
            out_raw = model(**encoded_input.to('cuda'))
            scores = out_raw.logits[:, 1]
            return scores

elif 'minilm12' == args.model:
    model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    bi_encoding = False

    def get_scores(model, features):
        with torch.no_grad():
            encoded_input = features['encoded_input'][0]
            out_raw = model(**encoded_input.to('cuda'))
            scores = out_raw.logits[:, 0]
            return scores
elif 'tinybert' == args.model:
    model_name = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    bi_encoding = False


    def get_scores(model, features):
        with torch.no_grad():
            encoded_input = features['encoded_input'][0]
            out_raw = model(**encoded_input.to('cuda'))
            scores = out_raw.logits[:, 0]
            return scores

elif 'contriever' == args.model:
    model_name = 'facebook/contriever-msmarco'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    bi_encoding = True

    def get_scores(model, features):
        with torch.no_grad():
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][0]
            emb_queries = model(**encoded_queries.to('cuda'))
            emb_docs = model(**encoded_docs.to('cuda'))
            emb_queries_av = mean_pooling(emb_queries[0], encoded_queries['attention_mask'])
            emb_docs_av = mean_pooling(emb_docs[0], encoded_docs['attention_mask'])
            scores = torch.bmm(emb_queries_av.unsqueeze(1), emb_docs_av.unsqueeze(-1)).squeeze()

            return scores 

elif 'duot5' == args.model:
    #castorini/monot5-base-msmarco
    model_name = 'castorini/duot5-base-msmarco' 
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    bi_encoding = False
    def get_scores(model, features):
        with torch.no_grad():
            encoded_input = features['encoded_input'][0]
            out_raw = model(**encoded_input.to('cuda'))
            print(scores.shape)
            scores = out_raw.logits[:, 1]
            return scores


elif 'tctcolbert' == args.model:
    model_name = "castorini/tct_colbert-msmarco"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bi_encoding = True
    prepend_type = True

    def get_scores(model, features):
        with torch.no_grad():
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][0]
            emb_queries = model(**encoded_queries.to('cuda'))
            emb_docs = model(**encoded_docs.to('cuda'))
            emb_queries_mean = mean_pooling(emb_queries["last_hidden_state"], encoded_queries['attention_mask'])
            emb_docs_mean = mean_pooling(emb_docs["last_hidden_state"][:, 4:, :], encoded_docs['attention_mask'][:, 4:])
            scores = torch.bmm(emb_queries_mean.unsqueeze(1), emb_docs_mean.unsqueeze(-1)).squeeze()
            return scores 


    encoding_fn = bi_encoding

elif 'splade' == args.model:
    model_name = 'splade_max'
    bi_encoding = True
    model = Splade(model_name, agg='max')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_scores(model, features):
        with torch.no_grad():
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][0]
            emb_queries = model(**encoded_queries.to('cuda'))
            emb_docs = model(**encoded_docs.to('cuda'))
            scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
            return scores

elif 'monolarge' == args.model:
    model_name = 'castorini/monobert-large-msmarco-finetune-only' 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    bi_encoding = True

    def get_scores(model, features):
        with torch.no_grad():
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][0]
            emb_queries = model(**encoded_queries.to('cuda')).last_hidden_state[:,0,:]
            emb_docs = model(**encoded_docs.to('cuda')).last_hidden_state[:,0,:]
            #emb_queries_av = mean_pooling(emb_queries, encoded_queries['attention_mask'])
            #emb_docs_av = mean_pooling(emb_docs, encoded_docs['attention_mask'])
            #scores = torch.bmm(emb_queries_av.unsqueeze(1), emb_docs_av.unsqueeze(-1)).squeeze()
            scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
            return scores 

elif 'cocondenser' == args.model:
    model_name = 'Luyu/co-condenser-marco-retriever' 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    bi_encoding = True

    def get_scores(model, features):
        with torch.no_grad():

            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][0]
            emb_queries = model(**encoded_queries.to('cuda')).last_hidden_state[:,0,:]
            emb_docs = model(**encoded_docs.to('cuda')).last_hidden_state[:,0,:]
            #emb_queries_av = mean_pooling(emb_queries, encoded_queries['attention_mask'])
            #emb_docs_av = mean_pooling(emb_docs, encoded_docs['attention_mask'])
            #scores = torch.bmm(emb_queries_av.unsqueeze(1), emb_docs_av.unsqueeze(-1)).squeeze()
            scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()

            return scores

elif 'tasb' == args.model:
    model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    bi_encoding = True

    def get_scores(model, features):
        with torch.no_grad():
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][0]
            emb_queries = model(**encoded_queries.to('cuda'))[0][:,0,:].squeeze(0)
            emb_docs = model(**encoded_docs.to('cuda'))[0][:,0,:].squeeze(0)
            scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
            return scores 

elif 'distilldot' == args.model:
    model_name = 'sebastian-hofstaetter/distilbert-dot-margin_mse-T2-msmarco'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    bi_encoding = True


    def get_scores(model, features):
        with torch.no_grad():
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][0]

            emb_queries = model(**encoded_queries.to('cuda'))[0][:,0,:].squeeze(0)
            emb_docs = model(**encoded_docs.to('cuda'))[0][:,0,:].squeeze(0)
            scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
            return scores 

elif 'sentencebert' == args.model:
    model_name = "sentence-transformers/msmarco-bert-base-dot-v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    bi_encoding = True

    def get_scores(model, features):
        with torch.no_grad():
            encoded_queries = features['encoded_queries']
            encoded_docs = features['encoded_docs'][0]
            emb_queries = model(**encoded_queries.to('cuda'))
            emb_docs = model(**encoded_docs.to('cuda'))
            emb_queries_av = mean_pooling(emb_queries, encoded_queries['attention_mask'])
            emb_docs_av = mean_pooling(emb_docs, encoded_docs['attention_mask'])
            scores = torch.bmm(emb_queries_av.unsqueeze(1), emb_docs_av.unsqueeze(-1)).squeeze()
            return scores 



model.eval()
model = model.to(DEVICE)
if torch.cuda.device_count() > 1 and not args.single_gpu:
    model = torch.nn.DataParallel(model)

MODEL_DIR = f'/project/draugpu/{args.experiment_folder}/{args.model}/'
MODEL_DIR += args.add_to_dir
print('model dir', MODEL_DIR)
os.makedirs(MODEL_DIR, exist_ok=True)


# instantiate Data Reader
dataset_test = DataReader(tokenizer, DATA_FILE_TEST, 1, False, id2q_test, id2d, MB_SIZE, bi_encoding=bi_encoding, prepend_type=prepend_type)
dataloader_test = DataLoader(dataset_test, batch_size=None, num_workers=1, pin_memory=False, collate_fn=dataset_test.collate_fn)


def drop_query_words(self, batch_input, n=None):
        for i, inp in enumerate(batch_input['input_ids']):
                query, query_end = set(), False
                for j, t in enumerate(inp):
                        t = t.tolist()
                        if t == 0:
                                break
                        elif t == 102:
                                query_end = True
                                continue
                        if not query_end:
                                query.add(t)
                                
                        if query_end and (t in (query)):
                                batch_input['input_ids'][i][j] = 103
        return batch_input

        
def keep_query_words(self, batch_input, n=None):
        for i, inp in enumerate(batch_input['input_ids']):
                query, query_end = set(), False
                for j, t in enumerate(inp):
                        t = t.tolist()
                        if t == 0:
                                break
                        elif t == 102:
                                query_end = True
                                continue
                        if not query_end:
                                query.add(t)
                        if query_end and (t not in (query)):
                                batch_input['input_ids'][i][j] = 103
        return batch_input

def prepend(sent_list, prepend):
    return [prepend + s for s in sent_list]


res_test = {}
for num_i, features in enumerate(dataloader_test):
    out = get_scores(model, features)
    batch_num_examples = len(features['meta'])
    # for each example in batch
    for i in range(batch_num_examples):
        q = features['meta'][i][0]
        d = features['meta'][i][1]
        
        if q not in res_test:
            res_test[q] = {}
        if d not in res_test[q]:
            res_test[q][d] = 0
        res_test[q][d] += out[i].item()


        
def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)

sorted_scores = []
q_ids = []
# for each query sort after scores
for qid, docs in res_test.items():
    sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get, reverse=True)]
    q_ids.append(qid)
    sorted_scores.append(sorted_scores_q)


# RUN TREC_EVAL
test = Trec(args.eval_metric, 'trec_eval', QRELS_TEST, args.max_rank, ranking_file_path=f'{MODEL_DIR}/model_eval_ranking')

eval_val = test.score(sorted_scores, q_ids)
print_message('model:{}, {}@{}:{}'.format("eval", args.eval_metric, args.max_rank, eval_val))


