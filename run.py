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
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForSeq2SeqLM, T5Tokenizer, AutoModelForMaskedLM
from torch.autograd import Variable
from bert_cat import BERT_Cat
from splade import Splade
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--add_to_dir", type=str, default='')
parser.add_argument("--model", type=str, required=True )
parser.add_argument("--experiment_folder", type=str, required=True )
parser.add_argument("--dataset", type=str, required=True )
parser.add_argument("--mb_size_test", type=int, default=128)
parser.add_argument("--mb_size_train", type=int, default=1024)
parser.add_argument("--single_gpu", action='store_true')
parser.add_argument("--shuffle", action='store_true')
parser.add_argument("--train", action='store_true')
parser.add_argument("--eval_metric", default='ndcg_cut_10')
parser.add_argument("--max_rank", default='1000')
parser.add_argument("--keep_q", action='store_true')
parser.add_argument("--drop_q", action='store_true')
parser.add_argument("--learning_rate", type=float, default=0.00001)


args = parser.parse_args()
print(args)




#experiments_path = 'project/draugpu/experiments_rank_model/'
model_dir = "/".join(args.model.split('/')[:-1])

# train data
DATA_FILE_TRAIN = "data/msmarco/qidpidtriples.train.full.tsv"
ID2Q_TRAIN = "data/msmarco/queries.train.tsv" 

# test data
if args.dataset == '2020':
    QRELS_TEST = "data/msmarco/2020qrels-pass.txt"
    DATA_FILE_TEST = "data/msmarco/msmarco-passagetest2020-top1000_ranking_results_style_54.tsv" 
    ID2Q_TEST = "data/msmarco/msmarco-test2020-queries.tsv"
    ID2DOC = 'data/msmarco/collection.tsv'

elif args.dataset == '2019':
    QRELS_TEST = "data/msmarco/2019qrels-pass.txt"
    DATA_FILE_TEST = "data/msmarco/msmarco-passagetest2019-top1000_43_ranking_results_style.tsv"
    ID2Q_TEST = "data/msmarco/msmarco-test2019-queries_43.tsv"
    ID2DOC = 'data/msmarco/collection.tsv'



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

    def get_scores(model, features, index ):
        encoded_input = features['encoded_input'][index]
        out_raw = model(**encoded_input.to('cuda'))
        scores = out_raw.logits[:, 1]
        return scores


if 'bert' == args.model:
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    bi_encoding = False

    def get_scores(model, features, index):
        encoded_input = features['encoded_input'][index]
        out_raw = model(**encoded_input.to('cuda'))
        scores = out_raw.logits[:, 1]
        return scores

elif 'minilm12' == args.model:
    model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    bi_encoding = False

    def get_scores(model, features, index):
        encoded_input = features['encoded_input'][index]
        out_raw = model(**encoded_input.to('cuda'))
        scores = out_raw.logits[:, 0]
        return scores
elif 'tinybert' == args.model:
    model_name = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    bi_encoding = False


    def get_scores(model, features, index):
        encoded_input = features['encoded_input'][index]
        out_raw = model(**encoded_input.to('cuda'))
        scores = out_raw.logits[:, 0]
        return scores

elif 'contriever' == args.model:
    model_name = 'facebook/contriever-msmarco'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    bi_encoding = True

    def get_scores(model, features, index):
        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
        emb_queries = model(**encoded_queries.to('cuda'))
        emb_docs = model(**encoded_docs.to('cuda'))
        emb_queries_av = mean_pooling(emb_queries[0], encoded_queries['attention_mask'])
        emb_docs_av = mean_pooling(emb_docs[0], encoded_docs['attention_mask'])
        scores = torch.bmm(emb_queries_av.unsqueeze(1), emb_docs_av.unsqueeze(-1)).squeeze()

        return scores 

elif 'tctcolbert' == args.model:
    model_name = "castorini/tct_colbert-msmarco"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bi_encoding = True
    prepend_type = True

    def get_scores(model, features, index):
        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
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

    def get_scores(model, features, index):
        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
        emb_queries = model(**encoded_queries.to('cuda'))
        emb_docs = model(**encoded_docs.to('cuda'))
        scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
        return scores

elif 'monolarge' == args.model:
    model_name = 'castorini/monobert-large-msmarco-finetune-only' 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    bi_encoding = True

    def get_scores(model, features, index):
        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
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

    def get_scores(model, features, index):

        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
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

    def get_scores(model, features, index):
        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]
        emb_queries = model(**encoded_queries.to('cuda'))[0][:,0,:].squeeze(0)
        emb_docs = model(**encoded_docs.to('cuda'))[0][:,0,:].squeeze(0)
        scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
        return scores 

elif 'distilldot' == args.model:
    model_name = 'sebastian-hofstaetter/distilbert-dot-margin_mse-T2-msmarco'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    bi_encoding = True


    def get_scores(model, features, index):
        encoded_queries = features['encoded_queries']
        encoded_docs = features['encoded_docs'][index]

        emb_queries = model(**encoded_queries.to('cuda'))[0][:,0,:].squeeze(0)
        emb_docs = model(**encoded_docs.to('cuda'))[0][:,0,:].squeeze(0)
        scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
        return scores 

elif 'sentencebert' == args.model:
    model_name = "sentence-transformers/msmarco-bert-base-dot-v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    bi_encoding = True

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
        return scores 




# load data
id2q_test = File(ID2Q_TEST, encoded=False)
id2d = File(ID2DOC, encoded=False)

# instantiate Data Reader
if args.train:
    id2q_train = File(ID2Q_TRAIN, encoded=False)
    dataset_train = DataReader(tokenizer, DATA_FILE_TRAIN, 2, True, id2q_train, id2d, args.mb_size_train, bi_encoding=bi_encoding, prepend_type=prepend_type, drop_q=args.drop_q, keep_q=args.keep_q)
    dataloader_train = DataLoader(dataset_train, batch_size=None, num_workers=1, pin_memory=False, collate_fn=dataset_train.collate_fn)

dataset_test = DataReader(tokenizer, DATA_FILE_TEST, 1, False, id2q_test, id2d, args.mb_size_test, bi_encoding=bi_encoding, prepend_type=prepend_type, drop_q=args.drop_q, keep_q=args.keep_q, shuffle=args.shuffle)
dataloader_test = DataLoader(dataset_test, batch_size=None, num_workers=1, pin_memory=False, collate_fn=dataset_test.collate_fn)
model = model.to('cuda')

model_dir = f'/project/draugpu/{args.experiment_folder}/{args.model}/'

if not args.train:
    model_dir += args.add_to_dir
    model_dir += '_eval'
else:
    model_dir += args.add_to_dir
    model_dir += f'bz_{args.mb_size_train}_lr_{args.learning_rate}'


writer = SummaryWriter(f'{model_dir}/log/')
print('model dir', model_dir)
os.makedirs(model_dir, exist_ok=True)

if torch.cuda.device_count() > 1 and not args.single_gpu:
    model = torch.nn.DataParallel(model)

def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)



optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.0001)

if bi_encoding:
	criterion = nn.MarginRankingLoss(margin=1)
else:
	criterion = nn.CrossEntropyLoss()

def eval_model(model, get_scores, dataloader_tes, model_dir,  max_rank='1000', eval_metric='ndcg_cut_10', suffix=''):
    model.eval()
    res_test = {}
    for num_i, features in enumerate(dataloader_test):
        
        with torch.no_grad():
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
            
    sorted_scores = []
    q_ids = []
    # for each query sort after scores
    for qid, docs in res_test.items():
        sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get, reverse=True)]
        q_ids.append(qid)
        sorted_scores.append(sorted_scores_q)

    # RUN TREC_EVAL
    test = Trec(args.eval_metric, 'trec_eval', QRELS_TEST, max_rank, ranking_file_path=f'{model_dir}/model_eval_ranking{suffix}')
    eval_val = test.score(sorted_scores, q_ids)
    writer.add_scalar(f'Test/{eval_metric}', eval_val, suffix)
    print_message('model:{}, {}@{}:{}'.format("eval", eval_metric, max_rank, eval_val))
    return eval_val


def train_model(model, dataloader_train, dataloader_test, get_scores, criterion, optimizer,  model_dir, bi_encoding=False, num_epochs=40, epoch_size=1000, log_every=10, save_every=10):
    batch_iterator = iter(dataloader_train)
    total_examples_seen = 0
    for ep_idx in range(num_epochs):
        # TRAINING
        model.train()
        epoch_loss = 0.0
        mb_idx = 0
        while mb_idx  <   epoch_size:
            # get train data
            try:
                features = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(dataloader_train)
                continue
            optimizer.zero_grad()
            scores_doc_1, scores_doc_2 = get_scores(model, features, index=0), get_scores(model, features, index=1)

            if bi_encoding:
                train_loss = criterion(scores_doc_1, scores_doc_2, features['labels'].to('cuda'))
            else:
                scores = torch.stack((scores_doc_2, scores_doc_1),1 )
                train_loss = criterion(scores, features['labels'].long().to('cuda'))

            total_examples_seen += scores_doc_1.shape[0]
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()
            if mb_idx % log_every == 0:
                    print(f'MB {mb_idx + 1}/{epoch_size}')
                    print_message('examples:{}, train_loss:{}'.format(total_examples_seen, train_loss))
                    writer.add_scalar('Train/Train Loss', train_loss, total_examples_seen)
            mb_idx += 1

        print_message('epoch:{}, av loss:{}'.format(ep_idx + 1, epoch_loss / (epoch_size) ))

        eval_model(model, get_scores, dataloader_test, max_rank, eval_metric, model_dir, suffix=idx)

        print('saving_model')
        if save_every:
            model.module.save_pretrained(f'{model_dir}/model_{ep_idx+1}')


if args.train:
    train_model(model, dataloader_train, dataloader_test, get_scores, criterion, optimizer, model_dir, bi_encoding=bi_encoding)
eval_model(model, get_scores, dataloader_test, model_dir)
