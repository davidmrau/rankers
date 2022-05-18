import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


import os.path
import os
import datetime
import numpy as np
import shutil
import pickle
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from file_interface import File
from metrics import Trec
from data_reader import DataReader
import argparse
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from util import get_model, MarginMSELoss
from performance_monitor import PerformanceMonitor
parser = argparse.ArgumentParser()
parser.add_argument("--add_to_dir", type=str, default='')
parser.add_argument("--model", type=str, required=True )
parser.add_argument("--experiment_folder", type=str, required=True )
parser.add_argument("--dataset", type=str, required=True )
parser.add_argument("--mb_size_test", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=40)
parser.add_argument("--max_inp_len", type=int, default=512)
parser.add_argument("--max_q_len", type=int, default=None)
parser.add_argument("--collection", type=str, default=None)
parser.add_argument("--mb_size_train", type=int, default=1024)
parser.add_argument("--single_gpu", action='store_true')
parser.add_argument("--eval_metric", default='ndcg_cut_10')
parser.add_argument("--learning_rate", type=float, default=0.00001)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--continue_epoch", type=int, default=0)
parser.add_argument("--train", action='store_true')
parser.add_argument("--save_last_hidden", action='store_true')

parser.add_argument("--tf_embeds", action='store_true')

parser.add_argument("--no_pos_emb", action='store_true')
parser.add_argument("--shuffle", action='store_true')
parser.add_argument("--sort", action='store_true')
parser.add_argument("--eval_strategy", default='first_p', type=str)
parser.add_argument("--keep_q", action='store_true')
parser.add_argument("--drop_q", action='store_true')
parser.add_argument("--mse_loss", action='store_true')


args = parser.parse_args()
print(args)


if args.eval_strategy == 'last_p':
    truncation_side = 'left'
else:
    truncation_side = 'right'


#experiments_path = 'project/draugpu/experiments_rank_model/'
model_dir = "/".join(args.model.split('/')[:-1])

# train data
if args.mse_loss:
    DATA_FILE_TRAIN = "data/msmarco_ensemble/bert_cat_ensemble_msmarcopassage_ids_train_scores.tsv"
else:
    DATA_FILE_TRAIN = "data/msmarco/qidpidtriples.train.full.tsv"
ID2Q_TRAIN = "data/msmarco/queries.train.tsv" 


# test data
if args.dataset == '2020':
    QRELS_TEST = "data/msmarco/2020qrels-pass.txt"
    DATA_FILE_TEST = "data/msmarco/msmarco-passagetest2020-top1000_ranking_results_style_54.tsv" 
    ID2Q_TEST = "data/msmarco/msmarco-test2020-queries.tsv"
    ID2DOC = 'data/msmarco/collection.tsv'

if args.dataset == '2020_docs':
    QRELS_TEST = "data/msmarco_docs/2020qrels-docs.txt"
    DATA_FILE_TEST = "data/msmarco_docs/msmarco-doctest2020-top100" 
    ID2Q_TEST = "data/msmarco_docs/msmarco-test2020-queries.tsv"
    ID2DOC = 'data/msmarco_docs/msmarco-docs.tsv_test_2020.tsv'


elif args.dataset == '2021':
    QRELS_TEST = "data/msmarco/2021.qrels.pass.final.txt"
    ID2Q_TEST = "data/msmarco_2/2021_queries.tsv"
    ID2DOC = 'data/msmarco_2/passages_provided_top_100.tsv'

elif args.dataset == '2019':
    QRELS_TEST = "data/msmarco/2019qrels-pass.txt"
    DATA_FILE_TEST = "data/msmarco/msmarco-passagetest2019-top1000_43_ranking_results_style.tsv"
    ID2Q_TEST = "data/msmarco/msmarco-test2019-queries_43.tsv"
    ID2DOC = 'data/msmarco/collection.tsv'

elif args.dataset == 'robust':
    QRELS_TEST = "data/robust_test/qrels.robust2004.txt"
    DATA_FILE_TEST = "data/robust_test/run.robust04.bm25.no_stem.trec"
    ID2Q_TEST = 'data/robust_test/04.testset_num_query_lower'
    ID2DOC = 'data/robust/robust04_raw_docs.num_query'



#DATA_FILE_TEST = "data_interpolation/2021_passage_top100_judged.txt"
#DATA_FILE_TEST = "data_interpolation/run.msmarco-passage.bm25.topics.dl20_54.txt_top_100"
#DATA_FILE_TEST = "data_interpolation/run.msmarco-passage2019.bm25.stem.judged.txt_top_100"

if args.collection != None:
    ID2DOC = args.collection

# instanitae model
model, tokenizer, get_scores, encoding, prepend_type = get_model(args.model, args.checkpoint, truncation_side=truncation_side)

# load data
id2q_test = File(ID2Q_TEST, encoded=False)
id2d = File(ID2DOC, encoded=False)

# instantiate Data Reader
if args.train:
    id2q_train = File(ID2Q_TRAIN, encoded=False)
    dataset_train = DataReader(tokenizer, DATA_FILE_TRAIN, 2, True, id2q_train, id2d, args.mb_size_train, encoding=encoding, prepend_type=prepend_type, drop_q=args.drop_q, keep_q=args.keep_q, shuffle=args.shuffle, sort=args.sort, has_label_scores=args.mse_loss, max_inp_len=args.max_inp_len, max_q_len=args.max_q_len, tf_embeds=args.tf_embeds)
    dataloader_train = DataLoader(dataset_train, batch_size=None, num_workers=1, pin_memory=False, collate_fn=dataset_train.collate_fn)

dataset_test = DataReader(tokenizer, DATA_FILE_TEST, 1, False, id2q_test, id2d, args.mb_size_test, encoding=encoding, prepend_type=prepend_type, drop_q=args.drop_q, keep_q=args.keep_q, shuffle=args.shuffle, sort=args.sort, max_inp_len=args.max_inp_len, max_q_len=args.max_q_len, tf_embeds=args.tf_embeds, sliding_window=args.eval_strategy!='first_p' and args.eval_strategy != 'last_p')
dataloader_test = DataLoader(dataset_test, batch_size=None, num_workers=0, pin_memory=False, collate_fn=dataset_test.collate_fn)

model = model.to('cuda')

model_dir = f'/project/draugpu/{args.experiment_folder}/{args.model}/'

if not args.train:
    model_dir += args.add_to_dir
    model_dir += '_eval'
else:
    model_dir += f'bz_{args.mb_size_train}_lr_{args.learning_rate}'
    model_dir += args.add_to_dir


writer = SummaryWriter(f'{model_dir}/log/')
print('model dir', model_dir)
os.makedirs(model_dir, exist_ok=True)


with open(f'{model_dir}/args.json', 'wt') as f:
    json.dump(vars(args), f, indent=4)

if args.no_pos_emb:
    #emb = getattr(model, args.model.split('.')[0]).embeddings.position_embeddings
    if hasattr(model, 'bert'):
        emb = model.bert.embeddings.position_embeddings
    else:
        emb = model.embeddings.position_embeddings

    emb.weight.data = torch.zeros_like(emb.weight.data)
    emb.weight.requires_grad = False
    print('!!Removing positional Embeddings!!!')

if torch.cuda.device_count() > 1 and not args.single_gpu:
    model = torch.nn.DataParallel(model)

def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)



optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.0001)

if encoding == 'cross' or encoding == 'cross_fairseq':
    criterion = nn.CrossEntropyLoss()
elif encoding == 'bi':
    criterion = nn.MarginRankingLoss(margin=1)

if args.mse_loss:
    criterion = MarginMSELoss()

def eval_model(model, get_scores, dataloader_test, model_dir,  max_rank='1000', eval_metric='ndcg_cut_10', suffix='', save_hidden_states=False, eval_strategy='first_p'):
    model.eval()
    res_test = {}
    batch_latency = []
    perf_monitor = PerformanceMonitor.get()
    last_hidden = list()
    for num_i, features in tqdm(enumerate(dataloader_test)):
        with torch.no_grad():
            start_time = time.time()
            out = get_scores(model, features, index=0, save_hidden_states=save_hidden_states)
            scores = out['scores']
            timer = time.time()-start_time
            if 'time' in out: 
                timer = out['time']
            timer = (timer*1000)/scores.shape[0]

            batch_latency.append(timer)
            if save_hidden_states:
                hidden = out['last_hidden'].detach().cpu().numpy()
                last_hidden.append(hidden)
        batch_num_examples = scores.shape[0]
        # for each example in batch
        for i in range(batch_num_examples):
            q = features['meta'][i][0]
            d = features['meta'][i][1]
            
            if q not in res_test:
                res_test[q] = {}
            if d not in res_test[q]:
                res_test[q][d] = 0
            if eval_strategy == 'first_p' or eval_strategy == 'last_p':
                res_test[q][d] = scores[i].item()
            elif eval_strategy == 'max_p':
                if res_test[q][d] <= scores[i].item():
                    res_test[q][d] = scores[i].item()
            elif eval_strategy == 'sum_p':
                res_test[q][d] += scores[i].item()
    sorted_scores = []
    q_ids = []
    # for each query sort after scores
    for qid, docs in res_test.items():
        sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get, reverse=True)]
        q_ids.append(qid)
        sorted_scores.append(sorted_scores_q)
    perf_monitor.log_unique_value("encoding_gpu_mem",str(torch.cuda.memory_allocated()/float(1e9)) + " GB")
    perf_monitor.log_unique_value("encoding_gpu_mem_max",str(torch.cuda.max_memory_allocated()/float(1e9)) + " GB")


    perf_monitor.log_unique_value("eval_median_batch_pair_latency_ms", np.median(batch_latency)*1000)
    perf_monitor.print_summary()
    # RUN TREC_EVAL
    test = Trec(args.eval_metric, 'trec_eval', QRELS_TEST, max_rank, ranking_file_path=f'{model_dir}/model_eval_ranking{suffix}')
    eval_val = test.score(sorted_scores, q_ids)
    print_message('model:{}, {}@{}:{}'.format("eval", eval_metric, max_rank, eval_val))
    if save_hidden_states:
        pickle.dump(last_hidden, open(f'{model_dir}/last_hidden.p', 'wb'))
    return eval_val


def train_model(model, dataloader_train, dataloader_test, get_scores, criterion, optimizer,  model_dir, encoding='cross', num_epochs=40, epoch_size=1000, log_every=10, save_every=10, continue_epoch=0):
    batch_iterator = iter(dataloader_train)
    total_examples_seen = 0
    model.train()
    for ep_idx in range(continue_epoch, num_epochs+continue_epoch):
        print('epoch', ep_idx)
        # TRAINING
        epoch_loss = 0.0
        mb_idx = 0
        while mb_idx  <   epoch_size:
            # get train data
            try:
                features = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(dataloader_train)
                continue
            scores_doc_1, scores_doc_2 = get_scores(model, features, index=0)['scores'], get_scores(model, features, index=1)['scores']

            optimizer.zero_grad()

            if args.mse_loss:
                train_loss = criterion(scores_doc_1, scores_doc_2, torch.tensor(features['labels_1'], device='cuda'), torch.tensor(features['labels_2'], device='cuda'))
            elif encoding == 'bi':
                train_loss = criterion(scores_doc_1, scores_doc_2, features['labels'].to('cuda'))
            elif 'cross' in encoding:
                scores = torch.stack((scores_doc_2, scores_doc_1),1 )
                train_loss = criterion(scores, features['labels'].long().to('cuda'))
            else:
                raise NotImplementedError()

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

        eval_model(model, get_scores, dataloader_test, model_dir, suffix=ep_idx)

        print('saving_model')

        if ep_idx % save_every == 0:
            model.module.save_pretrained(f'{model_dir}/model_{ep_idx+1}')


if args.train:
    train_model(model, dataloader_train, dataloader_test, get_scores, criterion, optimizer, model_dir, encoding=encoding, num_epochs=args.num_epochs, continue_epoch=args.continue_epoch)
eval_model(model, get_scores, dataloader_test, model_dir, save_hidden_states=args.save_last_hidden, eval_strategy=args.eval_strategy)
