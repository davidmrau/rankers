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
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from file_interface import File
from metrics import Trec
from data_reader import DataReader, MSMARCO
import argparse
import gzip
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from util import get_model, MarginMSELoss
from performance_monitor import PerformanceMonitor
parser = argparse.ArgumentParser()
parser.add_argument("--add_to_dir", type=str, default='')
parser.add_argument("--model", type=str, required=True )
parser.add_argument("--out_folder", type=str, required=True )
parser.add_argument("--dataset", type=str, required=True )
parser.add_argument("--mb_size_test", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_inp_len", type=int, default=512)
parser.add_argument("--max_q_len", type=int, default=None)
parser.add_argument("--collection", type=str, default=None)
parser.add_argument("--run", type=str, default=None)
parser.add_argument("--mb_size_train", type=int, default=1024)
parser.add_argument("--single_gpu", action='store_true')
parser.add_argument("--eval_metric", default='ndcg_cut_10')
parser.add_argument("--learning_rate", type=float, default=0.00002)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--truncation_side", type=str, default='right')
parser.add_argument("--continue_epoch", type=int, default=0)
parser.add_argument("--train", action='store_true')
parser.add_argument("--encode", action='store_true')
parser.add_argument("--encode_query", action='store_true')
parser.add_argument("--save_last_hidden", action='store_true')

parser.add_argument("--aloss_scalar", type=float, default=0.00001)
parser.add_argument("--aloss", action='store_true')
parser.add_argument("--tf_embeds", action='store_true')

parser.add_argument("--no_pos_emb", action='store_true')
parser.add_argument("--shuffle", action='store_true')
parser.add_argument("--sort", action='store_true')
parser.add_argument("--eval_strategy", default='first_p', type=str)
parser.add_argument("--keep_q", action='store_true')
parser.add_argument("--drop_q", action='store_true')
parser.add_argument("--preserve_q", action='store_true')
parser.add_argument("--mse_loss", action='store_true')
parser.add_argument("--rand_passage", action='store_true')

args = parser.parse_args()
print(args)
print(args.eval_strategy == 'last_p')
if args.eval_strategy == 'last_p':
    truncation_side = 'left'
else:
    truncation_side = 'right'
args.truncation_side = truncation_side
print(truncation_side)
#experiments_path = 'project/draugpu/experiments_rank_model/'
model_dir = "/".join(args.model.split('/')[:-1])

# train data
if args.mse_loss:
    DATA_FILE_TRAIN = "data/msmarco_ensemble/bert_cat_ensemble_msmarcopassage_ids_train_scores.tsv"
else:
    DATA_FILE_TRAIN = "data/msmarco/qidpidtriples.train.full.tsv"
ID2Q_TRAIN = "data/msmarco/queries.train.tsv" 
ID2DOC_test = None

# test data
if args.dataset == '2020':
    QRELS_TEST = "data/msmarco/2020qrels-pass.txt"
    DATA_FILE_TEST = "data/msmarco/msmarco-passagetest2020-top1000_ranking_results_style_54.tsv" 
    ID2Q_TEST = "data/msmarco/msmarco-test2020-queries.tsv"
    ID2DOC_train = 'data/msmarco/collection.tsv'

if args.dataset == '2020_docs':
    QRELS_TEST = "data/msmarco_docs/2020qrels-docs.txt"
    DATA_FILE_TEST = "data/msmarco_docs/msmarco-doctest2020-top100_judged" 
    ID2Q_TEST = "data/msmarco_docs/msmarco-test2020-queries.tsv"
    ID2DOC_train = 'data/msmarco_docs/msmarco-docs.tsv_test_2020.tsv'

elif args.dataset == '2019_docs_tfidf':
    QRELS_TEST = "data/msmarco_docs/2020qrels-docs.txt"
    DATA_FILE_TEST = "data/msmarco_docs/msmarco-doctest2020-top100_judged" 
    ID2Q_TEST = "data/msmarco_docs/msmarco-test2020-queries.tsv"
    ID2DOC_train = 'data/msmarco_docs/msmarco-docs.in_triples.tfidf_decr.tsv'
    ID2DOC_test = 'data/msmarco_docs/msmarco-docs.tsv_test_2020.tsv_plm_512'
    #ID2DOC_test = 'data/msmarco_docs/msmarco-docs.tsv_test_2020.tsv_tfidf_msmarco_sorted_decreasing_uniq'
    ID2Q_TRAIN = "data/msmarco_docs/msmarco-doctrain-queries.tsv" 
    DATA_FILE_TRAIN = "data/msmarco_docs/triples.tsv"

elif args.dataset == '2019_docs':
    QRELS_TEST = "data/msmarco_docs/2020qrels-docs.txt"
    DATA_FILE_TEST = "data/msmarco_docs/msmarco-doctest2020-top100_judged" 
    ID2Q_TEST = "data/msmarco_docs/msmarco-test2020-queries.tsv"
    ID2DOC_train = 'data/msmarco_docs/msmarco-docs.in_triples.title+body.tsv'
    ID2DOC_test = 'data/msmarco_docs/msmarco-docs.tsv_test_2020.tsv'
    ID2Q_TRAIN = "data/msmarco_docs/msmarco-doctrain-queries.tsv" 
    DATA_FILE_TRAIN = "data/msmarco_docs/triples.tsv"

elif args.dataset == '2021':
    QRELS_TEST = "data/msmarco/2021.qrels.pass.final.txt"
    ID2Q_TEST = "data/msmarco_2/2021_queries.tsv"
    ID2DOC_train = 'data/msmarco_2/passages_provided_top_100.tsv'
elif args.dataset == 'clueweb':
    QRELS_TEST = "data/clue/qrels.web.1-200.txt"
    ID2DOC_train = "data/clue/clueweb09b_docs_cleaned_docs_in_run_spam_filtered_100"
    ID2Q_TEST = "data/clue/topics.web.1-200.txt"
    DATA_FILE_TEST = "data/clue/topics.web.1-200.xml.run.cw09b.bm25.top-100_stemmed_remove_stop_spam_filtered_self_extracted"

elif args.dataset == '2022_docs':
    QRELS_TEST = "data/msmarco_2/2022.qrels.docs.inferred.txt"
    ID2Q_TEST = "data/msmarco_2/2022_queries.tsv"
    DATA_FILE_TEST = "data/msmarco_2/2022_document_top100.txt"
    ID2DOC_train = 'data/msmarco_2/2022_docs_nist.tsv'

elif args.dataset == '2021_docs':
    QRELS_TEST = "data/msmarco_docs/2021.qrels.docs.final.txt"
    ID2Q_TEST = "data/msmarco_docs/2021_queries.tsv"
    DATA_FILE_TEST = "data/msmarco_docs/2021_document_top100_judged.txt"
    ID2DOC_train = 'data/msmarco_docs/msmarco_v2_2021_judged.tsv'

elif args.dataset == '2019':
    QRELS_TEST = "data/msmarco/2019qrels-pass.txt"
    DATA_FILE_TEST = "data/msmarco/msmarco-passagetest2019-top1000_43_ranking_results_style.tsv"
    ID2Q_TEST = "data/msmarco/msmarco-test2019-queries_43.tsv"
    ID2DOC_train = 'data/msmarco/collection.tsv'

elif args.dataset == 'robust':
    QRELS_TEST = "data/robust_test/qrels.robust2004.txt"
    #DATA_FILE_TEST = "data/robust_test/run.robust04.bm25.no_stem.trec"
    DATA_FILE_TEST = "data/robust_test/run.robust04.bm25.no_stem.trec_top_100"
    ID2Q_TEST = 'data/robust_test/04.testset_num_query_lower'
    ID2DOC_train = 'data/robust/robust04_raw_docs.num_query'

elif args.dataset == '100_callan':
    #QRELS_TEST = 'data/distributed_ir/qrels.adhoc.51-200.txt'
    QRELS_TEST = 'data/distributed_ir/qrel.51-150'
    ID2Q_TEST = 'data/distributed_ir/topics.adhoc.51-200.txt.tsv'
    DATA_FILE_TEST = 'data/distributed_ir/run.all_trec.bm25.topics.adhoc.51-200.txt'
    ID2DOC_train = 'data/distributed_ir/trec_123.tsv'

elif args.dataset == 'trec4_kmeans':
    QRELS_TEST = 'data/distributed_ir/qrels.adhoc.201-250.txt'
    ID2Q_TEST = 'data/distributed_ir/topics.adhoc.201-250.txt.desc.tsv'
    ID2DOC_train = 'data/distributed_ir/trec_23.tsv'
    DATA_FILE_TEST = 'data/distributed_ir/run.all_trec.bm25.topics.desc.adhoc.201-250.disks23.txt'

if args.collection != None:
    ID2DOC_train = args.collection
if args.run != None:
    DATA_FILE_TEST = args.run
# instanitae model
model, tokenizer, model_eval_fn, encoding, prepend_type = get_model(args.model, args.checkpoint, truncation_side=args.truncation_side, encoding=args.encode)


# instantiate Data Reader
if args.train:
    id2d_train = File(ID2DOC_train, encoded=False)
    id2q_train = File(ID2Q_TRAIN, encoded=False)
    dataset_train = DataReader(tokenizer, DATA_FILE_TRAIN, 2, True, id2q_train, id2d_train, args.mb_size_train, encoding=encoding, prepend_type=prepend_type, drop_q=args.drop_q, keep_q=args.keep_q, preserve_q=args.preserve_q, shuffle=args.shuffle, sort=args.sort, has_label_scores=args.mse_loss, max_inp_len=args.max_inp_len, max_q_len=args.max_q_len, tf_embeds=args.tf_embeds)
    dataloader_train = DataLoader(dataset_train, batch_size=None, num_workers=1, pin_memory=False, collate_fn=dataset_train.collate_fn)

if args.encode:
    dataset = MSMARCO(args.collection, tokenizer, max_len=args.max_inp_len)
    dataloader_encode = DataLoader(dataset, batch_size=args.mb_size_test, num_workers=1, collate_fn=dataset.collate_fn)
else:
    if ID2DOC_test == None:
        id2d_test = File(ID2DOC_train, encoded=False)
    else:
        id2d_test = File(ID2DOC_test, encoded=False)

    # load data
    id2q_test = File(ID2Q_TEST, encoded=False)


    dataset_test = DataReader(tokenizer, DATA_FILE_TEST, 1, False, id2q_test, id2d_test, args.mb_size_test, encoding=encoding, prepend_type=prepend_type, drop_q=args.drop_q, keep_q=args.keep_q, preserve_q=args.preserve_q, shuffle=args.shuffle, sort=args.sort, max_inp_len=args.max_inp_len, max_q_len=args.max_q_len, tf_embeds=args.tf_embeds, sliding_window=args.eval_strategy!='first_p' and args.eval_strategy != 'last_p', rand_passage=args.rand_passage)
    dataloader_test = DataLoader(dataset_test, batch_size=None, num_workers=0, pin_memory=False, collate_fn=dataset_test.collate_fn)

model = model.to('cuda')

model_dir = f'{args.out_folder}/{args.dataset}_{args.model}'
if not args.train:
    model_dir += args.add_to_dir
    model_dir += '_eval/'
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


def encode(model, tokenizer, collection, model_eval_fn, dataloader, model_dir, eval_strategy='first_p', encode_query=False):


    model.eval()
    
    if encode_query: 
        fname = f'{model_dir}/query_encoded_dict.p'
    else:
        fname = f'{model_dir}/doc_encoded_dict.p'
    emb_dict = {} 
    with torch.no_grad():
        for num_i, features in tqdm(enumerate(dataloader)):

            with torch.inference_mode():
                ids, embs = model_eval_fn(model, features, index=0)
                for id_, emb_ in zip(ids, embs.detach().cpu().numpy()):
                    emb_dict[id_] = emb_
        pickle.dump(emb_dict, open(fname, 'wb'))
                 

        

    #if encode_query: 
    #    f = open(f"{model_dir}_query_encoded.tsv", 'w', encoding='utf-8')
    #else:
    #    f = gzip.open(f"{model_dir}_docs_encoded.tsv.gz", 'wt', encoding='utf-8')
#            # splade decode docs
#
#           weight_range = 5
#            quant_range = 256
#            # decode and print random sample
#            if num_i == 0:
#                idxs = random.sample(range(len(ids)), 1)
#                for idx in idxs:
#                    print(ids[idx], tokenizer.decode(features[1]['input_ids'][idx]))
#
#            for id_, latent_term in zip(ids, latent_terms):
#                if encode_query:
#                    pseudo_str = []
#                    for tok, weight in latent_term.items():
#                        #weight_quanted = int(np.round(weight/weight_range*quant_range))
#                        weight_quanted = int(np.round(weight*100))
#                        pseudo_str += [tok] * weight_quanted
#                    latent_term = " ".join(pseudo_str)
#                    f.write(f"{id_}\t{latent_term}\n")
#                else:
#                    f.write( json.dumps({"id": id_, "vector": latent_term }) + '\n')




def eval_model(model, model_eval_fn, dataloader_test, model_dir,  max_rank='1000', eval_metric='ndcg_cut_10', suffix='', save_hidden_states=False, eval_strategy='first_p'):
    model.eval()
    res_test = {}
    batch_latency = []
    perf_monitor = PerformanceMonitor.get()
    last_hidden = list()
    for num_i, features in tqdm(enumerate(dataloader_test)):
        with torch.inference_mode():
            start_time = time.time()
            out = model_eval_fn(model, features, index=0, save_hidden_states=save_hidden_states)
            timer = time.time()-start_time
            scores = out['scores']

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
                res_test[q][d] = -10000
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


    perf_monitor.log_unique_value("eval_median_batch_pair_latency_ms", np.median(batch_latency))
    perf_monitor.print_summary()
    # RUN TREC_EVAL
    test = Trec(args.eval_metric, 'trec_eval', QRELS_TEST, max_rank, ranking_file_path=f'{model_dir}/model_eval_ranking{suffix}')
    eval_val = test.score(sorted_scores, q_ids)
    print_message('model:{}, {}@{}:{}'.format("eval", eval_metric, max_rank, eval_val))
    if save_hidden_states:
        pickle.dump(last_hidden, open(f'{model_dir}/last_hidden.p', 'wb'))
    return eval_val


def train_model(model, dataloader_train, dataloader_test, model_eval_fn, criterion, optimizer,  model_dir, encoding='cross', num_epochs=40, epoch_size=1000, log_every=10, save_every=1, continue_epoch=0, aloss=False, aloss_scalar=None):
    batch_iterator = iter(dataloader_train)
    total_examples_seen = 0
    model.train()
    for ep_idx in range(continue_epoch, num_epochs+continue_epoch):
        print('epoch', ep_idx)
        # TRAINING
        epoch_loss = 0.0
        mb_idx = 0
        if ep_idx != 0 : 
            aloss_scalar *= aloss_scalar
        print('aloss_scalar', aloss_scalar)
        while mb_idx  <   epoch_size:
            # get train data
            try:
                features = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(dataloader_train)
                continue
            out_1, out_2 = model_eval_fn(model, features, index=0), model_eval_fn(model, features, index=1)
            scores_doc_1, scores_doc_2 = out_1['scores'], out_2['scores']
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
            if aloss:
                l1_loss = (out_1['l1_queries'] + ((out_1['l1_docs'] +  out_2['l1_docs']) / 2) ) * aloss_scalar
                l0_loss = (( out_1['l0_docs'] + out_2['l0_docs']) /2 )
                unused_dims = (( out_1['unused_dims'] + out_2['unused_dims']) /2 )
            else:
                l1_loss = 0
                l0_loss = 0
                unused_dims = 0
            train_loss += l1_loss
            total_examples_seen += scores_doc_1.shape[0]
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()
            if mb_idx % log_every == 0:
                    print(f'MB {mb_idx + 1}/{epoch_size}')
                    print_message('examples:{}, train_loss:{}, l1_loss:{}, l0_loss:{}, unused:{}'.format(total_examples_seen, train_loss, l1_loss, l0_loss, unused_dims))
                    writer.add_scalar('Train/Train Loss', train_loss, total_examples_seen)
            mb_idx += 1

        print_message('epoch:{}, av loss:{}'.format(ep_idx + 1, epoch_loss / (epoch_size) ))

        eval_model(model, model_eval_fn, dataloader_test, model_dir, suffix=ep_idx)

        print('saving_model')

        if ep_idx % save_every == 0:
            model.module.save_pretrained(f'{model_dir}/model_{ep_idx+1}')


if args.train:
    train_model(model, dataloader_train, dataloader_test, model_eval_fn, criterion, optimizer, model_dir, encoding=encoding, num_epochs=args.num_epochs, continue_epoch=args.continue_epoch, aloss_scalar=args.aloss_scalar, aloss=args.aloss)
if args.encode:
    encode(model, tokenizer, args.collection, model_eval_fn, dataloader_encode, model_dir, encode_query=args.encode_query)
else:
    eval_model(model, model_eval_fn, dataloader_test, model_dir, save_hidden_states=args.save_last_hidden, eval_strategy=args.eval_strategy)
