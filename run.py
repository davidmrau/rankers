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
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from util import get_model, MarginMSELoss, RegWeightScheduler
from performance_monitor import PerformanceMonitor
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help='Model name defined in model.py')
parser.add_argument("--exp_dir", type=str, required=True, help='Base directory where files will be saved to.' )
parser.add_argument("--dataset_test", type=str, required=None, help='Test dataset name defined in dataset.json')
parser.add_argument("--dataset_train", type=str, default=None, help='Train dataset name defined in dataset.json')
parser.add_argument("--encode", type=str, default=None, help='Path to file to encode. Format "qid\tdid\n".')



parser.add_argument("--add_to_dir", type=str, default='', help='Will be appended to the default model directory')
parser.add_argument("--no_fp16", action='store_true', help='Disable half precision training.' )
parser.add_argument("--mb_size_test", type=int, default=128, help='Test batch size.')
parser.add_argument("--num_epochs", type=int, default=10, help='Number of training epochs.')
parser.add_argument("--max_inp_len", type=int, default=512, 'Max. total input length.')
parser.add_argument("--max_q_len", type=int, default=None, help='Max. Query length. ')
parser.add_argument("--run", type=str, default=None)
parser.add_argument("--mb_size_train", type=int, default=1024, help='Train batch size.')
parser.add_argument("--single_gpu", action='store_true', 'Limit training to a single gpu.')
parser.add_argument("--eval_metric", default='ndcg_cut_10', 'Evaluation Metric.')
parser.add_argument("--learning_rate", type=float, default=0.00002, 'Learning rate for training.')
parser.add_argument("--checkpoint", type=str, default=None, 'Folder of model checkpoint (will be loaded with huggingfaces .from_pretrained)')
parser.add_argument("--truncation_side", type=str, default='right', help='Truncate from left or right', choices=['left', 'right'])
parser.add_argument("--continue_line", type=int, default=0, help='Continue training in triples file from given line')
parser.add_argument("--save_last_hidden", action='store_true', 'Saves last hiden state under MODELDIR/last_hidden.p')

parser.add_argument("--aloss_scalar", type=float, default=0.0001, help='Loss scalar for the auxiliary sparsity loss.')
parser.add_argument("--aloss", action='store_true', help='Using auxilliary sparsity loss.')
parser.add_argument("--tf_embeds", action='store_true', help='[Experimental] Add term frequencies to input embeddings.')
parser.add_argument("--sparse_dim", type=int, default=10000, help='Dimensionality of the sparsity layer.')

parser.add_argument("--no_pos_emb", action='store_true', help='[Experimental] Removes the position embedding.')
parser.add_argument("--shuffle", action='store_true', help='[Experimental] Shuffles training and test tokens (after tokenization)')
parser.add_argument("--sort", action='store_true', help='[Experimental] Sortes document tokens in descending order by tokenid. ')
parser.add_argument("--eval_strategy", default='first_p', type=str, help='Evaluation strategy.', choices=['first_p', 'last_p', 'max_p'])
parser.add_argument("--keep_q", action='store_true', help='[Experimental] Remove all but query terms in document.')
parser.add_argument("--drop_q", action='store_true', help='[Experimental] Removes all query terms from document.')
parser.add_argument("--preserve_q", action='store_true', help='[Experimental]')
parser.add_argument("--mse_loss", action='store_true', help='[Experimental]')
parser.add_argument("--rand_passage", action='store_true', help='[Experimental] Select a random passage of length "max_q_len" from entire input.')

args = parser.parse_args()
print(args)
# adjust truncation side dependent on evaluation strategy
if args.eval_strategy == 'last_p':
    truncation_side = 'left'
else:
    truncation_side = 'right'
args.truncation_side = truncation_side
model_dir = "/".join(args.model.split('/')[:-1])

#if args.mse_loss:
#    DATA_FILE_TRAIN = "data/msmarco_ensemble/bert_cat_ensemble_msmarcopassage_ids_train_scores.tsv"


# instanitae model
model, tokenizer, model_eval_fn, encoding, prepend_type = get_model(args.model, args.checkpoint, truncation_side=args.truncation_side, encoding=args.encode, sparse_dim=args.sparse_dim)


dataset = json.loads(open('dataset.json').read())


# instantiate Data Reader
if args.dataset_train:
    model_dir += f'bz_{args.mb_size_train}_lr_{args.learning_rate}'
    model_dir += args.add_to_dir


    docs_file = dataset['train'][args.datatset]['docs']
    queries_file = dataset['train'][args.datatset]['queries']
    triples = dataset['train'][args.datatset]['triples']

    #load file
    queries = File(queries_file, encoded=False)
    docs = File(docs_file, encoded=False)

    dataset_train = DataReader(tokenizer, triples, 2, True, queries, docs, args.mb_size_train, encoding=encoding, prepend_type=prepend_type, drop_q=args.drop_q, keep_q=args.keep_q, preserve_q=args.preserve_q, shuffle=args.shuffle, sort=args.sort, has_label_scores=args.mse_loss, max_inp_len=args.max_inp_len, max_q_len=args.max_q_len, tf_embeds=args.tf_embeds, continue_line=args.continue_line)
    dataloader_train = DataLoader(dataset_train, batch_size=None, num_workers=1, pin_memory=False, collate_fn=dataset_train.collate_fn)


# encode 
if args.encode:
    model_dir += args.add_to_dir
    model_dir += '_encode/'
    encode_file = args.encode
    dataset = MSMARCO(encode_file, tokenizer, max_len=args.max_inp_len)
    dataloader_encode = DataLoader(dataset, batch_size=args.mb_size_test, num_workers=1, collate_fn=dataset.collate_fn)

if args.dataset_test:
    model_dir += args.add_to_dir
    model_dir += '_test/'
#if we are not encoding then we carrying out testing by default
    docs_file = dataset['test'][args.datatset]['docs']
    queries_file = dataset['test'][args.datatset]['queries']
    trec_run = dataset['test'][args.datatset]['trec_run']
    
    #load file
    queries = File(queries_file, encoded=False)
    # if training and testing docs are the same and they are loaded already don't load them again
    if docs_file != dataset['train'][args.dataset]['docs'] and not args.dataset_train:
        docs = File(docs_file, encoded=False)

    dataset_test = DataReader(tokenizer, trec_run, 1, False, queries, docs, args.mb_size_test, encoding=encoding, prepend_type=prepend_type, drop_q=args.drop_q, keep_q=args.keep_q, preserve_q=args.preserve_q, shuffle=args.shuffle, sort=args.sort, max_inp_len=args.max_inp_len, max_q_len=args.max_q_len, tf_embeds=args.tf_embeds, sliding_window=args.eval_strategy!='first_p' and args.eval_strategy != 'last_p', rand_passage=args.rand_passage)
    dataloader_test = DataLoader(dataset_test, batch_size=None, num_workers=0, pin_memory=False, collate_fn=dataset_test.collate_fn)

# determine the name of the model directory
model_dir = f'{args.exp_dir}/{args.dataset}_{args.model}'
# print model directory
print('model dir', model_dir)
# create directory
os.makedirs(model_dir, exist_ok=True)
# write parameter flags to model folder
with open(f'{model_dir}/args.json', 'wt') as f:
    json.dump(vars(args), f, indent=4)
# initialize summary writer
writer = SummaryWriter(f'{model_dir}/log/')

# model to gpu
model = model.to('cuda')
# use multiple gpus if available
if torch.cuda.device_count() > 1 and not args.single_gpu:
    model = torch.nn.DataParallel(model)

# set position embeddings to zero if parameter is passed
if args.no_pos_emb:
    #emb = getattr(model, args.model.split('.')[0]).embeddings.position_embeddings
    if hasattr(model, 'bert'):
        emb = model.bert.embeddings.position_embeddings
    else:
        emb = model.embeddings.position_embeddings

    emb.weight.data = torch.zeros_like(emb.weight.data)
    emb.weight.requires_grad = False
    print('!!Removing positional Embeddings!!!')




optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=6000, num_training_steps=150000)
scaler = torch.cuda.amp.GradScaler(enabled=not args.no_fp16)


def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)


# select losses according to model architecture
if encoding == 'cross' or encoding == 'cross_fairseq':
    criterion = nn.CrossEntropyLoss()
elif encoding == 'bi':
    criterion = nn.MarginRankingLoss(margin=1)
    reg = RegWeightScheduler(args.aloss_scalar, 5000)
    #logsoftmax = torch.nn.LogSoftmax(dim=1)
if args.mse_loss:
    criterion = MarginMSELoss()




def encode(encode_file, model, tokenizer, model_eval_fn, dataloader, model_dir, eval_strategy='first_p'):

    emb_file = encode_file + '.encoded.p'
    model.eval() 
    emb_dict = {} 
    with torch.no_grad():
        for num_i, features in tqdm(enumerate(dataloader)):

            with torch.inference_mode():
                ids, embs = model_eval_fn(model, features, index=0)
                for id_, emb_ in zip(ids, embs.detach().cpu().numpy()):
                    emb_dict[id_] = emb_
        pickle.dump(emb_dict, open(emb_file, 'wb'))
                 

        

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


def train_model(model, dataloader_train, dataloader_test, model_eval_fn, criterion, optimizer,  model_dir, encoding='cross', num_epochs=40, epoch_size=1000, log_every=10, save_every=1, aloss=False, aloss_scalar=None, fp16=True):
    batch_iterator = iter(dataloader_train)
    total_examples_seen = 0
    model.train()
    for ep_idx in range(num_epochs):
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
            with torch.cuda.amp.autocast(enabled=fp16):
                out_1, out_2 = model_eval_fn(model, features, index=0), model_eval_fn(model, features, index=1)
                scores_doc_1, scores_doc_2 = out_1['scores'], out_2['scores']
                optimizer.zero_grad()
                if args.mse_loss:
                    train_loss = criterion(scores_doc_1, scores_doc_2, torch.tensor(features['labels_1'], device='cuda'), torch.tensor(features['labels_2'], device='cuda'))
                elif encoding == 'bi':
                    train_loss = criterion(scores_doc_1, scores_doc_2, features['labels'].to('cuda'))
                    #train_loss = logsoftmax(torch.cat([scores_doc_1.unsqueeze(1), scores_doc_2.unsqueeze(1)], dim=1))
                    #train_loss = torch.mean(-train_loss[0,:])
                    
                elif 'cross' in encoding:
                    scores = torch.stack((scores_doc_2, scores_doc_1),1 )
                    train_loss = criterion(scores, features['labels'].long().to('cuda'))
                else:
                    raise NotImplementedError()
                if aloss:
                    #l1_loss = (out_1['l1_queries'] + ((out_1['l1_docs'] +  out_2['l1_docs']) / 2) ) * aloss_scalar
                    l1_loss = (out_1['l1_queries'] * aloss_scalar).mean() + (((out_1['l1_docs']) * aloss_scalar).mean() +  (out_2['l1_docs'] * aloss_scalar).mean() / 2)
                    l0_loss = (( out_1['l0_docs'] + out_2['l0_docs']) /2 )
                    used_dims = (( out_1['used_dims'] + out_2['used_dims']) /2 )
                    aloss_scalar = reg.step()
                else:
                    l1_loss = 0
                    l0_loss = 0
                    used_dims = 0
                if torch.isnan(train_loss).sum()> 0:
                    raise ValueError('loss contains nans, aborting training')
                    exit()
                train_loss += l1_loss
                total_examples_seen += scores_doc_1.shape[0]
                # scaler.scale(loss) returns scaled losses, before the backward() is called
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_loss += train_loss.item()
                if mb_idx % log_every == 0:
                        print(f'MB {mb_idx + 1}/{epoch_size}')
                        print_message('examples:{}, train_loss:{:.6f}, l1_loss:{:.6f}, l0_loss:{:.2f}, used:{:.2f}'.format(total_examples_seen, train_loss, l1_loss, l0_loss, used_dims))
                        writer.add_scalar('Train/Train Loss', train_loss, total_examples_seen)
                mb_idx += 1

        print_message('epoch:{}, av loss:{}'.format(ep_idx + 1, epoch_loss / (epoch_size) ))

        eval_model(model, model_eval_fn, dataloader_test, model_dir, suffix=ep_idx)

        print('saving_model')

        if ep_idx % save_every == 0:
            model.module.save_pretrained(f'{model_dir}/model_{ep_idx+1}')


if args.dataset_train:
    train_model(model, dataloader_train, dataloader_test, model_eval_fn, criterion, optimizer, model_dir, encoding=encoding, num_epochs=args.num_epochs, aloss_scalar=args.aloss_scalar, aloss=args.aloss, fp16=not args.no_fp16)
if args.encode:
    encode(args.encode, model, tokenizer, model_eval_fn, dataloader_encode, model_dir)
if args.dataset_test:
    eval_model(model, model_eval_fn, dataloader_test, model_dir, save_hidden_states=args.save_last_hidden, eval_strategy=args.eval_strategy)
