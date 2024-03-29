import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import wandb
import os
import json
import torch
from torch.utils.data import DataLoader
from file_interface import File
from data_reader import DataReader, MSMARCO, MsMarcoHardNegatives,MsMarcoHardNegativesCausalLM 
import argparse
from tqdm import tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from models import *
from util import MarginMSELoss, RegWeightScheduler, DistilMarginMSE
from train_model import train_model
from eval_model import eval_model
from encode import encode
from decode import decode

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name defined in model.py: 'Bert', 'BiEncoder', 'Bigbird', 'BowBert', 'Contriever', 'CrossEncoder', 'CrossEncoder2', 'DistilDot', 'DUOBert', 'Electra', 'IDCM', 'LongformerQA', 'Longformer', 'MiniLM12', 'MiniLM6', 'MonoLarge', 'nboostCrossEncoder', 'SentenceBert', 'ShuffleBert', 'SortBert', 'SparseBert', 'SpladeCocondenserEnsembleDistil', 'SpladeCocondenserSelfDistil', 'TinyBert'.")
parser.add_argument("--exp_dir", type=str, help='Base directory where files will be saved to.' )
parser.add_argument("--dataset_test", type=str, required=None, help="Test dataset name defined in dataset.json: 'example', '2019_pass', '2019_doc', '2020_pass', '2020_pass_scores', '2020_doc', '2021_pass', '2021_doc', '2022_doc', 'clueweb', 'robust', 'robust_100_callan', 'robust_100_kmeans'.")
parser.add_argument("--dataset_train", type=str, default=None, help="Train dataset name defined in dataset.json': 'example', 'pass', 'doc', 'doc_tfidf', 'pass_scores'.")
parser.add_argument("--encode", type=str, default=None, help='Path to file to encode. Input Format "qid\tdid\n".')
parser.add_argument("--decode", type=str, default=None, help='Path to file to decode (sparse representations). Input Format "qid\tdid\n".')

parser.add_argument("--num_warmup_steps", type=int, default=0, help='Number of training warmup steps.')


parser.add_argument("--add_to_dir", type=str, default='', help='Will be appended to the default model directory')
parser.add_argument("--no_fp16", action='store_true', help='Disable half precision training.' )
parser.add_argument("--mb_size_test", type=int, default=128, help='Test batch size.')
parser.add_argument("--max_inp_len", type=int, default=512, help='Max. total input length.')
parser.add_argument("--max_q_len", type=int, default=None, help='Max. Query length. ')
parser.add_argument("--mb_size_train", type=int, default=1024, help='Train batch size.')
parser.add_argument("--single_gpu", action='store_true', help='Limit training to a single gpu.')
parser.add_argument("--eval_metric", default='ndcg_cut_10', help='Evaluation Metric.')
parser.add_argument("--learning_rate", type=float, default=0.00002, help='Learning rate for training.')
parser.add_argument("--checkpoint", type=str, default=None, help='Folder of model checkpoint (will be loaded with huggingfaces .from_pretrained)')
parser.add_argument("--truncation_side", type=str, default='right', help='Truncate from left or right', choices=['left', 'right'])
parser.add_argument("--continue_line", type=int, default=0, help='Continue training in triples file from given line')
parser.add_argument("--save_last_hidden", action='store_true', help='Saves last hiden state under exp_dir/model_dir/last_hidden.p')

parser.add_argument("--aloss_scalar_d", type=float, default=0, help='Loss scalar for the auxiliary sparsity loss of document.')
parser.add_argument("--aloss_scalar_q", type=float, default=0, help='Loss scalar for the auxiliary sparsity loss of document.')
parser.add_argument("--aloss_steps", type=float, default=50000, help='Number of steps of the regularization scheduler..')
parser.add_argument("--aloss", action='store_true', help='Using auxilliary sparsity loss.')
parser.add_argument("--tf_embeds", action='store_true', help='[Experimental] Add term frequencies to input embeddings.')
parser.add_argument("--sparse_dim", type=int, default=10000, help='Dimensionality of the sparsity layer.')
parser.add_argument("--num_terms", type=int, default=32, help='Reducing docs to num_terms tokens using the selector.')
parser.add_argument("--training_steps", type=int, default=150000, help='Number of training steps')
parser.add_argument("--accumulation_steps", type=int, default=1, help='Number of gradient accumulation steps')

parser.add_argument("--no_pos_emb", action='store_true', help='[Experimental] Removes the position embedding.')
parser.add_argument("--shuffle", action='store_true', help='[Experimental] Shuffles training and test tokens (after tokenization)')
parser.add_argument("--sort", action='store_true', help='[Experimental] Sortes document tokens in descending order by tokenid. ')
parser.add_argument("--eval_strategy", default='first_p', type=str, help='Evaluation strategy.', choices=['first_p', 'last_p', 'max_p'])
parser.add_argument("--keep_q", action='store_true', help='[Experimental] Remove all but query terms in document.')
parser.add_argument("--drop_q", action='store_true', help='[Experimental] Removes all query terms from document.')
parser.add_argument("--preserve_q", action='store_true', help='[Experimental]')
parser.add_argument("--distil", action='store_true', help='[Experimental]')
parser.add_argument("--rand_passage", action='store_true', help='[Experimental] Select a random passage of length "max_q_len" from entire input.')

parser.add_argument("--eval_every", type=int, default=2500, help='Evaluate every n steps')
args = parser.parse_args()
print(args)
# adjust truncation side dependent on evaluation strategy
if args.eval_strategy == 'last_p':
    truncation_side = 'left'
else:
    truncation_side = 'right'
args.truncation_side = truncation_side

#if args.mse_loss:
#    DATA_FILE_TRAIN = "data/msmarco_ensemble/bert_cat_ensemble_msmarcopassage_ids_train_scores.tsv"


print(vars(args))
# instanitae model
ranker = globals()[args.model](vars(args))

#if checkpoint load from_pretrained
if args.checkpoint:
    ranker.model = ranker.model.from_pretrained(args.checkpoint)


# model to gpu
ranker.model = ranker.model.to('cuda')
# use multiple gpus if available
if torch.cuda.device_count() > 1 and not args.single_gpu:
    ranker.model = torch.nn.DataParallel(ranker.model)

# load dataset paths
dataset = json.loads(open('datasets.json').read())

# determine the name of the model directory

# instantiate Data Reader
if args.dataset_train:
    model_dir = f'{args.exp_dir}/train/{args.dataset_train}_{args.model}_{args.dataset_test}_bz_{args.mb_size_train}_lr_{args.learning_rate}_training_steps_{args.training_steps}_max_inp_len_{args.max_inp_len}'

    docs_file = dataset['train'][args.dataset_train]['docs']
    queries_file = dataset['train'][args.dataset_train]['queries']
    triples = dataset['train'][args.dataset_train]['triples']

    #load file
    queries = File(queries_file, encoded=False)
    docs = File(docs_file, encoded=False)
    if ranker.type == 'causallm':
        qrels_path = dataset['train'][args.dataset_train]['qrels']
        dataset_train = MsMarcoHardNegativesCausalLM(queries, docs, triples, qrels_path, args.max_inp_len, ranker.tokenizer)
        dataloader_train = DataLoader(dataset_train, shuffle=False, batch_size=args.mb_size_train, num_workers=4, collate_fn=dataset_train.collate_fn)
    elif 'distil' in args.dataset_train:
        qrels_path = dataset['train'][args.dataset_train]['qrels']
        dataset_train = MsMarcoHardNegatives(queries, docs, triples, qrels_path, args.max_inp_len, ranker.tokenizer)
        dataloader_train = DataLoader(dataset_train, shuffle=False, batch_size=args.mb_size_train, num_workers=4, collate_fn=dataset_train.collate_fn)

    else:
        dataset_train = DataReader(ranker.tokenizer, ranker.type, triples, 2, True, queries, docs, args.mb_size_train, drop_q=args.drop_q, keep_q=args.keep_q, preserve_q=args.preserve_q, shuffle=args.shuffle, sort=args.sort, max_inp_len=args.max_inp_len, max_q_len=args.max_q_len, tf_embeds=args.tf_embeds, continue_line=args.continue_line)
        dataloader_train = DataLoader(dataset_train, batch_size=None, num_workers=1, pin_memory=False, collate_fn=dataset_train.collate_fn)


# encode 
if args.encode or args.decode:
    if args.encode:
        base = 'encode'
        file_base = args.encode.split('/')[-1]
        encode_file = args.encode
    elif args.decode:
        base = 'decode'
        file_base = args.decode.split('/')[-1]
        encode_file = args.decode
    if not args.checkpoint:
        model_dir = f'{args.exp_dir}/{base}/{args.model}_{file_base}'
    else:
        model_dir = '{args.checkpoint}/{base}_{file_base}'
    dataset = MSMARCO(encode_file, ranker.tokenizer, max_len=args.max_inp_len)
    dataloader_encode = DataLoader(dataset, batch_size=args.mb_size_test, num_workers=4, collate_fn=dataset.collate_fn)

if args.dataset_test:
    # if we are training then just save evaluation to training foler
    if not args.dataset_train:
        model_dir = f'{args.exp_dir}/test/{args.dataset_test}_{args.model}_max_inp_len_{args.max_inp_len}'

    if args.checkpoint:
        model_dir = f'{args.checkpoint}_test/{args.dataset_test}'

#if we are not encoding then we carrying out testing by default
    docs_file = dataset['test'][args.dataset_test]['docs']
    queries_file = dataset['test'][args.dataset_test]['queries']
    trec_run = dataset['test'][args.dataset_test]['trec_run']
    qrels_file = dataset['test'][args.dataset_test]['qrels']
    
    #load file
    queries = File(queries_file, encoded=False)
    # if training and testing docs are the same and they are loaded already don't load them again
    if not args.dataset_train or docs_file != dataset['train'][args.dataset_train]['docs']:
        docs = File(docs_file, encoded=False)

    dataset_test = DataReader(ranker.tokenizer, ranker.type, trec_run, 1, False, queries, docs, args.mb_size_test, drop_q=args.drop_q, keep_q=args.keep_q, preserve_q=args.preserve_q, shuffle=args.shuffle, sort=args.sort, max_inp_len=args.max_inp_len, max_q_len=args.max_q_len, tf_embeds=args.tf_embeds, sliding_window=args.eval_strategy!='first_p' and args.eval_strategy != 'last_p', rand_passage=args.rand_passage)
    dataloader_test = DataLoader(dataset_test, batch_size=None, num_workers=0, pin_memory=False, collate_fn=dataset_test.collate_fn)
# append add_to_dir
model_dir += f'_{args.add_to_dir}'
# print model directory
print('model dir', model_dir)
# create directory
os.makedirs(model_dir, exist_ok=True)
# write parameter flags to model folder
with open(f'{model_dir}/args.json', 'wt') as f:
    json.dump(vars(args), f, indent=4)

# set position embeddings to zero if parameter is passed
if args.no_pos_emb:
    #emb = getattr(model, args.model.split('.')[0]).embeddings.position_embeddings
    if hasattr(ranker.model, 'bert'):
        emb = ranker.model.bert.embeddings.position_embeddings
    else:
        emb = ranker.model.embeddings.position_embeddings

    emb.weight.data = torch.zeros_like(emb.weight.data)
    emb.weight.requires_grad = False
    print('!!Removing positional Embeddings!!!')

optimizer = AdamW(filter(lambda p: p.requires_grad, ranker.model.parameters()), lr=args.learning_rate, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.training_steps)
reg_d = RegWeightScheduler(args.aloss_scalar_d, args.aloss_steps)
reg_q = RegWeightScheduler(args.aloss_scalar_q, args.aloss_steps)




# select losses according to model architecture
if 'cross' in ranker.type or 'cross-selector' == ranker.type:
    criterion = torch.nn.CrossEntropyLoss()
elif 'bi' in ranker.type:
    criterion = torch.nn.MarginRankingLoss(margin=1)
    #logsoftmax = torch.nn.LogSoftmax(dim=1)
elif 'causallm' == ranker.type:
    criterion = None
if args.dataset_train and 'distil' in args.dataset_train:
    criterion = DistilMarginMSE()


wandb.login()
wandb.init( project=args.exp_dir.split('/')[-2], config=args)

if args.dataset_train:
    train_model(ranker, dataloader_train, dataloader_test, qrels_file, criterion, optimizer, scheduler, reg_d, reg_q, model_dir, training_steps=args.training_steps, aloss=args.aloss, fp16=not args.no_fp16, wandb=wandb, accumulation_steps=args.accumulation_steps, eval_every=args.eval_every)
if args.encode:
    encode(ranker, args.encode, dataloader_encode, model_dir)
if args.decode:
    decode(ranker, args.decode, dataloader_encode, model_dir)
if args.dataset_test:
    eval_model(ranker, dataloader_test, qrels_file, model_dir, save_hidden_states=args.save_last_hidden, eval_strategy=args.eval_strategy, eval_metric=args.eval_metric, wandb=wandb)
