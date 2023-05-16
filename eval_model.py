from performance_monitor import PerformanceMonitor
import torch
from tqdm import tqdm
from metrics import Trec
import time 
import numpy as np
import pickle
from util import print_message

def eval_model(ranker, dataloader_test, qrels_file, model_dir,  max_rank='1000', eval_metric='ndcg_cut_10', suffix='', save_hidden_states=False, eval_strategy='first_p'):
    ranker.model.eval()
    res_test = {}
    batch_latency = []
    perf_monitor = PerformanceMonitor.get()
    last_hidden = list()
    for num_i, features in tqdm(enumerate(dataloader_test)):
        with torch.inference_mode():
            start_time = time.time()
            out = ranker.get_scores(features, index=0)
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
    test = Trec(eval_metric, 'trec_eval', qrels_file, max_rank, ranking_file_path=f'{model_dir}/ranking{suffix}')
    eval_val = test.score(sorted_scores, q_ids)
    print_message('model:{}, {}@{}:{}'.format("eval", eval_metric, max_rank, eval_val))
    if save_hidden_states:
        pickle.dump(last_hidden, open(f'{model_dir}/last_hidden.p', 'wb'))
    return eval_val
