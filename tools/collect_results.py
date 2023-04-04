import os
import sys
import glob
from trectools import TrecQrel, TrecRun, TrecEval
import numpy as np


qrel = sys.argv[1]
folder = sys.argv[2]
depth = 100
#2020_docs_minilm12_rand_passage_128_1_eval
qrels = TrecQrel(qrel)


ndcg_list = list()
map_list = list()
mrr_list = list()
for f in glob.glob(folder):
    print(f)
    run_f = f + '/model_eval_ranking.trec'
    run = TrecRun(run_f)
    te = TrecEval(run, qrels)
    ndcg_10 = te.get_ndcg(depth=10, trec_eval=True)
    map_100 = te.get_map(depth=100, trec_eval=True)
    mrr = te.get_reciprocal_rank(depth=10, trec_eval=True)
    ndcg_list.append(ndcg_10)
    map_list.append(map_100)
    mrr_list.append(mrr)

print(round(np.mean(ndcg_list)*100, 2), ' & ', round(np.mean(mrr_list)*100, 2), ' & ', round(np.mean(map_list)*100, 2))

