
import sys
from collections import defaultdict
from metrics import Trec
import numpy as np
from tqdm import tqdm
import random 

from itertools import product


bm25_f = sys.argv[1]
neural_f = sys.argv[2]
qrel = sys.argv[3]

bm25 = defaultdict(dict)
neural = defaultdict(dict)


def norm(data, max_, min_):
    return ((data - min_)/ (max_-min_))



max_rank = 10
metric = 'ndcg_cut_10'
test = Trec(metric, 'trec_eval', qrel, max_rank, ranking_file_path=f'/tmp/ranking.trec')

def get_alphas(num_d):
    alphas = [0,1]
    return product(alphas, repeat=num_d)



#read
for l in open(bm25_f):
    q, place, d, rank, score, tag = l.strip().split()
    bm25[q][d] = float(score)


#read
for l in open(neural_f):
    q, _, d, rank, score, _ = l.strip().split()
    neural[q][d] = float(score)

#normalize
for q in bm25:
    scores = []
    for d in bm25[q]:
        scores.append(bm25[q][d])
    min_, max_ = np.min(scores), np.max(scores)
    for d in bm25[q]:
        bm25[q][d] = norm(bm25[q][d], max_, min_)


#normalize
for q in neural:
    scores = []
    for d in neural[q]:
        scores.append(neural[q][d])
    min_, max_ = np.min(scores), np.max(scores)
    for d in neural[q]:
        neural[q][d] = norm(neural[q][d], max_, min_)

best = defaultdict(lambda: ([], 0))
for q in tqdm(bm25):
    #interpolated = defaultdict(dict)
    interpolated = neural
    n = 5
    alphas_all = get_alphas(n)

    #for alpha, d in zip(alphas, bm25[q]):
    for alphas in alphas_all:
        #for alpha, d in zip(alphas, random.sample(bm25[q].keys()], n)):
        #for alpha, d in zip(alphas, list(bm25[q].keys())[:10]):
        #    score_interpolated = alpha * bm25[q][d] + ((1-alpha) * neural[q][d])
        #    interpolated[q][d] = score_interpolated

        sorted_scores = []
        q_ids = []
        # for each query sort after scores
        for qid, docs in interpolated.items():
            sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get, reverse=True)]
            q_ids.append(qid)
            sorted_scores.append(sorted_scores_q)

        eval_val = test.score(sorted_scores, q_ids)
        print(q, alphas, eval_val)
        if eval_val > best[q][1]:
            best[q] = (alphas, eval_val)
    print('final', q, best[q])

print(best)
out_f = open(neural_f + 'interpolated_per_doc_bm25.txt', 'w')
opt_interpolated = defaultdict(dict)
for q in tqdm(bm25):
    for d, alpha in zip(bm25[q], best[q][0]):
        score_interpolated = alpha * bm25[q][d] + ((1-alpha) * neural[q][d])
        opt_interpolated[q][d] = score_interpolated
        out_f.write(f'{q}\t{d}\t{alpha}\t{score_interpolated}\t{bm25[q][d]}\t{neural[q][d]}\n')

sorted_scores = []
q_ids = []
# for each query sort after scores
for qid, docs in opt_interpolated.items():
    sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get, reverse=True)]
    q_ids.append(qid)
    sorted_scores.append(sorted_scores_q)

eval_val = test.score(sorted_scores, q_ids)


print(eval_val)

