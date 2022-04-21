
import sys
from collections import defaultdict
from metrics import Trec
import numpy as np
from tqdm import tqdm
import random

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


for l in open(bm25_f):
    q, place, d, rank, score, tag = l.strip().split()
    bm25[q][d] = float(score)


for l in open(neural_f):
    q, _, d, rank, score, _ = l.strip().split()
    neural[q][d] = float(score)

for q in bm25:
    scores = []
    for d in bm25[q]:
        scores.append(bm25[q][d])
    min_, max_ = np.min(scores), np.max(scores)
    for d in bm25[q]:
        bm25[q][d] = norm(bm25[q][d], max_, min_)
for q in neural:
    scores = []
    for d in neural[q]:
        scores.append(neural[q][d])
    min_, max_ = np.min(scores), np.max(scores)
    for d in neural[q]:
        neural[q][d] = norm(neural[q][d], max_, min_)


best = defaultdict(lambda : (0,0))
for q in tqdm(bm25):
    for alpha in np.arange(0,1.1, 0.1):
        interpolated = defaultdict(dict)
        #num_sample = len(bm25[q]) if len(bm25[q]) < n else n
        #for d in random.sample(bm25[q].keys(), num_sample):
        for d in bm25[q]:
            score_interpolated = (alpha * bm25[q][d]) + ((1-alpha) * neural[q][d])
            interpolated[q][d] = score_interpolated

        # for each query sort after scores
        docs = interpolated[q]
        sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get, reverse=True)]
        eval_val = test.score([sorted_scores_q], [q])
        if eval_val > best[q][1]:
            best[q] = (alpha, eval_val) 
print(best)
out_f = open(neural_f + 'interpolated_bm25_all_scores.txt', 'w')
opt_interpolated = defaultdict(dict)
for q in tqdm(bm25):
    alpha = best[q][0] 
    for d in bm25[q]:
        score_interpolated = (alpha * bm25[q][d]) + ((1-alpha) * neural[q][d])
        opt_interpolated[q][d] = score_interpolated
        out_f.write(f'{q}\t{d}\t{alpha}\t{score_interpolated}\t{bm25[q][d]}\t{neural[q][d]}\n')





test = Trec(metric, 'trec_eval', qrel, max_rank, ranking_file_path=neural_f + 'interpolated_bm25.trec')
sorted_scores = []
q_ids = []
# for each query sort after scores
for qid, docs in opt_interpolated.items():
    sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get, reverse=True)]
    q_ids.append(qid)
    sorted_scores.append(sorted_scores_q)
eval_val = test.score(sorted_scores, q_ids)


print(eval_val)

