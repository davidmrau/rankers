
from collections import defaultdict
import sys

inp = sys.argv[1]
res = defaultdict(dict)
for l in open(inp):
    q_id, _, d_id, _, score, _ = l.strip().split('\t')
    res[q_id][d_id] = score


with open(inp + '.sorted.tsv', 'w') as outf:
    for qid, docs in res.items():
        sorted_scores_q = [(did, docs[did]) for did in sorted(docs, key=docs.get, reverse=True)]
        for i, (did, score) in enumerate(sorted_scores_q):
            outf.write(f'{qid}\tQ0\t{did}\t{i+1}\t{score}\teval\n')
    
