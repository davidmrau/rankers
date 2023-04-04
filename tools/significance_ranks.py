from scipy.stats import ttest_rel
import sys

from collections import defaultdict

run_f = sys.argv[1]
ref_f = sys.argv[2]
metric = sys.argv[3]

run = {}

for l in open(run_f):
    metric_, query, score = l.strip().split('\t')
    metric_, query, score = metric_.strip(), query.strip(), score.strip()
    if metric_ == metric and query != 'all':
        run[query] = float(score)

ref = {}

for l in open(ref_f):
    metric_, query, score = l.strip().split('\t')
    metric_, query, score = metric_.strip(), query.strip(), score.strip()
    if metric_ == metric and query != 'all':
        ref[query] = float(score)


run_list = list()
ref_list = list()
for k in run:
    if k not in run:
        pass
        #print(k, 'not in run')
    elif k not in ref:
        pass
        #print(k, 'not in ref')
    else:
        ref_list.append(ref[k])
        run_list.append(run[k])

assert len(run_list) == len(ref_list)
pvalue = ttest_rel(run_list, ref_list).pvalue
if pvalue < 0.05:
    print('\t - significant')
    
