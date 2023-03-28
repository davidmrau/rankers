from trectools import TrecQrel, TrecRun, TrecEval
import sys

r1_f = sys.argv[1]
qrel = sys.argv[2]
metric = sys.argv[3]
r2_f = sys.argv[4]
#qrel = 'qrels.adhoc.201-250.txt'
#qrel = 'qrels.adhoc.51-200.txt'
print('qrel', qrel)


r1 = TrecRun(r1_f)
r2 = TrecRun(r2_f)



coverage = r1.check_run_coverage(r2, topX=10)
print('how many documents, on average, in the top 10 of r1 were retrieved in the top 10 of r2', round(coverage, 4))

qrels = TrecQrel(qrel)
result_r1 = r1.evaluate_run(qrels, per_query=True)
result_r2 = r2.evaluate_run(qrels, per_query=True)

pvalue = result_r1.compare_with(result_r2, metric=metric)
print('Inspect for statistically significant differences between the two runs for  {metric}  using two-tailed Student t-test. pvalue:', pvalue)


