import sys

f = sys.argv[1]
k = int(sys.argv[2])

f_out = open(f + f'_top_{k}', 'w')

qid_prev = None
count = 0
for line in open(f):
	qid, template, did, rank, score, tag = line.rstrip().split()

	if qid_prev == None:
		qid_prev = qid

	if qid_prev != qid:
		count = 0

	if count < k:
		f_out.write(line)

	count += 1
	qid_prev = qid
