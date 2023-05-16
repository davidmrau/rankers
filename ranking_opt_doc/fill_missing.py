import sys



ref_file = sys.argv[1]
run_file = sys.argv[2]


ref = {}
for l in open(ref_file):
    qid, _, did, _, score, _ = l.strip().split()
    ref[f'{qid} {did}'] = score
    

run = {}
for l in open(run_file):
    qid, _, did, _, score, _ = l.strip().split()
    run[f'{qid} {did}'] = score


count = 0
for k in ref:
    if k not in run:
        count += 1
        run[k] = ref[k]
assert len(run) == len(ref)
print('filled:', count)
