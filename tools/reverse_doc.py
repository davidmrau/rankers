
import sys
from transformers import BasicTokenizer
from tqdm import tqdm
t = BasicTokenizer()
f_in = sys.argv[1]
f_out = open(f_in+'_reversed', 'w', encoding='utf-8')

for l in tqdm(open(f_in, encoding='utf-8')):
    id_, text = l.strip().split('\t')
    tokenized = t.tokenize(text)
    tokenized_reversed = ' '.join(tokenized[::-1])
    f_out.write(f'{id_}\t{tokenized_reversed}\n')
