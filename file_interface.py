import numpy as np
import gzip
from tqdm import tqdm
import subprocess
import pickle
import os 
from nltk.corpus import stopwords
import subprocess
import json 

class FileJson():
    def __init__(self, fname, field='plm_bert_tokenized', stopwords_to_zero=True, encoded=True):
        self.file = {}
        self.field = field
        self.stopwords_to_zero = stopwords_to_zero
        self.stop_words = set(stopwords.words('english'))
        count_empty_docs = 0
        if os.path.isfile(f'{fname}.p'):
            print(f'Loading cached file: {fname}.p')
            self.file = pickle.load(open(f'{fname}.p', 'rb'))
        else:
        #n_lines = int(subprocess.check_output(f"wc -l {fname}", shell=True).decode().split(' ')[0])
            n_lines = 8841823
            with open(fname, 'r', encoding='utf-8') as f:
                for line in tqdm(f, f'Loading file: {fname}', total=n_lines):
                    js = json.loads(line)
                    id_ = js['doc_id']
                    vector = js[self.field]
                    if len(vector) ==  0:
                        count_empty_docs += 1
                        continue 
                    ids, scores = [], []
                    for term, score in vector:
                        if self.stopwords_to_zero:
                            if term in self.stop_words:
                                score = 0
                        ids.append(term)
                        scores.append(score)
                    self.file[id_] = (ids, scores)
                print('empty docs' , count_empty_docs)
                pickle.dump(self.file, open(f'{fname}.p', 'wb'))
                
    def __getitem__(self, id_):
        if id_ not in self.file:    
            print(f'"{id_}" not found!')
            return None
        return self.file[id_]

    def __len__(self):
        return len(self.file)





class File():
    def __init__(self, fname, encoded=False):
        self.file = {}
        count_empty_docs = 0
        #n_lines = int(subprocess.check_output(f"wc -l {fname}", shell=True).decode().split(' ')[0])
        n_lines = 8841823 
        with open(fname, 'r', encoding='utf-8') as f:
            for line in tqdm(f, f'Loading file: {fname}', total=n_lines):
                delim_pos = line.find('\t')
                id_ = line[:delim_pos]
                id_ = id_.strip()
                if encoded:
                    # in case the encoded of the line is empy, and the line only contains the id, then we return None
                    # example of line with empy text: line = "1567 \n" -> len(line[delim_pos+1:]) == 1
                    if len(line[delim_pos+1:]) < 2:
                        self.file[id_] = None
                        count_empty_docs += 1
                    else:
                        # extracting the token_ids and creating a numpy array
                        self.file[id_] = np.fromstring(line[delim_pos+1:], dtype=int, sep=' ')
                else:
                    self.file[id_] = line[delim_pos+1:].strip()
    def __getitem__(self, id_):
        if id_ not in self.file:    
            print(f'"{id_}" not found!')
            return ' '
        return self.file[id_]

    def __len__(self):
        return len(self.file)
