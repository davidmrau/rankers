import numpy as np
from tqdm import tqdm
import subprocess
import json 
class File():
    def __init__(self, fname, encoded=True):
        self.file = {}
        count_empty_docs = 0
        n_lines = int(subprocess.check_output(f"wc -l {fname}", shell=True).decode().split(' ')[0])
        with open(fname, 'r', encoding='utf-8') as f:
            for line in tqdm(f, f'Loading file: {fname}', total=n_lines):
                js = json.loads(line)
                id_ = js['id']
                vector = js['vector']
                ids, scores = [], []
                for k in vector:
                    ids.append(k)
                    scores.append(vector[k])
                self.file[id_] = (ids, scores)
                    

                
    def __getitem__(self, id_):
        if id_ not in self.file:    
            print(f'"{id_}" not found!')
            return ' '
        return self.file[id_]

    def __len__(self):
        return len(self.file)
