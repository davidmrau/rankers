import torch
import numpy as np
import transformers
import nltk

transformers.logging.set_verbosity_error()



class DataReader(torch.utils.data.IterableDataset):
                

                
    def __init__(self, tokenizer, data_file, num_docs, multi_pass, id2q, id2d, MB_SIZE, qrel_columns={'doc': 0, 'query': 2, 'score': 4}, continue_line=None, bi_encoding=False, prepend_type=False, keep_q=False, drop_q=False):
            print(data_file)
            self.num_docs = num_docs
            self.doc_col = 2 if self.num_docs <= 1 else 1
            self.MB_SIZE = MB_SIZE
            self.multi_pass = multi_pass
            self.id2d = id2d
            self.id2q = id2q
            self.bi_encoding = bi_encoding
            self.tokenizer = tokenizer
            self.prepend_type = prepend_type
            self.reader = open(data_file, mode='r', encoding="utf-8")
            self.keep_q = keep_q
            self.drop_q = drop_q
            if continue_line != None:
                print(f'continuing line {continue_line}')
            if continue_line:
                    for i, line in enumerate(self.reader):
                            if i == continue_line-1:
                                    break
            self.reader.seek(0)
            self.qrel_columns = qrel_columns
    

    def __iter__(self):
            self.ignored_docs = 0
            self.done = False
            while True:
                    features = {}
                    features['labels'] = torch.ones(self.MB_SIZE, dtype=torch.float)
                    features['meta'] = []
                    features['encoded_input'] = list()
                    batch_queries, batch_docs, batch_q_lengths, batch_d_lengths = list(), list(), list(), list()
                    while len(batch_queries) < self.MB_SIZE:
                            
                        row = self.reader.readline()
                        if row == '':
                                if self.multi_pass:
                                        self.reader.seek(0)
                                        row = self.reader.readline()
                                else:
                                        # file end while testing: stop iter by returning None and set seek to file start
                                        print('Training file exhausted, read again...')
                                        self.reader.seek(0)
                                        print('Ignored Docs:', self.ignored_docs)
                                        self.done = True
                                        break
                        cols = row.split()

                        q = self.id2q[cols[0]]
                        # get doc_ids       
                        ds_ids = [  cols[self.doc_col + i].strip() for i in range(self.num_docs)]   
                        # get doc content
                        ds = [self.id2d[id_] for id_ in ds_ids]

                        # if any of the docs is None skip triplet   
                        if any(x is None for x in ds) or q is None:
                                self.ignored_docs += 1
                                continue

                        if self.prepend_type:
                            q = '[Q] ' + q
                            ds = ['[D] ' + d_ for d_ in ds]

                        if self.drop_q:
                            ds = [self.drop_query_terms(q, d_) for d_ in ds]
                        if self.keep_q:
                            ds = [self.keep_query_terms(q, d_) for d_ in ds]
                        batch_queries.append(q)
                        batch_docs.append(ds)

                        if self.num_docs == 1:
                            features['meta'].append([cols[self.qrel_columns['doc']], cols[self.qrel_columns['query']]])

                    if self.bi_encoding == False:
                        features['encoded_input'] = [self.tokenizer(batch_queries, [bd[i] for bd in batch_docs], padding=True, truncation='only_second', return_tensors='pt')  for i in range(self.num_docs) ]
                             
                    else:
                        batch_queries = self.tokenizer(batch_queries, padding=True, return_tensors="pt", truncation=True)
                        batch_docs = [self.tokenizer([bd[i] for bd in batch_docs], padding=True, return_tensors="pt", truncation=True) for i in range(self.num_docs)]
                        features['encoded_queries'] = batch_queries 
                        features['encoded_docs'] = batch_docs


                    yield features

                    if self.done:
                        self.done = False
                        return
                    
                    
    def collate_fn(self, batch):
            return batch

    def drop_query_terms(self, q, d):
        q_set_tokenized = set(nltk.word_tokenize(q))
        d_tokenized = nltk.word_tokenize(d)
        d_drop = []
        for d_term in d_tokenized:
            if d_term not in q_set_tokenized:
                d_drop.append(d_term)
            else:
                d_drop.append(self.tokenizer.mask_token)
        return ' '.join(d_drop)
    
    def keep_query_terms(self, q, d):
        q_set_tokenized = set(nltk.word_tokenize(q))
        d_tokenized = nltk.word_tokenize(d)
        d_keep = []
        for d_term in d_tokenized:
            if d_term in q_set_tokenized:
                d_keep.append(d_term)
            else:
                d_keep.append(self.tokenizer.mask_token)
        return ' '.join(d_keep) 
