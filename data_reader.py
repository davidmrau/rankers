import torch
import numpy as np
import transformers
import nltk
import random
from fairseq.data.data_utils import collate_tokens
transformers.logging.set_verbosity_error()



class DataReader(torch.utils.data.IterableDataset):
                

                
    def __init__(self, tokenizer, data_file, num_docs, multi_pass, id2q, id2d, MB_SIZE, qrel_columns={'doc': 0, 'query': 2, 'score': 4}, continue_line=None, encoding='cross', prepend_type=False, keep_q=False, drop_q=False, shuffle=False, has_label_scores=False, max_q_len=None, max_inp_len=512):
            print(data_file)
            self.num_docs = num_docs
            self.doc_col = 2 if self.num_docs <= 1 else 1
            self.MB_SIZE = MB_SIZE
            self.multi_pass = multi_pass
            self.id2d = id2d
            self.id2q = id2q
            self.encoding = encoding
            self.tokenizer = tokenizer
            self.prepend_type = prepend_type
            self.reader = open(data_file, mode='r', encoding="utf-8")
            self.keep_q = keep_q
            self.drop_q = drop_q
            self.shuffle = shuffle
            self.has_label_scores = has_label_scores

            self.max_q_len = max_q_len
            self.max_inp_len = max_inp_len

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
                    if not self.has_label_scores:
                        features['labels'] = torch.ones(self.MB_SIZE, dtype=torch.float)
                    else:
                        features['labels_1'] = list()
                        features['labels_2'] = list()
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
                            q = ' [Q] ' + q
                            ds = [' [D] ' + d_ for d_ in ds]
                        if self.shuffle:
                            q = self.shuffle_fn(q)
                            ds = [self.shuffle_fn(d_) for d_ in ds]
                        if self.drop_q:
                            ds = [self.drop_query_terms(q, d_) for d_ in ds]
                        if self.keep_q:
                            ds = [self.keep_query_terms(q, d_) for d_ in ds]

                        batch_queries.append(q)
                        batch_docs.append(ds)

                        if self.num_docs == 1:
                            features['meta'].append([cols[self.qrel_columns['doc']], cols[self.qrel_columns['query']]])

                        if self.has_label_scores:
                            features['labels_1'].append(float(cols[3]))
                            features['labels_2'].append(float(cols[4]))

                    if self.encoding == 'bi':
                        batch_queries = self.tokenizer(batch_queries, padding=True, return_tensors="pt", truncation=True)
                        batch_docs = [self.tokenizer([bd[i] for bd in batch_docs], padding=True, return_tensors="pt", truncation=True) for i in range(self.num_docs)]
                        features['encoded_queries'] = batch_queries 
                        features['encoded_docs'] = batch_docs
                    elif self.encoding == 'cross':
                        if self.max_q_len != None:
                            batch_queries = self.truncate_queries(batch_queries, self.max_q_len)
                        features['encoded_input'] = [self.tokenizer(batch_queries, [bd[i] for bd in batch_docs], padding=True, truncation='only_second', return_tensors='pt', return_token_type_ids=True, max_length=self.max_inp_len)  for i in range(self.num_docs)]
                        #features['encoded_input'] = [self.tokenizer(batch_queries, [bd[i] for bd in batch_docs], padding=True, truncation='only_second', return_tensors='pt')  for i in range(self.num_docs)]
                    #elif self.encoding == 'cross_fairseq':
                     #   batch = [collate_tokens([self.tokenizer(q_, d_) for q_, d_ in zip(batch_queries, [bd[i] for bd in batch_docs])], pad_idx=1) for i in range(self.num_docs) ]
                      #  features['encoded_input'] = batch

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


    def shuffle_fn(self, sent):
        sent_tokenized = nltk.word_tokenize(sent)
        random.shuffle(sent_tokenized)
        sent_shuffled = ' '.join(sent_tokenized)
        return sent_shuffled



    def truncate_queries(self, batch, max_len):
        q_batch_trunc = list()

        for q in batch:
            q_token = self.tokenizer.tokenize(q)

            if len(q_token) <= max_len:
                q_batch_trunc.append(q)
            else:
                q_token_trunc = q_token[:max_len]

                q_trunc = ''
                for t in q_token_trunc:
                    if t.startswith('##'):
                        q_trunc += q_trunc.rstrip(' ')
                        q_trunc += t.lstrip('##')
                    else:
                        q_trunc += f'{t} '
                q_batch_trunc.append(q_trunc.rstrip(' '))

        return q_batch_trunc	       
