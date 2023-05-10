import torch
import numpy as np
import transformers
import random
from collections import defaultdict
transformers.logging.set_verbosity_error()
import gzip
from transformers import BasicTokenizer

class DataReader(torch.utils.data.IterableDataset):
                

                
    def __init__(self, tokenizer, model_type, data_file, num_docs, multi_pass, id2q, id2d,  MB_SIZE, qrel_columns={'doc': 0, 'query': 2, 'score': 4},continue_line=None, prepend_type=False, keep_q=False, drop_q=False, preserve_q=False, shuffle=False, sort=False, has_label_scores=False, max_q_len=None, max_inp_len=512, tf_embeds=False, sliding_window=False, rand_length=False, rand_passage=False):
            print(data_file)
            self.num_docs = num_docs
            self.doc_col = 2 if self.num_docs <= 1 else 1

            self.rs = defaultdict(list)

            #for l in open('/home/draugpu/distributed_ir/ORACLE_tec4_kmeans'):
            #    spl = l.rstrip().split()
            #    query, index = spl[0], spl[2]
            #    if len(self.rs[query]) < 10:
            #        self.rs[query].append(index)

            self.MB_SIZE = MB_SIZE
            self.multi_pass = multi_pass
            self.id2d = id2d
            self.id2q = id2q
            self.model_type = model_type
            self.tokenizer = tokenizer
            self.prepend_type = prepend_type
            self.reader = open(data_file, mode='r', encoding="utf-8")
            self.data_file = data_file
            self.keep_q = keep_q
            self.drop_q = drop_q
            self.preserve_q = preserve_q
            self.shuffle = shuffle
            self.sort = sort
            self.tf_embeds = tf_embeds
            self.has_label_scores = has_label_scores
            self.sliding_window = sliding_window
            self.random_passage = rand_passage
            self.max_q_len = max_q_len
            self.max_inp_len = max_inp_len
            self.basic_tokenizer = BasicTokenizer()
            self.rand_length = rand_length
            if continue_line != None:
                print(f'continuing line {continue_line}')
            if continue_line:
                    for i, line in enumerate(self.reader):
                            if i == continue_line-1:
                                    break
            self.reader.seek(0)
            self.qrel_columns = qrel_columns

    def norm(self, scores):
        scores = np.array(scores)
        return 10 * np.round(scores / max(scores), 1)
        #return np.rint( 100 * (scores / max(scores)))

    def new_batch(self):
        features = {}
        if not self.has_label_scores:
            features['labels'] = torch.ones(self.MB_SIZE, dtype=torch.float)
        else:
            features['labels_1'] = list()
            features['labels_2'] = list()
        features['meta'] = list()
        features['encoded_input'] = list()
        features['tf_embeds'] = list()
        features['scores'] = list()
        batch_queries, batch_docs = list(), list()
        return features, batch_queries, batch_docs
    
    def get_and_add_scores(self, docs):
        score_docs = list()
        for doc in docs:
            d, importance_score = doc
            importance_score = self.norm(importance_score)
            new_doc = list()
            for term, score in zip(d, importance_score):
                #new_doc.append(f'[unused{int(score)}]')
                new_doc.append(f'{int(score)}')
                new_doc.append(str(term))
            #score_docs.append(' '.join(new_doc))
            score_docs.append(new_doc)
        return score_docs

    def sort_after_scores(self, tokens, scores):
        
        sorted_idxs = np.argsort(scores)[::-1].astype(int)
        return np.array(tokens)[sorted_idxs].tolist(), np.array(scores)[sorted_idxs].tolist()

    def __iter__(self):
            self.ignored_docs = 0
            self.done = False
            self.first_batch = False
            while True:
                    features, batch_queries, batch_docs = self.new_batch()
                    if self.rand_length:
                        self.max_inp_len = random.choice([32, 64, 128, 256, 512]) 
                        if self.max_inp_len == 32 or self.max_inp_len == 64:
                            self.max_q_len = 16
                        else:
                            self.max_q_len = None
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
                                        yield self.prepare_input(features, batch_queries, batch_docs, scores)
                                        return
                        cols = row.split()
                        q_id = cols[0]
                        #index = self.data_file.split('/')[-1].split('.')[1]
                        #if len(self.rs[q_id]) == 0:
                        #    return
                        #if index not in self.rs[q_id]:
                        #    continue
                        q = self.id2q[cols[0]]
                        # get doc_ids       
                        ds_ids = [  cols[self.doc_col + i].strip() for i in range(self.num_docs)]
                        ds, scores = list(), list()
                        for id_ in ds_ids:
                            ids, score = self.id2d[id_]
                            #ids, score = self.sort_after_scores(ids, score)
                            ds.append(ids)
                            scores.append(score)

                        # if any of the docs is None skip triplet   
                        if any(x is None for x in ds) or q is None:
                                self.ignored_docs += 1
                                continue

                        #ds = self.get_and_add_scores(ds)
                        

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

                        if self.preserve_q:
                            ds = [self.preserve_query_terms(q, d_) for d_ in ds]

                        if self.has_label_scores:
                            features['labels_1'].append(float(cols[3]))
                            features['labels_2'].append(float(cols[4]))
                        
                        if self.sliding_window or self.random_passage:  
                            if self.num_docs != 1:
                                raise NotImplementedError()

                            d_tokenized = self.basic_tokenizer.tokenize(ds[-1])
                            d_tokenized = d_tokenized
                            len_chunk = 512
                            stride = 256
                            chunks = [d_tokenized[i:i+len_chunk] for i in range(0, len(d_tokenized), len_chunk-stride)]
                    
                            if self.sliding_window:
                                if len(chunks) > 30:
                                    chunks = [chunks[0]] + [chunks[i] for i in random.sample(range(1, len(chunks)-1), 28)] + [chunks[-1]]
                                for chunk in chunks:
                                    if len(batch_docs) >= self.MB_SIZE:
                                        yield self.prepare_input(features, batch_queries, batch_docs)
                                        features, batch_queries, batch_docs = self.new_batch() 
                                    batch_docs.append([' '.join(chunk)])
                                    batch_queries.append(q)
                                    features['meta'].append([cols[self.qrel_columns['doc']], cols[self.qrel_columns['query']]])

                            elif self.random_passage:
                                features['meta'].append([cols[self.qrel_columns['doc']], cols[self.qrel_columns['query']]])
                                batch_queries.append(q)

                                if len(chunks) == 1:
                                    rand_passage = chunks[0]
                                else:
                                    rand_passage = chunks[random.randint(0, len(chunks)-1)]

                                batch_docs.append([' '.join(rand_passage)])

                        else:
                            #if self.num_docs == 1:
                            features['meta'].append([cols[self.qrel_columns['doc']], cols[self.qrel_columns['query']]])
                            batch_queries.append(q)
                            batch_docs.append(ds)
                    yield self.prepare_input(features, batch_queries, batch_docs, scores)
    def prepare_input(self, features, batch_queries, batch_docs, scores): 
        doc_ids = [el[0] for el in features['meta']]
        if self.model_type == 'bi':
            batch_queries = self.tokenizer(batch_queries, padding=True, return_tensors="pt", truncation=True)
            batch_docs = [self.tokenizer([bd[i] for bd in batch_docs], padding=True, return_tensors="pt", truncation=True, max_length=self.max_inp_len) for i in range(self.num_docs)]
            
            for batch_doc in batch_docs:
                position_ids = torch.zeros_like(batch_doc['input_ids'])
                for i, d_scores in enumerate(importance_scores):
                    for j, scores in enumerate(d_scores):
                        position_ids[i][j+1] = scores
                batch_doc['poisition_ids'] = position_ids
                    
            features['encoded_queries'] = batch_queries 
            features['encoded_docs'] = batch_docs
            if not self.first_batch:
                idxs = random.sample(range(len(doc_ids)),min(3, len(doc_ids)))
                for idx in idxs:
                    print(doc_ids[idx], self.tokenizer.decode(features['encoded_docs'][0]['input_ids'][idx]))
                self.first_batch=True
            if self.sort:
                raise NotImplementedError()

        elif self.model_type == 'cross':
            if self.max_q_len != None:
                batch_queries = self.truncate_queries(batch_queries, self.max_q_len)
            features['encoded_input'] = [self.tokenizer(batch_queries, [bd[i] for bd in batch_docs], padding=True, truncation='only_second', return_tensors='pt', return_token_type_ids=True, max_length=self.max_inp_len)  for i in range(self.num_docs)]
            if self.sort:
                features['encoded_input'] = [ self.sort_fn(el) for  el in  features['encoded_input']]


            for i in range(self.num_docs):
                features['scores'].append(torch.zeros_like(features['encoded_input'][i]['token_type_ids']).float())
                decreasing = torch.arange(features['encoded_input'][i]['token_type_ids'].shape[1], 0, -1)
                decreasing_doc = features['encoded_input'][i]['token_type_ids'] * decreasing
                starts_docs = torch.argmax(decreasing_doc, 1)
                shape = features['scores'][i].shape
                for j in range(shape[0]):
                    end = min(len(scores[i]), (shape[1] - starts_docs[j]))
                    features['scores'][i][j, starts_docs[j]: starts_docs[j]+end] = torch.FloatTensor([scores[i][:end]])

            if not self.first_batch:
                #dids = [ el[1] for el in features['meta'3]
                idxs = random.sample(range(len(doc_ids)), min(len(doc_ids), 3))
                for idx in idxs:
                    print(doc_ids[idx], self.tokenizer.decode(features['encoded_input'][-1]['input_ids'][idx]))
                self.first_batch=True
            
            
        if self.tf_embeds:
            features['tf_embeds'] = [self.get_tf_embeds(inp['input_ids']) for inp in features['encoded_input']]
        return features 
    def collate_fn(self, batch):
        return batch

    def drop_query_terms(self, q, d):
        q_set_tokenized = set(self.basic_tokenizer.tokenize(q))
        d_tokenized = self.basic_tokenizer.tokenize(d)
        d_drop = []
        for d_term in d_tokenized:
            if d_term not in q_set_tokenized:
                d_drop.append(d_term)
            else:
                d_drop.append(self.tokenizer.mask_token)
        return ' '.join(d_drop)
    
    def keep_query_terms(self, q, d):
        q_set_tokenized = set(self.basic_tokenizer.tokenize(q))
        d_tokenized = self.basic_tokenizer.tokenize(d)
        d_keep = []
        for d_term in d_tokenized:
            if d_term in q_set_tokenized:
                d_keep.append(d_term)
            else:
                d_keep.append(self.tokenizer.mask_token)
        return ' '.join(d_keep)

        
    def preserve_query_terms(self, q, d):
        q_set_tokenized = set(self.basic_tokenizer.tokenize(q))
        d_tokenized = self.basic_tokenizer.tokenize(d)
        d_keep = []
        for d_term in d_tokenized:
            if d_term in q_set_tokenized:
                d_keep.append(d_term)
                d_keep.append(d_term)
            else:
                d_keep.append(d_term)
        return ' '.join(d_keep)

    def shuffle_fn(self, sent):
        sent_tokenized = self.basic_tokenizer.tokenize(sent)
        random.shuffle(sent_tokenized)
        sent_shuffled = ' '.join(sent_tokenized)
        return sent_shuffled



    def sort_fn(self, batch_input):
        for i, inp in enumerate(batch_input['input_ids']):
            query_ended = False
            doc_end = len(inp)
            for j, t in enumerate(inp):
                t = t.tolist()
                if t == self.tokenizer.sep_token_id and not query_ended:
                        query_end = j
                        query_ended = True
                if t == 0:
                        doc_end = j
            q = inp[1:query_end]
            batch_input['input_ids'][i][1:query_end] =  q.sort(descending=True)[0]
            
            d = inp[query_end+1:doc_end]
            batch_input['input_ids'][i][query_end+1:doc_end] = d.sort(descending=True)[0]
        return batch_input

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
                        q_trunc = q_trunc.rstrip(' ')
                        q_trunc += t.lstrip('##')
                    else:
                        q_trunc += f'{t} '
                q_batch_trunc.append(q_trunc.rstrip(' '))

        return q_batch_trunc	      

    def get_tf_embeds(self, docs):
        tf_embed = torch.zeros(docs.shape[0], docs.shape[1], 768)
        for i, d in enumerate(docs):
            tf_doc = defaultdict(int)
            doc_start = None
            for k, t in enumerate(d.tolist()):
                if t == self.tokenizer.sep_token_id and doc_start is None:
                    doc_start = k
                if doc_start and t != self.tokenizer.sep_token_id and t != self.tokenizer.pad_token_id: 
                    tf_doc[t] += 1
            tfs = list(tf_doc.values())
            min_ = np.min(tfs)
            max_ = np.max(tfs)
            if min_ == max_:
                continue
            for j, t in enumerate(d.tolist()):
                if j > doc_start and t != 0:
                    #tf_norm = (2 * ((tf_doc[t]-min_) / (max_ - min_)) - 1) * .3
                    tf_norm = ((tf_doc[t]-min_) / (max_ - min_))*.5
                    tf_embed[i, j] = tf_norm
        return tf_embed

    def get_inputs(self, tokenizer, query_text, sents, masked_lm=False, max_seq_len=2048, max_sent_num=256, window_size=128):

        # Query tokens.
        # Query tokens.
        input_ids = [0] # Global attention token.
        input_ids.extend(tokenizer.encode(query_text))
        QL = len(input_ids)

        # Sentence tokens.
        # input_ids.append(0) # Global attention token.
        sid = 0
        sent_locs = []
        sent_mask = []
        while sid < max_sent_num and len(input_ids) < max_seq_len:
            sent_locs.append(len(input_ids))
            sent_mask.append([1.0])
            # Add the global attention token and sent tokens.
            input_ids.append(0)
            input_ids.extend(tokenizer.encode(sents[sid]))
            sid += 1
            if sid == len(sents): break
        last_pos = len(input_ids)
        input_ids.append(2)
        
        # Padding handling.
        L = len(input_ids)
        num_sent = len(sent_locs)
        if self.batch_size > 1:
            if num_sent < max_sent_num:
                sent_mask.extend(
                        [[0.] for i in range(max_sent_num - num_sent)])
                sent_locs.extend([0] * (max_sent_num - num_sent))
            if L < self.max_seq_len:
                input_ids.extend([1] * (max_seq_len - 1 - L))
                input_ids.append(2)

        if L > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            L = max_seq_len

        if len(input_ids) < 16:
            input_ids.extend([1] * (16 - L))


        ## Fill the attention masks.
        # Anchor BOS indices.
        bos_idx = [i for i, x in enumerate(input_ids) if x == 0]
        # Fill in local attention first.
        tok_mask = np.zeros([len(input_ids)], dtype=np.long)
        tok_mask[range(L)] = 1

        # Adjust global attention by the pattern.
        # Longformer-QA
        if self.attn_pattern == 1:
            tok_mask[0] = 2
            tok_mask[sent_locs[0]] = 2
            self.attn_cnt += 2


        # QDS-Transformer
        if self.attn_pattern == 2:
            tok_mask[range(QL)] = 2
            tok_mask[bos_idx] = 2
            self.attn_cnt += QL + len(bos_idx)
            

        # QDS-Transformer (Q)
        if self.attn_pattern == 3:
            tok_mask[range(QL)] = 2
            tok_mask[sent_locs[0]] = 2
            self.attn_cnt += QL + 1

        # QDS-Transformer (S)
        if self.attn_pattern == 4:
            tok_mask[0] = 2
            tok_mask[bos_idx] = 2
            self.attn_cnt += 1 + len(bos_idx)
        
        self.attn_cnt += window_size * 2
        self.attn_total += L

        return input_ids, tok_mask, sent_locs, sent_mask


    def to_tensor(self, batch):
        return {
                'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
                'tok_mask': torch.tensor(batch['tok_mask'], dtype=torch.float),
                'sent_locs': torch.tensor(
                    batch['sent_locs'], dtype=torch.long),
                'sent_mask': torch.tensor(
                    batch['sent_mask'], dtype=torch.float), 'scores': torch.tensor(batch['scores'], dtype=torch.floar)}


            
class MSMARCO(torch.utils.data.IterableDataset):
    def __init__(self, file_, tokenizer, max_len):
        self.file_ = file_
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __iter__(self):
        if self.file_.endswith('.gz'):
            open_fn = gzip.open
        else:
            open_fn = open
        with open_fn(self.file_, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                pid, passage = line.split('\t')
                yield (pid, passage)
    def collate_fn(self, inp):
        pid, data = [x[0] for x in inp], [x[1] for x in inp]
        tokenized = self.tokenizer(data, add_special_tokens=True, padding=True,truncation=True, return_tensors='pt', max_length=self.max_len)
        return (pid, tokenized)

