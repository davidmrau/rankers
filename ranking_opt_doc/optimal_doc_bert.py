from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
import torch
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
import json
from collections import defaultdict
from file_interface import File
import pickle 
from tqdm import tqdm 

def get_scores(model, tokenizer, queries, docs):
    with torch.no_grad():
        scores = model(**tokenizer(queries, docs, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda'))
    #return scores.logits[:, 1]
    return scores.logits.ravel()

def dot_last(model, tokenizer, queries, docs):
    with torch.no_grad():
        emb_queries = model(**tokenizer(queries, padding=True, truncation=True, return_tensors='pt').to('cuda'),output_hidden_states=True  ).hidden_states[-1][:,0,:]
        emb_docs = model(**tokenizer(docs, padding=True, truncation=True, return_tensors='pt').to('cuda'), output_hidden_states=True).hidden_states[-1][:,0,:]
    scores = torch.bmm(emb_queries.unsqueeze(1), emb_docs.unsqueeze(-1)).squeeze()
    return scores.ravel()



def get_score(model, tokenizer, query, doc):
     return model(**tokenizer([query], [doc], padding=True, truncation=True, return_tensors='pt').to('cuda')).logits.ravel().item()

def add_token(doc, doc_tokens):
    doc_inputs = list()
    for token in doc_tokens:
        doc_inputs.append(f"{doc} {token}")
    return doc_inputs

def remove_token(doc_tokens):
    doc_inputs = list()
    for i, token in enumerate(doc_tokens):
        tmp = doc_tokens.copy()
        del tmp[i]
        doc_inputs.append(' '.join(tmp))
    return doc_inputs



#model = AutoModelForSequenceClassification.from_pretrained('/project/draugpu/experiments_ictir/bert/bz_128_lr_3e-06/model_30/')
#model = AutoModelForSequenceClassification.from_pretrained('dmrau/bow-bert')
#model = AutoModelForSequenceClassification.from_pretrained('dmrau/crossencoder-msmarco')
model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
model = model.to('cuda')
model.eval()






optim_doc = ""
if False:
    print('optimal_doc')
    for i in range(len(remaining_tokens)):
        batch_docs = add_token(optim_doc, remaining_tokens)
        scores = get_scores(model, tokenizer, [query] * len(batch_docs) , batch_docs) 
        max_idx = torch.argmax(scores)
        optim_term  = remaining_tokens[max_idx]
        optim_doc += optim_term + ' '
        print(scores[max_idx].item(), optim_doc)
        remaining_tokens.remove(optim_term)

if False:
    remaining_tokens = doc_tokens.copy()
    #remaining_tokens = tokenizer.tokenize(optim_doc)[:64]
    batch_docs = remove_token(remaining_tokens)
    scores = get_scores(model, tokenizer, [query] * len(batch_docs) , batch_docs) 
    _, indices = torch.topk(scores.ravel(), 32)
    new_doc = ''
    for ind in indices:
        new_doc += ' ' + doc_tokens[ind]
    print('doc single:', new_doc)
    score = get_score(model, tokenizer, query, new_doc)
    print('score:', round(score, 2))

#query_file = 'data/msmarco/msmarco-test2020-queries.tsv'
#doc_file = 'data/msmarco/collection.tsv' 
#run_file = 'ranking_opt_doc/run.msmarco-passage.bm25.topics.dl20.judged.txt_top_100.tsv'

query_file = 'data/msmarco_docs/msmarco-test2020-queries.tsv'
doc_file = 'data/msmarco_docs/msmarco-docs.tsv_test_2020.tsv' 
#run_file = 'data/msmarco_docs/msmarco-doctest2020-top100_judged'
run_file = 'data/msmarco_docs/msmarco-doctest2020-top100_judged_top_10'

queries = File(query_file)
docs = File(doc_file)

result = defaultdict(dict)

for l in tqdm(open(run_file), total=4500):
    #q_id, q0, d_id  = l.strip().split()
    spl = l.strip().split()
    q_id, d_id  = spl[0], spl[2] 
    doc = docs[d_id]
    query = queries[q_id]
    doc_tokens = tokenizer.tokenize(doc)[:512]
    remaining_tokens = doc_tokens.copy()
    for i in tqdm(range(len(remaining_tokens))):
        batch_docs = remove_token(remaining_tokens)
        scores = get_scores(model, tokenizer, [query] * len(batch_docs) , batch_docs) 
        #elif len(remaining_tokens) > 480:
        #    _, max_idxs = torch.topk(scores, 224)
        #    for max_idx in max_idxs.cpu().detach().numpy():
        #        del remaining_tokens[max_idx]   
        max_idx = torch.argmax(scores)
        del remaining_tokens[max_idx]


        if len(remaining_tokens) in [16, 32, 64, 128, 256, 320, 384,448, 480, 512 ]:
            result[f'{q_id} {d_id}'][len(remaining_tokens)] = " ".join(remaining_tokens)
            score = scores[max_idx].item()
            open(f'{run_file}.bert_selected.{len(remaining_tokens)}.trec', 'a').write(f'{q_id}\tQ0\t{d_id}\t1\t{score}\teval\n')
            
        pickle.dump(result, open(f'{run_file}.bert_selected.p' ,'wb'))
exit()
    #print('doc iteratively', " ".join(remaining_tokens))
    #score = get_score(model, tokenizer, query, " ".join(remaining_tokens))
    #print('score:', round(score, 2))

if False:
    remaining_tokens = doc_tokens.copy()
    #remaining_tokens = tokenizer.tokenize(optim_doc)[:64]
    for i in range(len(remaining_tokens)):
        batch_docs = remove_token(remaining_tokens)
        scores = dot_last(model, tokenizer, [query] * len(batch_docs) , batch_docs) 
        max_idx = torch.argmax(scores)
        #for token, score in zip(remaining_tokens, scores):
        #    print(token, round(score.item(), 2))
        #print('removing', remaining_tokens[max_idx])
        del remaining_tokens[max_idx]
        #print(scores[max_idx].item(), ' '.join(remaining_tokens))
        if len(remaining_tokens) <= 32:
            break
    print('doc iterativelely dot last', " ".join(remaining_tokens))
    score = get_score(model, tokenizer, query, " ".join(remaining_tokens))
    print('score:', round(score, 2))

exit()
js = json.loads('{"id": "1020327", "contents": "From U.S. Citizenship and Immigration Services (USCIS) Naturalization Guide to Naturalization Child Citizenship Act Naturalization Test. Laws and Regulations Read the Code of Federal Regulation Chapter 8 Section 319.2, Expeditious Naturalization regulation and read the INA section 319(b). Department of State Employees and Spouses Only", "vector": {"from": 16, "u": 32, ".": 51, "s": 10, "citizenship": 132, "and": 16, "immigration": 91, "services": 37, "(": 17, "usc": 107, "##is": 55, ")": 15, "natural": 144, "##ization": 80, "guide": 93, "to": 47, "child": 92, "act": 87, "test": 90, "laws": 62, "regulations": 70, "read": 55, "the": 25, "code": 103, "of": 4, "federal": 80, "regulation": 75, "chapter": 78, "8": 94, "section": 74, "319": 122, "2": 33, ",": 14, "ex": 57, "##ped": 75, "##iti": 64, "##ous": 33, "ina": 109, "b": 36, "department": 87, "state": 64, "employees": 56, "spouse": 86, "##s": 22, "only": 81, "[SEP]": 0}}')

doc_splade = sorted(js['vector'], key=js['vector'].get, reverse=True)
doc_splade = " ".join(doc_splade[:32])
print('doc splade:', doc_splade)
score = get_score(model, tokenizer, query, doc_splade)
print('score:', round(score, 2))


