from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import pickle
import numpy as np
import re


input_file = sys.argv[1]

def tokenize(s):
	return s.split(' ')
if True:

	corpus = list()
	for i, l in enumerate(open(input_file, encoding='utf-8')):
		if i % 100000 == 0:
			print(i)
		_, text = l.split('\t')
		corpus.append(text.strip())
		
	vectorizer = TfidfVectorizer(tokenizer=tokenize)
	X = vectorizer.fit_transform(corpus)

	pickle.dump(vectorizer, open(f'{input_file}_tfidf_vectorizer', 'wb'))
	pickle.dump(X, open(f'{input_file}_tfidf_sparse', 'wb'))

else:
	#print('!!! change this back!!!')
	vec_file = 'data/msmarco/collection.tsv_tfidf_vectorizer' #f'{input_file}_tfidf_vectorizer'
	x_file = 'data/msmarco/collection.tsv_tfidf_sparse'   # f'{input_file}_tfidf_sparse'
	#vec_file = f'{input_file}_tfidf_vectorizer'
	#x_file = f'{input_file}_tfidf_sparse'
	vectorizer = pickle.load(open(vec_file , 'rb'))
	X = pickle.load(open(x_file,  'rb'))
	#ids = range(1, X.get_shape()[0]+1)


terms = np.array(vectorizer.get_feature_names())
term2id = {t: i for i,t in enumerate(terms)}
tokenizer =  vectorizer.build_tokenizer()
out_file_uniq = open(f'{input_file}_tfidf_sorted_decreasing_uniq', 'w', encoding='utf-8')
out_file = open(f'{input_file}_tfidf_sorted_decreasing', 'w', encoding='utf-8')
for i, (doc, line) in enumerate(zip(X, open(input_file, encoding='utf-8'))):
	if i % 100000 == 0:
		print(i)
	id_, text = line.split('\t') 
	try:
		text = text.lower().strip()
		text = re.sub('\s+', ' ', text)
		terms_doc = tokenizer(text)
		terms_doc_set = list(set(terms_doc))
		terms_doc_set = [t for t in terms_doc_set if t in term2id]
		idx = [ term2id[t] for t in  terms_doc_set ]
		values_doc = X[i][:, idx].toarray()[-1]
		assert len(values_doc) == len(terms_doc_set)

		sorted_after_values = [ ' '.join([x] * terms_doc.count(x)) for _, x in sorted(zip(values_doc, terms_doc_set), key=lambda x: x[0], reverse=True)]
		sorted_after_values_str = ' '.join(sorted_after_values)
		sorted_after_values_uniq = [ x   for v, x in sorted(zip(values_doc, terms_doc_set), key=lambda x: x[0], reverse=True)]
		sorted_after_values_str_uniq = ' '.join(sorted_after_values_uniq)



	except Exception as e:
		print(e)
		print('error', i)
		sorted_after_values_str = ' '
		sorted_after_values_str_uniq = ' '
	out_file_uniq.write(f'{id_}\t{sorted_after_values_str_uniq}\n')
	out_file.write(f'{id_}\t{sorted_after_values_str}\n')
		
