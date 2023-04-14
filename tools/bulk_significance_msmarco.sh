run=$1
metric=$2

echo "BM25"
REF=data/msmarco_docs/2020_top_100.scores_all.trec
python3 significance_ranks.py $REF $1 $2

echo "big bird"
REF=models/2019_docs_bigbertbz_16_lr_3e-06_docs_train_lr_3e-6/model_eval_ranking9.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2


echo "longformer"
REF=models/2019_docs_longformerbz_16_lr_3e-06_docs_train_lr_3e-6/model_eval_ranking5.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2


echo "longformer-qa"
REF=models/2019_docs_longformer-qabz_16_lr_3e-06_docs_train_lr_3e-6_title+body/model_eval_ranking5.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2


echo "MiniLM 12 MaxP"
REF=runs/2020_docs_minilm12_max_p_512_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2


echo "IDCM"
REF=runs/2020_docs_idcm_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2

echo "MiniLM 12 FirstP 64"
REF=runs/2020_docs_minilm12first_p_64_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2


echo "MiniLM 12 RandP 64"
REF=runs/2020_docs_minilm12_rand_passage_64_1_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2

echo "CT zero-shot"
REF=runs/2019_docs_tfidf_minilm12first_p_64_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2
