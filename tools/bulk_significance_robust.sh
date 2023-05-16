run=$1
metric=$2

echo "BM25"
REF=../runs/robust_top_100_bm25.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2

echo "big bird"
REF=../runs/robust_bigbert_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2


echo "longformer"
REF=../runs/robust_longformer_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2


echo "longformer-qa"
REF=../runs/robust_longformer-qa_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2



echo "IDCM"
REF=../runs/robust_idcm_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2

echo "MiniLM 12 MaxP"
REF=../runs/robust_minilm12_max_p_512_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2

echo "MiniLM 12 FirstP 64"
REF=../runs/robust_minilm12first_p_64_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2


echo "MiniLM 12 RandP 64"
REF=../runs/robust_minilm12_rand_passage_64_1_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2


echo "PLM 64"
REF=../runs/robust_minilm12plm_64_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 significance_ranks.py $REF $1 $2
