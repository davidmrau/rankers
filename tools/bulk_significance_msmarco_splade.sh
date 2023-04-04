run=$1
metric=$2



echo "BM25"
REF=data/msmarco_docs/2020_top_100.scores_all.trec
python3 siginificance_own.py $REF $1 $2



echo "Splade MaxP"
REF=/scratch-shared/draugpu/eval/2020_docs_splademaxp_512_ensemle_distill_eval/model_eval_ranking.trec.scores_all_topics.trec
python3 siginificance_own.py $REF $1 $2

