#FILES=`ls -d /project/draugpu/cikm_eval/robust/crossencoder/*wiki*`
FILES=`ls -d /project/draugpu/cikm_eval/msmarco/crossencoder/*`

for FILE in $FILES;do
	echo $FILE
	#bash eval_robust.sh $FILE/model_eval_ranking.trec
	bash eval_docs.sh $FILE/model_eval_ranking.trec
done
