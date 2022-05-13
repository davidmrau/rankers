

QREL=data/msmarco_docs/2020qrels-docs.txt

MAP=`./trec_eval  $QREL $1 -m all_trec  | grep -w -E 'map'| awk '{print $3}'`
RECIP=`./trec_eval $QREL $1 -m all_trec | grep -w -E 'recip_rank' | awk '{print $3}'` 
RECALL=`./trec_eval $QREL $1 -m all_trec | grep -w -E 'recall_100' | awk '{print $3}'`
BPREF=`./trec_eval $QREL $1 -m all_trec | grep -w -E 'bpref' | awk '{print $3}'`
NDCG=`./trec_eval $QREL $1 -m all_trec | grep -w -E 'ndcg_cut_10' | awk '{print $3}'`


 
deci(){
	printf "%.2f\n" $(echo "scale=0; ${1}*100" | bc)
}

NDCG=`deci $NDCG`
MAP=`deci $MAP`
RECIP=`deci $RECIP`
RECALL=`deci $RECALL`
BPREF=`deci $BPREF`
echo NDCG " & " MAP " & " RECIP " & " RECALL " & " BPREF \\\\  
echo $NDCG " & " $MAP " & " $RECIP " & " $RECALL " & " $BPREF \\\\  





#./trec_eval data/msmarco/2020qrels-pass.txt $1 -m all_trec -q >> ${1}.all_scores.trec
