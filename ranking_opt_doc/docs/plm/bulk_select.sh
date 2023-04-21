

for i in 32 64 128 256 ; do
	echo $i
	echo msmarco-doctest2020-top100_judged_top_10.bert_selected.${i}.trec 
	#awk '{print $1"\t"$2"\t"$3}' msmarco-doctest2020-top100_judged_top_10.bert_selected.${i}.trec | xargs -I{} grep -w {} ranking.plm_full.${i}.trec > ranking.plm_full.${i}.trec.219 
	awk '{print $1"\t"$2"\t"$3}' msmarco-doctest2020-top100_judged_top_10.bert_selected.${i}.trec | xargs -I{} grep -w {} ranking.firstp.${i}.trec > ranking.firstp.${i}.trec.219 


done


