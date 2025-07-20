#!/bin/bash

# Evaluation with RIBES scores
# How to use: ./eval.sh firstmodel-hypothesis-filename second-model-hypothesis-filename reference-filename
# e.g. ./eval.sh ./fm_hyp.txt ./sm_hyp.txt ./ref.txt

fm=$1; sm=$2; ref=$3;
i=0;

while read line 
do
        fm_arr[$i]="$line";
        i=$((i+1));
done < "$fm"

j=0;
while read line 
do
        sm_arr[$j]="$line";
        j=$((j+1));
done < "$sm"

k=0;
while read line 
do
        ref_arr[$k]="$line";
        k=$((k+1));
done < "$ref"

len=${#fm_arr[@]};

for (( i=0; i<$len; i++ )); 
do 
 echo "" > fm_hyp.txt;
 echo "" > sm_hyp.txt;
 echo "" > ref.txt;

 echo "${fm_arr[$i]}" > fm_hyp.txt ; 
 echo "${sm_arr[$i]}" > sm_hyp.txt ; 
 echo "${ref_arr[$i]}" > ref.txt ; 

#echo "Evaluation with ribes score:";
fm_rs=`python ./RIBES-1.03.1/RIBES.py -r ref.txt fm_hyp.txt`
sm_rs=`python ./RIBES-1.03.1/RIBES.py -r ref.txt sm_hyp.txt`

if [[ "$fm_rs" > "$sm_rs" ]]; then
   echo "${fm_arr[$i]}" >> rs.txt
else
  echo "${sm_arr[$i]}" >> rs.txt	
fi 

done
