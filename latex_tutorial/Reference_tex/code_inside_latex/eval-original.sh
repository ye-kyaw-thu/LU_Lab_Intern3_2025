#!/bin/bash

# Evaluation with RIBES scores
# How to use: ./eval.sh firstmodel-hypothesis-filename second-model-hypothesis-filename reference-filename
# e.g. ./eval.sh ./fm_hyp.txt ./sm_hyp.txt ./ref.txt

fm=$1
sm=$2
ref=$3

#echo "မင်း အွဥ်ႏဖွို့ꩻတဝ်း နဲ့ အလင်ꩻ တစ်ခု မဟုတ် ဘူး နော် ဟုတ်သလား ။" > fm_hyp.txt
#echo "မင်း အလင်ꩻ အွဥ်ႏဖွို့ꩻတဝ်း တစ် ခု နဲ့ လဲ ။" > sm_hyp.txt
#echo "မင်း အဲ့ဒါ ကို အခြား တစ်ခုနဲ့ မ ချိတ် ဘူးလား" > ref.txt

i=0
while read line 
do
        fm_arr[$i]="$line"
        i=$((i+1))
done < "$fm"

j=0
while read line 
do
        sm_arr[$j]="$line"
        j=$((j+1))
done < "$sm"

k=0
while read line 
do
        ref_arr[$k]="$line"
        k=$((k+1))
done < "$ref"

len=${#fm_arr[@]}

#echo "$len"
#echo ${arr[0]}

for (( i=0; i<$len; i++ )); 
do 
 echo "" > fm_hyp.txt
 echo "" > sm_hyp.txt
 echo "" > ref.txt;

 echo "${fm_arr[$i]}" > fm_hyp.txt ; 
 echo "${sm_arr[$i]}" > sm_hyp.txt ; 
 echo "${ref_arr[$i]}" > ref.txt ; 

#echo "Evaluation with ribes score:";
fm_rs=`python ./RIBES-1.03.1/RIBES.py -r ref.txt fm_hyp.txt`
sm_rs=`python ./RIBES-1.03.1/RIBES.py -r ref.txt sm_hyp.txt`

#fm_rs=`python ./RIBES-1.03.1/RIBES.py -r $ref $fm`
#sm_rs=`python ./RIBES-1.03.1/RIBES.py -r $ref $fm`

#echo "$fm_rs";
#echo "$sm_rs";

if [[ "$fm_rs" > "$sm_rs" ]]; then

   echo "${fm_arr[$i]}" >> rs.txt

else

  echo "${sm_arr[$i]}" >> rs.txt	

fi 

done


