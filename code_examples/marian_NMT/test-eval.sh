#!/bin/bash

for i in {5000..55000..5000}
do
    marian-decoder -m ./model.iter$i.npz -v /home/ye/exp/nmt/my-rk/data/vocab/vocab.my.yml /home/ye/exp/nmt/my-rk/data/vocab/vocab.rk.yml --devices 0 --output hyp.iter$i.rk < /home/ye/exp/nmt/my-rk/data/test.my;
    echo "Evaluation with hyp.iter$i.th, Transformer model:" >> eval-result.txt;
    perl /home/ye/tool/mosesbin/ubuntu-17.04/moses/scripts/generic/multi-bleu.perl /home/ye/exp/nmt/my-rk/data/test.rk < ./hyp.iter$i.rk >> eval-result.txt;

done
