#!/bin/bash

## Written by Ye Kyaw Thu, LST, NECTEC, Thailand
## Experiments for my-rk demo for Internship3 Students

## Old Notes
#     --mini-batch-fit -w 10000 --maxi-batch 1000 \
#    --mini-batch-fit -w 1000 --maxi-batch 100 \
#     --tied-embeddings-all \
#     --tied-embeddings \
#     --valid-metrics cross-entropy perplexity translation bleu \
#     --transformer-dropout 0.1 --label-smoothing 0.1 \
#     --learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
#     --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \

mkdir -p my-rk/model.tf.myrk;

marian \
    --model my-rk/model.tf.myrk/model.npz --type transformer \
    --train-sets /home/ye/exp/nmt/my-rk/data/train.my \
    /home/ye/exp/nmt/my-rk/data/train.rk \
    --max-length 200 \
    --vocabs /home/ye/exp/nmt/my-rk/data/vocab/vocab.my.yml \
    /home/ye/exp/nmt/my-rk/data/vocab/vocab.rk.yml \
    --mini-batch-fit -w 1000 --maxi-batch 100 \
    --early-stopping 10 \
    --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
    --valid-metrics cross-entropy perplexity bleu \
    --valid-sets /home/ye/exp/nmt/my-rk/data/dev.my \
    /home/ye/exp/nmt/my-rk/data/dev.rk \
    --valid-translation-output my-rk/model.tf.myrk/valid.my-rk.output \
    --quiet-translation \
    --valid-mini-batch 64 \
    --beam-size 6 --normalize 0.6 \
    --log my-rk/model.tf.myrk/train.log \
    --valid-log my-rk/model.tf.myrk/valid.log \
    --enc-depth 2 --dec-depth 2 \
    --transformer-heads 8 \
    --transformer-postprocess-emb d \
    --transformer-postprocess dan \
    --transformer-dropout 0.3 --label-smoothing 0.1 \
    --learn-rate 0.0003 --lr-warmup 0 --lr-decay-inv-sqrt 16000 --lr-report \
    --clip-norm 5 \
    --tied-embeddings \
    --devices 0 --sync-sgd --seed 1111 \
    --exponential-smoothing \
    --dump-config > my-rk/model.tf.myrk/my-rk.config.yml

time marian -c my-rk/model.tf.myrk/my-rk.config.yml  2>&1 | tee my-rk/model.tf.myrk/transformer-myrk.log

