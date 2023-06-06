#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATA_BIN=$YOUR_DATA

python train.py $DATA_BIN \
  --task long_denoising \
  --max-epoch 130 \
  --arch "bart_large" \
  --use-xformers \
  --attention-name block_noglobal \
  --pooling-layers 4 \
  --fast-stat-sync \
  --criterion cross_entropy \
  --truncate-source \
  --truncate-target \
  --required-seq-len-multiple 1024 \
  --max-source-positions 16384 \
  --max-target-positions 1024 \
  --update-freq 4 \
  --batch-size 2 \
  --optimizer "adam" \
  --adam-betas "(0.9, 0.98)" \
  --clip-norm 0.1 \
  --lr 1e-4 \
  --checkpoint-activations \
  --lr-scheduler "polynomial_decay" \
  --warmup-updates 500 \
  --seed=3 \
  --total-num-update 10000 \
  --num-workers 4 \
  --skip-invalid-size-inputs-valid-test \
  --combine-val \
  --no-epoch-checkpoints \
  --memory-efficient-fp16 \
  --log-format json \
  --log-interval 10 \
  --custom-dict ../checkpoints/dict.txt \
  --restore-file ../checkpoints/model_100k.pt
