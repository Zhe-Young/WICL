#!/bin/bash

tasks=('sst2' 'mr') # 'subj' 'agnews' 'cb' 'dbpedia' 'rte' 'boolq'
shots=(8 16)
model_size="1.3B" # "355M" "2.7B" "6.7B" 13B"
indicator='MSP' # 'validate_20'

for task in "${tasks[@]}"
do
    for shot in "${shots[@]}"
    do
        CUDA_VISIBLE_DEVICES=0 \
        python re_attention.py \
        --model "KoboldAI/fairseq-dense-${model_size}" \
        --task ${task} \
        --repeat_num 100 \
        --max_length 2000 \
        --balanced \
        --shot ${shot} \
        --beam_num 1 \
        --indicator "${indicator}" \
        --re_weight_place "before_softmax" \
        --weight_space "0.9 1.0 1.1" \
        --log_path "./Log/${model_size}_${task}_${shot}shot_beam1_${indicator}.json"
    done
done

