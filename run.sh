#!/usr/bin bash
export CUDA_VISIBLE_DEVICES=1
for dataset in 'hwu64' 'clinc' 'wos'
do
    for seed in 0 1 2
    do 
    python dna.py \
        --dataset $dataset\
        --seed $seed \
        --save_premodel \
        --save_premodel_path pre_model \
        --save_model \
        --save_model_path model
    done
done