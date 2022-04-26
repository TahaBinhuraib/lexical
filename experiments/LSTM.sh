#!/bin/bash

languages=("ame" "bra" "ckt" "itl")
seeds=(0)
bathces=(20 40)
lrs=(1)
lstmdims=(128 256)
dropouts=(0.1 0.3)
lstmlayers=(4 6)
max_step=2000
warmup_steps=1000
valid_steps=100

cd ..

for seed in ${seeds[@]}; do
    for language in ${languages[@]}; do
        for lr in ${lrs[@]}; do
            for batch in ${bathces[@]}; do
                for lstmdim in ${lstmdims[@]}; do
                    for dropout in ${dropouts[@]}; do
                        for lstmlayer in ${lstmlayers[@]}; do
                        
                            python main.py \
                            --seed $seed\
                            --language $language\
                            --n_layers $lstmlayer\
                            --n_batch $batch\
                            --lr $lr\
                            --dropout $dropout\
                            --dim $lstmdim\
                            --max_step ${max_step}\
                            --warmup_steps ${warmup_steps}\
                            --valid_steps ${valid_steps}\
                            --copy True > ./logs/logs_seed_{$seed}_lan_{$language}_batch_{$batch}_dim_{$dim}_layer_{$lstmlayer}.txt 2> ./logs/err_seed_{$seed}_lan_{$language}_batch_{$batch}_dim_{$dim}_layer_{$lstmlayer}.txt

                        done
                    done
                done
            done 
        done
    done
done
