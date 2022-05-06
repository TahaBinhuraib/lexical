#!/bin/bash

languages=("ame" "bra")
seeds=(1 2)
bathces=(20)
lrs=(1)
lstmdims=(256)
dropouts=(0.2)
lstmlayers=(2 4)
max_step=2
warmup_steps=1
valid_steps=1

cd ..
mkdir -p logs
mkdir -p logs/time_exp

SECONDS=0
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
                            --copy True > ./logs/logs_seed_{$seed}_lan_{$language}_batch_{$batch}_dim_{$lstmdim}_layer_{$lstmlayer}_lr{$lr}_dropout_{$dropout}_copy_true.txt 2> ./logs/err_seed_{$seed}_lan_{$language}_batch_{$batch}_dim_{$lstmdim}_layer_{$lstmlayer}_lr{$lr}_dropout_{$dropout}_copy_true.txt

                        done
                    done
                done
            done 
        done
    done
done
ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ${ELAPSED} > ./logs/time_exp/time_copy_true.txt