#!/bin/bash

languages=("ame" "bra" "ckt" "itl")
seeds=(0 1 2)

max_step=2000
warmup_steps=1000
valid_steps=100
cd ..

for i in ${seeds[@]}; do
    for j in ${languages[@]}; do
        python main.py \
        --seed $i\
        --language $j\
        --max_step ${max_step}\
        --warmup_steps ${warmup_steps}\
        --valid_steps ${valid_steps}
    done
done
