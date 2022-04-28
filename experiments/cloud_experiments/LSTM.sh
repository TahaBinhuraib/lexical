#!/bin/bash
#SBATCH --job-name=morph
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:1
#SBATCH --array=0-3887

languages=("ame" "bra" "ckt" "itl")
seeds=(0 1 2)
bathces=(20 40 120)
lrs=(1 0.1 0.01)
lstmdims=(128 256 512)
dropouts=(0.1 0.3 0.4)
lstmlayers=(1 2 3 4)
max_step=20000
warmup_steps=4000
valid_steps=500


i=0
mkdir -p logs

for seed in ${seeds[@]}; do
    for language in ${languages[@]}; do
        for lr in ${lrs[@]}; do
            for batch in ${bathces[@]}; do
                for lstmdim in ${lstmdims[@]}; do
                    for dropout in ${dropouts[@]}; do
                        for lstmlayer in ${lstmlayers[@]}; do
                        if [[ $i -eq $SLURM_ARRAY_TASK_ID ]]; then
                        
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
                                rm checkpoint.pt
                        fi
                        i=$(( i + 1 ))

                        done
                    done
                done
            done 
        done
    done
done