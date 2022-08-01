#!/bin/bash
#SBATCH --job-name=morph
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:1
#SBATCH --array=0-71

languages=("ail" "itl" "ame" "bra" "ckt")
seeds=(1 2)
batch=120
lr=1
lstmdim=256
dropout=0.4
lstmlayer=2
max_step=20000
warmup_steps=4000
valid_steps=500


i=0
mkdir -p logs
mkdir -p logs/time_exp
SECONDS=0
for seed in ${seeds[@]}; do
    for language in ${languages[@]}; do
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
                --copy True > ./logs/logs_seed_${seed}_lan_${language}_batch_${batch}_dim_${lstmdim}_layer_${lstmlayer}_lr${lr}_dropout_${dropout}_copy_true.txt 2> ./logs/err_seed_${seed}_lan_${language}_batch_${batch}_dim_${lstmdim}_layer_${lstmlayer}_lr${lr}_dropout_${dropout}_copy_true.txt
        fi
        i=$(( i + 1 ))

    done
done
ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ${ELAPSED} > ./logs/time_exp/time_copy_true.txt
