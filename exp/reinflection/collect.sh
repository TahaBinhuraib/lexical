#!/bin/bash

std(){
  awk '{sum+=$1; sumsq+=$1*$1}END{print "± " sqrt(sumsq/NR - (sum/NR)**2)}'
}

mean(){
  awk 'BEGIN { sum=0 } { sum+=$1 } END {print sum / NR}'
}
min(){
  sort -n | head -1
}
max(){
  sort -n | tail -1
}


stdmean(){
  echo -n "TEST:"
  mu=$(printf "$1" | mean | tr -d '\n')
  sigma=$(printf "$1" | std | tr -d '\n')
  maximum=$(printf "$1" | max | tr -d '\n')
  minimum=$(printf "$1" | min | tr -d '\n')
  length=$(printf "$1" | wc -l)
  echo -n "$mu ($sigma) (max: $maximum , min: $minimum ) count ($length)"
  echo
}

for lr in 1.0; do
  for warmup_steps in 4000; do #orig: 4000
  for max_steps in 8000; do #orig 30000
  for n_batch in 128; do
    for prefix in  "pmi_" "soft_pmi_" "learned_pmi__" "0.5aligner_" "fast_forward_" "fast_intersect_" "soft_aligner_" "learn_aligner_" "goodman_" "soft_goodman_" "LSTM_" "LSTM_copy_";do
	expname=${prefix}COGS_lr_${lr}_warmup_${warmup_steps}_max_${max_steps}
      expname=simple_drop
      if [ -d "$expname" ]; then
        cd $expname
        numbers1=$(grep -oh 'test evaluation (greedy)/acc [0-9].*' *.out | awk '{print $4}' FS=" ")
        #numbers1=$(grep -oh 'train model/5500/best\_acc' *.out | awk '{print $4}' FS=" ")
        line1=$(echo "$numbers1" | wc -l)
        if [ "$line1" -lt 2 ]; then
          cd ..
          continue
        fi
        echo -n "${expname}"
        stdmean "$numbers1"
        cd ..
      else
        continue
      fi

    done
  done
done
done
done
