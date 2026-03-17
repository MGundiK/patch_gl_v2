#!/bin/bash

# ============================================================
# MULTI-SEED TABLE 14 — TRAFFIC
# ============================================================
# Config: sl=720, lr=0.005, batch=46
# Seeds: 2022, 2023, 2024 (seed 2021 already done)
# Estimated: ~3-4 hrs per seed × 3 = ~9-12 hrs
# NOTE: Traffic is the slowest dataset due to 862 channels

alpha=0.3; beta=0.3; model_name=GLPatch
LOGDIR="./logs/multiseed_t14/Traffic"
mkdir -p ${LOGDIR}

for seed in 2022 2023 2024; do
  echo ""
  echo ">>> [$(date '+%H:%M:%S')] Traffic seed=${seed} sl=720 lr=0.005"

  sed -i "s/fix_seed = [0-9]*/fix_seed = ${seed}/" run.py

  sdir="${LOGDIR}/seed${seed}"; mkdir -p ${sdir}
  for pl in 96 192 336 720; do
    echo "  [$(date '+%H:%M:%S')] Traffic seed=${seed} pl=${pl}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path traffic.csv \
      --model_id ms_Traffic_s${seed}_${pl} --model $model_name --data custom \
      --features M --seq_len 720 --pred_len $pl --enc_in 862 \
      --des 'Exp' --itr 1 --batch_size 46 --learning_rate 0.005 \
      --lradj 'sigmoid' --ma_type ema --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pl}.log
  done

  echo "  === Traffic seed=${seed} complete ==="
  for pl in 96 192 336 720; do
    result=$(grep "mse:" ${sdir}/${pl}.log | tail -1)
    echo "    pl=${pl}: ${result}"
  done
  echo ""
done

sed -i "s/fix_seed = [0-9]*/fix_seed = 2021/" run.py
echo "Restored fix_seed = 2021"
echo ""
echo "========== TRAFFIC MULTI-SEED COMPLETE [$(date '+%H:%M:%S')] =========="
