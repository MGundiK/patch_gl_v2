#!/bin/bash

# ============================================================
# MULTI-SEED TABLE 14 — SOLAR
# ============================================================
# Config: sl=720, lr=0.005, batch=256
# Seeds: 2022, 2023, 2024 (seed 2021 already done)
# Estimated: ~2 hrs per seed × 3 = ~6 hrs

alpha=0.3; beta=0.3; model_name=GLPatch
LOGDIR="./logs/multiseed_t14/Solar"
mkdir -p ${LOGDIR}

for seed in 2022 2023 2024; do
  echo ""
  echo ">>> [$(date '+%H:%M:%S')] Solar seed=${seed} sl=720 lr=0.005"

  sed -i "s/fix_seed = [0-9]*/fix_seed = ${seed}/" run.py

  sdir="${LOGDIR}/seed${seed}"; mkdir -p ${sdir}
  for pl in 96 192 336 720; do
    echo "  [$(date '+%H:%M:%S')] Solar seed=${seed} pl=${pl}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path solar.txt \
      --model_id ms_Solar_s${seed}_${pl} --model $model_name --data Solar \
      --features M --seq_len 720 --pred_len $pl --enc_in 137 \
      --des 'Exp' --itr 1 --batch_size 256 --learning_rate 0.005 \
      --lradj 'sigmoid' --ma_type ema --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pl}.log
  done

  echo "  === Solar seed=${seed} complete ==="
  for pl in 96 192 336 720; do
    result=$(grep "mse:" ${sdir}/${pl}.log | tail -1)
    echo "    pl=${pl}: ${result}"
  done
  echo ""
done

sed -i "s/fix_seed = [0-9]*/fix_seed = 2021/" run.py
echo "Restored fix_seed = 2021"
echo ""
echo "========== SOLAR MULTI-SEED COMPLETE [$(date '+%H:%M:%S')] =========="
