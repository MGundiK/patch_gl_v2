#!/bin/bash

# ============================================================
# MULTI-SEED TABLE 14 — SMALL DATASETS
# ============================================================
# ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange, ILI
# Seeds: 2022, 2023, 2024 (seed 2021 already done)
# Estimated: ~4 hrs per seed × 3 = ~12 hrs

alpha=0.3; beta=0.3; model_name=GLPatch
LOGDIR="./logs/multiseed_t14"
mkdir -p ${LOGDIR}

for seed in 2022 2023 2024; do
  echo ""
  echo "################################################################"
  echo "  SEED = ${seed} — [$(date '+%H:%M:%S')]"
  echo "################################################################"

  sed -i "s/fix_seed = [0-9]*/fix_seed = ${seed}/" run.py
  echo "  Set fix_seed = ${seed}"

  # ---- ETTh1: sl=512, lr=0.0001 ----
  ds=ETTh1; sdir="${LOGDIR}/${ds}/seed${seed}"; mkdir -p ${sdir}
  echo ">>> [$(date '+%H:%M:%S')] ${ds} seed=${seed}"
  for pl in 96 192 336 720; do
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv \
      --model_id ms_${ds}_s${seed}_${pl} --model $model_name --data ETTh1 \
      --features M --seq_len 512 --pred_len $pl --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 2048 --learning_rate 0.0001 \
      --lradj 'sigmoid' --ma_type ema --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pl}.log
  done
  echo "  === ${ds} seed=${seed} complete ==="
  for pl in 96 192 336 720; do echo "    pl=${pl}: $(grep 'mse:' ${sdir}/${pl}.log | tail -1)"; done

  # ---- ETTh2: sl=720, lr=0.0001 ----
  ds=ETTh2; sdir="${LOGDIR}/${ds}/seed${seed}"; mkdir -p ${sdir}
  echo ">>> [$(date '+%H:%M:%S')] ${ds} seed=${seed}"
  for pl in 96 192 336 720; do
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ETTh2.csv \
      --model_id ms_${ds}_s${seed}_${pl} --model $model_name --data ETTh2 \
      --features M --seq_len 720 --pred_len $pl --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 2048 --learning_rate 0.0001 \
      --lradj 'sigmoid' --ma_type ema --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pl}.log
  done
  echo "  === ${ds} seed=${seed} complete ==="
  for pl in 96 192 336 720; do echo "    pl=${pl}: $(grep 'mse:' ${sdir}/${pl}.log | tail -1)"; done

  # ---- ETTm1: sl=336, lr=0.0001 ----
  ds=ETTm1; sdir="${LOGDIR}/${ds}/seed${seed}"; mkdir -p ${sdir}
  echo ">>> [$(date '+%H:%M:%S')] ${ds} seed=${seed}"
  for pl in 96 192 336 720; do
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ETTm1.csv \
      --model_id ms_${ds}_s${seed}_${pl} --model $model_name --data ETTm1 \
      --features M --seq_len 336 --pred_len $pl --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 2048 --learning_rate 0.0001 \
      --lradj 'sigmoid' --ma_type ema --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pl}.log
  done
  echo "  === ${ds} seed=${seed} complete ==="
  for pl in 96 192 336 720; do echo "    pl=${pl}: $(grep 'mse:' ${sdir}/${pl}.log | tail -1)"; done

  # ---- ETTm2: sl=720, lr=0.00005 ----
  ds=ETTm2; sdir="${LOGDIR}/${ds}/seed${seed}"; mkdir -p ${sdir}
  echo ">>> [$(date '+%H:%M:%S')] ${ds} seed=${seed}"
  for pl in 96 192 336 720; do
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv \
      --model_id ms_${ds}_s${seed}_${pl} --model $model_name --data ETTm2 \
      --features M --seq_len 720 --pred_len $pl --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 2048 --learning_rate 0.00005 \
      --lradj 'sigmoid' --ma_type ema --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pl}.log
  done
  echo "  === ${ds} seed=${seed} complete ==="
  for pl in 96 192 336 720; do echo "    pl=${pl}: $(grep 'mse:' ${sdir}/${pl}.log | tail -1)"; done

  # ---- Weather: sl=512, lr=0.0003 (IMPROVED) ----
  ds=Weather; sdir="${LOGDIR}/${ds}/seed${seed}"; mkdir -p ${sdir}
  echo ">>> [$(date '+%H:%M:%S')] ${ds} seed=${seed} (lr=0.0003 IMPROVED)"
  for pl in 96 192 336 720; do
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path weather.csv \
      --model_id ms_${ds}_s${seed}_${pl} --model $model_name --data custom \
      --features M --seq_len 512 --pred_len $pl --enc_in 21 \
      --des 'Exp' --itr 1 --batch_size 1024 --learning_rate 0.0003 \
      --lradj 'sigmoid' --ma_type ema --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pl}.log
  done
  echo "  === ${ds} seed=${seed} complete ==="
  for pl in 96 192 336 720; do echo "    pl=${pl}: $(grep 'mse:' ${sdir}/${pl}.log | tail -1)"; done

  # ---- Exchange: sl=96, lr=0.000005 ----
  ds=Exchange; sdir="${LOGDIR}/${ds}/seed${seed}"; mkdir -p ${sdir}
  echo ">>> [$(date '+%H:%M:%S')] ${ds} seed=${seed}"
  for pl in 96 192 336 720; do
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path exchange_rate.csv \
      --model_id ms_${ds}_s${seed}_${pl} --model $model_name --data custom \
      --features M --seq_len 96 --pred_len $pl --enc_in 8 \
      --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.000005 \
      --lradj 'sigmoid' --ma_type ema --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pl}.log
  done
  echo "  === ${ds} seed=${seed} complete ==="
  for pl in 96 192 336 720; do echo "    pl=${pl}: $(grep 'mse:' ${sdir}/${pl}.log | tail -1)"; done

  # ---- ILI: sl=36, lr=0.01, ma_type=reg ----
  ds=ILI; sdir="${LOGDIR}/${ds}/seed${seed}"; mkdir -p ${sdir}
  echo ">>> [$(date '+%H:%M:%S')] ${ds} seed=${seed} (ma_type=reg)"
  for pl in 24 36 48 60; do
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path national_illness.csv \
      --model_id ms_${ds}_s${seed}_${pl} --model $model_name --data custom \
      --features M --seq_len 36 --label_len 18 --pred_len $pl --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.01 \
      --lradj 'type3' --patch_len 6 --stride 3 \
      --ma_type reg --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pl}.log
  done
  echo "  === ${ds} seed=${seed} complete ==="
  for pl in 24 36 48 60; do echo "    pl=${pl}: $(grep 'mse:' ${sdir}/${pl}.log | tail -1)"; done

  echo ""
  echo "========== SEED ${seed} SMALL DATASETS COMPLETE [$(date '+%H:%M:%S')] =========="
  echo ""
done

sed -i "s/fix_seed = [0-9]*/fix_seed = 2021/" run.py
echo "Restored fix_seed = 2021"
echo ""
echo "========== ALL SMALL DATASET SEEDS COMPLETE =========="
