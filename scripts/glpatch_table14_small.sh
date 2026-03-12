#!/bin/bash

# ============================================================
# GLPatch TABLE 14 — HYPERPARAMETER SEARCH
# ============================================================
# Matches xPatch Table 14 configs EXACTLY:
#   - Same seq_len per dataset
#   - Same LR per dataset (critical — drops to 0.0001 at longer seq_lens)
#   - Same batch_size per dataset
#   - ILI uses ma_type=reg (no EMA decomposition)
#
# Datasets split into small (this script) and large (separate script).
#
# This script: ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange, ILI
# Estimated: ~3-4 hours
#
# xPatch also ran this at seq_len={336,512,720} and picked the best.
# The Table 14 script has ETT at 336, but they searched all.
# We replicate the BEST config from their search.

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch
LOGDIR="./logs/table14"

echo ""
echo "========== [$(date '+%H:%M:%S')] GLPatch TABLE 14 — SMALL DATASETS =========="
echo ""

# ============================================================
# ETT datasets: seq_len=336, lr=0.0001 (xPatch T14 config)
# NOTE: xPatch searched {336,512,720}, we do all 3 to find our best
# ============================================================

for sl in 336 512 720; do
  for ds_info in "ETTh1:ETTh1.csv:ETTh1:7" "ETTh2:ETTh2.csv:ETTh2:7" "ETTm1:ETTm1.csv:ETTm1:7" "ETTm2:ETTm2.csv:ETTm2:7"; do
    IFS=':' read -r ds data_path data_flag enc_in <<< "$ds_info"
    sdir="${LOGDIR}/sl${sl}/${ds}"
    mkdir -p ${sdir}

    echo ">>> [$(date '+%H:%M:%S')] ${ds} sl=${sl} lr=0.0001"
    for pred_len in 96 192 336 720; do
      echo "  [$(date '+%H:%M:%S')] ${ds} sl=${sl} pl=${pred_len}"
      python -u run.py \
        --is_training 1 --root_path ./dataset/ --data_path ${data_path} \
        --model_id t14_${ds}_sl${sl}_${pred_len}_${ma_type} --model $model_name --data ${data_flag} \
        --features M --seq_len $sl --pred_len $pred_len --enc_in $enc_in \
        --des 'Exp' --itr 1 --batch_size 2048 --learning_rate 0.0001 \
        --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
        --use_amp --num_workers 2 \
        2>&1 | tee ${sdir}/${pred_len}.log
    done

    echo "  === ${ds} sl=${sl} complete ==="
    for pred_len in 96 192 336 720; do
      result=$(grep "mse:" ${sdir}/${pred_len}.log | tail -1)
      echo "    pl=${pred_len}: ${result}"
    done
    echo ""
  done
done

# ============================================================
# Weather: seq_len=512, lr=0.0001, batch=1024 (xPatch T14 config)
# Also try 336 and 720
# ============================================================

for sl in 336 512 720; do
  sdir="${LOGDIR}/sl${sl}/Weather"
  mkdir -p ${sdir}

  # batch adjustments for memory
  if [ $sl -le 512 ]; then batch=1024; else batch=512; fi

  echo ">>> [$(date '+%H:%M:%S')] Weather sl=${sl} lr=0.0001 batch=${batch}"
  for pred_len in 96 192 336 720; do
    echo "  [$(date '+%H:%M:%S')] Weather sl=${sl} pl=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path weather.csv \
      --model_id t14_Weather_sl${sl}_${pred_len}_${ma_type} --model $model_name --data custom \
      --features M --seq_len $sl --pred_len $pred_len --enc_in 21 \
      --des 'Exp' --itr 1 --batch_size $batch --learning_rate 0.0001 \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pred_len}.log
  done

  echo "  === Weather sl=${sl} complete ==="
  for pred_len in 96 192 336 720; do
    result=$(grep "mse:" ${sdir}/${pred_len}.log | tail -1)
    echo "    pl=${pred_len}: ${result}"
  done
  echo ""
done

# ============================================================
# Exchange: seq_len=96, lr=0.0001 (xPatch T14 uses higher LR than T13!)
# xPatch T14 stays at sl=96 but changes LR from 0.00001 to 0.0001
# Also try our tuned lr=0.000005 at seq_len=96 for comparison
# ============================================================

for lr in 0.0001 0.000005; do
  sdir="${LOGDIR}/sl96_lr${lr}/Exchange"
  mkdir -p ${sdir}

  echo ">>> [$(date '+%H:%M:%S')] Exchange sl=96 lr=${lr}"
  for pred_len in 96 192 336 720; do
    echo "  [$(date '+%H:%M:%S')] Exchange sl=96 lr=${lr} pl=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path exchange_rate.csv \
      --model_id t14_Exchange_lr${lr}_${pred_len}_${ma_type} --model $model_name --data custom \
      --features M --seq_len 96 --pred_len $pred_len --enc_in 8 \
      --des 'Exp' --itr 1 --batch_size 32 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pred_len}.log
  done

  echo "  === Exchange lr=${lr} complete ==="
  for pred_len in 96 192 336 720; do
    result=$(grep "mse:" ${sdir}/${pred_len}.log | tail -1)
    echo "    pl=${pred_len}: ${result}"
  done
  echo ""
done

# ============================================================
# ILI: seq_len=36, lr=0.01, ma_type=REG (no EMA!)
# xPatch T14 switches to ma_type=reg for ILI
# ============================================================

sdir="${LOGDIR}/sl36_reg/ILI"
mkdir -p ${sdir}

echo ">>> [$(date '+%H:%M:%S')] ILI sl=36 lr=0.01 ma_type=reg"
for pred_len in 24 36 48 60; do
  echo "  [$(date '+%H:%M:%S')] ILI reg pl=${pred_len}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path national_illness.csv \
    --model_id t14_ILI_reg_${pred_len} --model $model_name --data custom \
    --features M --seq_len 36 --label_len 18 --pred_len $pred_len --enc_in 7 \
    --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.01 \
    --lradj 'type3' --patch_len 6 --stride 3 \
    --ma_type reg --alpha $alpha --beta $beta \
    --use_amp --num_workers 2 \
    2>&1 | tee ${sdir}/${pred_len}.log
done

echo "  === ILI (reg) complete ==="
for pred_len in 24 36 48 60; do
  result=$(grep "mse:" ${sdir}/${pred_len}.log | tail -1)
  echo "    pl=${pred_len}: ${result}"
done
echo ""

# Also try ILI with ema at different seq_lens for comparison
for sl in 48 72 104; do
  sdir="${LOGDIR}/sl${sl}_ema/ILI"
  mkdir -p ${sdir}

  ll=$((sl / 2))
  echo ">>> [$(date '+%H:%M:%S')] ILI sl=${sl} lr=0.01 ma_type=ema"
  for pred_len in 24 36 48 60; do
    echo "  [$(date '+%H:%M:%S')] ILI sl=${sl} ema pl=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path national_illness.csv \
      --model_id t14_ILI_sl${sl}_ema_${pred_len} --model $model_name --data custom \
      --features M --seq_len $sl --label_len $ll --pred_len $pred_len --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.01 \
      --lradj 'type3' --patch_len 6 --stride 3 \
      --ma_type ema --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pred_len}.log
  done

  echo "  === ILI sl=${sl} ema complete ==="
  for pred_len in 24 36 48 60; do
    result=$(grep "mse:" ${sdir}/${pred_len}.log | tail -1)
    echo "    pl=${pred_len}: ${result}"
  done
  echo ""
done

# ============================================================
# ALSO: Re-run ETT + Weather at lr=0.0001 with seq_len=96
# This tests whether lr=0.0001 is better even at short lookback
# (since our tuned LRs were 0.0005-0.0007 for these)
# ============================================================

echo ">>> [$(date '+%H:%M:%S')] ETT + Weather at sl=96, lr=0.0001 (xPatch T14 LR)"
for ds_info in "ETTh1:ETTh1.csv:ETTh1:7" "ETTh2:ETTh2.csv:ETTh2:7" "ETTm1:ETTm1.csv:ETTm1:7" "ETTm2:ETTm2.csv:ETTm2:7" "Weather:weather.csv:custom:21"; do
  IFS=':' read -r ds data_path data_flag enc_in <<< "$ds_info"
  sdir="${LOGDIR}/sl96_lr0.0001/${ds}"
  mkdir -p ${sdir}

  echo ">>> [$(date '+%H:%M:%S')] ${ds} sl=96 lr=0.0001"
  for pred_len in 96 192 336 720; do
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ${data_path} \
      --model_id t14_${ds}_sl96_lr0001_${pred_len}_${ma_type} --model $model_name --data ${data_flag} \
      --features M --seq_len 96 --pred_len $pred_len --enc_in $enc_in \
      --des 'Exp' --itr 1 --batch_size 2048 --learning_rate 0.0001 \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 \
      2>&1 | tee ${sdir}/${pred_len}.log
  done

  echo "  === ${ds} sl=96 lr=0.0001 complete ==="
  for pred_len in 96 192 336 720; do
    result=$(grep "mse:" ${sdir}/${pred_len}.log | tail -1)
    echo "    pl=${pred_len}: ${result}"
  done
  echo ""
done

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "========== [$(date '+%H:%M:%S')] TABLE 14 SMALL DATASETS COMPLETE =========="
echo ""
echo "Results summary:"
echo "================"

for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
  echo ""
  echo "${ds} (lr=0.0001):"
  echo "  seq_len |     96      192      336      720"
  echo "  --------+------------------------------------"
  for sl in 96 336 512 720; do
    dir="${LOGDIR}/sl${sl}/${ds}"
    if [ "$sl" = "96" ]; then dir="${LOGDIR}/sl96_lr0.0001/${ds}"; fi
    printf "  %6d  |" $sl
    for pred_len in 96 192 336 720; do
      logfile="${dir}/${pred_len}.log"
      if [ -f "$logfile" ]; then
        mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
        [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
      else
        printf "     N/A"
      fi
    done
    echo ""
  done
done

echo ""
echo "Weather (lr=0.0001):"
echo "  seq_len |     96      192      336      720"
echo "  --------+------------------------------------"
for sl in 96 336 512 720; do
  dir="${LOGDIR}/sl${sl}/Weather"
  if [ "$sl" = "96" ]; then dir="${LOGDIR}/sl96_lr0.0001/Weather"; fi
  printf "  %6d  |" $sl
  for pred_len in 96 192 336 720; do
    logfile="${dir}/${pred_len}.log"
    if [ -f "$logfile" ]; then
      mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
    else
      printf "     N/A"
    fi
  done
  echo ""
done

echo ""
echo "Exchange (sl=96):"
for lr in 0.0001 0.000005; do
  dir="${LOGDIR}/sl96_lr${lr}/Exchange"
  printf "  lr=%-10s|" $lr
  for pred_len in 96 192 336 720; do
    logfile="${dir}/${pred_len}.log"
    if [ -f "$logfile" ]; then
      mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
    else
      printf "     N/A"
    fi
  done
  echo ""
done

echo ""
echo "ILI:"
for config in "sl36_reg:reg" "sl48_ema:ema48" "sl72_ema:ema72" "sl104_ema:ema104"; do
  IFS=':' read -r dirkey label <<< "$config"
  dir="${LOGDIR}/${dirkey}/ILI"
  printf "  %-10s|" $label
  for pred_len in 24 36 48 60; do
    logfile="${dir}/${pred_len}.log"
    if [ -f "$logfile" ]; then
      mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
    else
      printf "     N/A"
    fi
  done
  echo ""
done

echo ""
echo "Total runs: ~140+ (ETT×4×3×4 + Weather×3×4 + Exchange×2×4 + ILI×4×4 + ETT/W×5×4)"
echo "Log files: ${LOGDIR}/<config>/<dataset>/<pred_len>.log"
