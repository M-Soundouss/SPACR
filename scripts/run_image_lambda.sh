#!/bin/bash

set -o pipefail

declare -a DATASETS=("utkface" "drift")
declare -a SEEDS=("93" "753" "1416" "2001" "6789")
declare -a LAMBDAS=("0.01" "0.05" "0.1" "0.5" "1.0" "10.0" "50.0" "100.0" "500.0" "1000.0" "5000.0" "10000.0")
declare -a EPSILONS=("0.01" "0.05" "0.1")

# Training parameters
MAX_EPOCH=200
TRAIN_BATCH_SIZE=256
DEFAULT_TRAIN_EPSILON="0.01"

# Evaluation parameters
EVAL_BATCH_SIZE=512

BASE_RESULTS_DIR="experiment_results_lambda"
mkdir -p "$BASE_RESULTS_DIR"

TIMING_CSV="${BASE_RESULTS_DIR}/overall_timings.csv"
if [ ! -f "$TIMING_CSV" ]; then
    echo "Dataset,Seed,Method,Lambda,Epsilon,Beta_Value,Training_Time_Sec,Inference_Time_Sec" > "$TIMING_CSV"
fi

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do

      echo -e "\n****************************************************************"
      echo "🎯 STARTING SPACR IMAGE LAMBDA SWEEP | Dataset: $dataset | Seed: $seed"
      echo "****************************************************************"

      for lam in "${LAMBDAS[@]}"; do
          LAMBDA_DIR="${BASE_RESULTS_DIR}/lambda_${lam}"
          OUTPUT_DIR="${LAMBDA_DIR}/output"
          CHECKPOINT_DIR="${LAMBDA_DIR}/checkpoint"

          mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

          RESULTS_FILE="${LAMBDA_DIR}/${dataset}_seed_${seed}_lambda_${lam}.log"
          > "$RESULTS_FILE"

          echo -e "\n==========================================================" | tee -a "$RESULTS_FILE"
          echo "🚀 STARTING TRAINING: SPACR | Lambda: $lam" | tee -a "$RESULTS_FILE"
          echo "==========================================================" | tee -a "$RESULTS_FILE"

          start_time=$(date +%s.%N)

          python train_images.py \
              --model-name "$dataset" \
              --method "SPACR" \
              --max-epoch "$MAX_EPOCH" \
              --batch-size "$TRAIN_BATCH_SIZE" \
              --epsilon "$DEFAULT_TRAIN_EPSILON" \
              --lam "$lam" \
              --seed "$seed" \
              --checkpoint-dir "$CHECKPOINT_DIR" >> "$RESULTS_FILE"

          if [ $? -ne 0 ]; then
              echo "❌ ERROR: Training failed for lambda=$lam" | tee -a "$RESULTS_FILE"
              continue
          fi

          end_time=$(date +%s.%N)
          TRAIN_TIME=$(python -c "print(${end_time} - ${start_time})")

          echo -e "----------------------------------------------------------" | tee -a "$RESULTS_FILE"
          echo "✅ TRAINING COMPLETED: SPACR (Lambda: $lam)" | tee -a "$RESULTS_FILE"
          echo "⏱️  Time Taken: $TRAIN_TIME seconds" | tee -a "$RESULTS_FILE"
          echo -e "----------------------------------------------------------" | tee -a "$RESULTS_FILE"

          for epsilon in "${EPSILONS[@]}"; do
              echo "📊 EVALUATING: SPACR for epsilon: $epsilon" | tee -a "$RESULTS_FILE"

              python conformal_evaluate_images.py \
                  --model-name "$dataset" \
                  --method "SPACR" \
                  --batch-size "$EVAL_BATCH_SIZE" \
                  --epsilon "$epsilon" \
                  --seed "$seed" \
                  --checkpoint-dir "$CHECKPOINT_DIR" \
                  --output-dir "$OUTPUT_DIR" >> "$RESULTS_FILE"

              TIMING_FILE="$OUTPUT_DIR/$dataset/seed_$seed/SPACR/timing_eps_$epsilon.txt"
              if [ -f "$TIMING_FILE" ]; then
                  INF_TIME=$(grep "Total Inference Time" "$TIMING_FILE" | awk '{print $4}')
                  BETA_VAL=$(grep "Beta Parameter" "$TIMING_FILE" | awk '{print $3}')
                  echo "$dataset,$seed,SPACR,$lam,$epsilon,$BETA_VAL,$TRAIN_TIME,$INF_TIME" >> "$TIMING_CSV"
              fi
          done

          echo "💾 Saved metrics to $TIMING_CSV" | tee -a "$RESULTS_FILE"
          echo "Completed lambda=$lam" | tee -a "$RESULTS_FILE"
      done

      echo "Finished all lambdas for $dataset (seed $seed)"
      echo -e "\n"
  done
done

echo -e "\n🎉 ALL IMAGE LAMBDA SWEEP EXPERIMENTS COMPLETED SUCESSFULLY! 🎉"
echo "Results and CSVs are cleanly stored in ${BASE_RESULTS_DIR}/"