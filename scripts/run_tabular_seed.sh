#!/bin/bash

declare -a DATASETS=("bike_sharing" "brazilian_houses" "california" "cpu_act" "diamonds" "fifa" "house_sales" "isolet" "medical_charges" "pol" "superconduct" "wine_quality")
declare -a METHODS=("SICP" "NICP" "CQR" "DOICR" "SPACR" "SPACR_EFF" "SPACR_VAL")
declare -a EPSILONS=("0.01" "0.05" "0.1")
declare -a SEEDS=("93" "753" "1416" "2001" "6789")

DEFAULT_EPSILON="0.01"

# Training parameters
MAX_EPOCH=200
TRAIN_BATCH_SIZE=256
EVAL_BATCH_SIZE=512
LAM=5.0

BASE_RESULTS_DIR="results"
RESULTS_DIR="${BASE_RESULTS_DIR}/experimental_results"

mkdir -p "$RESULTS_DIR"

TIMING_CSV="${BASE_RESULTS_DIR}/overall_timings.csv"
if [ ! -f "$TIMING_CSV" ]; then
    echo "Dataset,Seed,Method,Epsilon,Beta_Value,Training_Time_Sec,Inference_Time_Sec" > "$TIMING_CSV"
fi

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
      RESULTS_FILE="${BASE_RESULTS_DIR}/${dataset}_seed_${seed}_master_results.log"
      > "$RESULTS_FILE"

      echo -e "\n****************************************************************" | tee -a "$RESULTS_FILE"
      echo "🎯 STARTING EXPERIMENTS | Dataset: $dataset | Seed: $seed" | tee -a "$RESULTS_FILE"
      echo "****************************************************************" | tee -a "$RESULTS_FILE"

      for method in "${METHODS[@]}"; do

          # ==============================================================================
          # GROUP 1: Methods that depend on Epsilon during training (CQR, DOICR)
          # ==============================================================================
          if [ "$method" = "DOICR" ] || [ "$method" = "CQR" ]; then
              for epsilon in "${EPSILONS[@]}"; do

                  echo -e "\n==========================================================" | tee -a "$RESULTS_FILE"
                  echo "🚀 STARTING TRAINING: $method | Dataset: $dataset | Epsilon: $epsilon" | tee -a "$RESULTS_FILE"
                  echo "==========================================================" | tee -a "$RESULTS_FILE"

                  start_time=$(date +%s.%N)
                  python train_tabular.py --model-name "$dataset" --method "$method" \
                      --max-epoch "$MAX_EPOCH" --batch-size "$TRAIN_BATCH_SIZE" \
                      --epsilon "$epsilon" --lam "$LAM" --seed "$seed" > /dev/null 2>&1

                  if [ $? -ne 0 ]; then
                      echo "❌ ERROR: Training failed for $method (eps=$epsilon)" | tee -a "$RESULTS_FILE"
                      continue
                  fi

                  end_time=$(date +%s.%N)
                  TRAIN_TIME=$(python -c "print(${end_time} - ${start_time})")

                  echo -e "----------------------------------------------------------" | tee -a "$RESULTS_FILE"
                  echo "✅ TRAINING COMPLETED: $method (Epsilon: $epsilon)" | tee -a "$RESULTS_FILE"
                  echo "⏱️  Time Taken: $TRAIN_TIME seconds" | tee -a "$RESULTS_FILE"
                  echo -e "----------------------------------------------------------" | tee -a "$RESULTS_FILE"

                  echo "📊 EVALUATING: $method for epsilon: $epsilon" | tee -a "$RESULTS_FILE"

                  python conformal_evaluate_tabular.py --model-name "$dataset" --method "$method" \
                      --batch-size "$EVAL_BATCH_SIZE" --epsilon "$epsilon" --seed "$seed" \
                      --output-dir "$RESULTS_DIR" >> "$RESULTS_FILE"

                  TIMING_FILE="$RESULTS_DIR/$dataset/seed_$seed/$method/timing_eps_$epsilon.txt"
                  if [ -f "$TIMING_FILE" ]; then
                      INF_TIME=$(grep "Total Inference Time" "$TIMING_FILE" | awk '{print $4}')
                      BETA_VAL=$(grep "Beta Parameter" "$TIMING_FILE" | awk '{print $3}')
                      echo "$dataset,$seed,$method,$epsilon,$BETA_VAL,$TRAIN_TIME,$INF_TIME" >> "$TIMING_CSV"
                  fi

                  echo "💾 Saved metrics to $TIMING_CSV" | tee -a "$RESULTS_FILE"
              done

          # ==============================================================================
          # GROUP 2: Methods that are Epsilon-Agnostic during training (SICP, NICP, SPACR)
          # ==============================================================================
          else
              echo -e "\n==========================================================" | tee -a "$RESULTS_FILE"
              echo "🚀 STARTING BASE TRAINING: $method | Dataset: $dataset | Default Epsilon: $DEFAULT_EPSILON" | tee -a "$RESULTS_FILE"
              echo "==========================================================" | tee -a "$RESULTS_FILE"

              start_time=$(date +%s.%N)

              python train_tabular.py --model-name "$dataset" --method "$method" \
                  --max-epoch "$MAX_EPOCH" --batch-size "$TRAIN_BATCH_SIZE" \
                  --epsilon "$DEFAULT_EPSILON" --lam "$LAM" --seed "$seed" > /dev/null 2>&1

              if [ $? -ne 0 ]; then
                  echo "❌ ERROR: Training failed for $method" | tee -a "$RESULTS_FILE"
                  continue
              fi

              end_time=$(date +%s.%N)
              TRAIN_TIME=$(python -c "print(${end_time} - ${start_time})")

              echo -e "----------------------------------------------------------" | tee -a "$RESULTS_FILE"
              echo "✅ BASE TRAINING COMPLETED: $method" | tee -a "$RESULTS_FILE"
              echo "⏱️  Time Taken: $TRAIN_TIME seconds" | tee -a "$RESULTS_FILE"
              echo -e "----------------------------------------------------------" | tee -a "$RESULTS_FILE"

              for epsilon in "${EPSILONS[@]}"; do
                  echo "📊 EVALUATING: $method for epsilon: $epsilon" | tee -a "$RESULTS_FILE"

                  python conformal_evaluate_tabular.py --model-name "$dataset" --method "$method" \
                      --batch-size "$EVAL_BATCH_SIZE" --epsilon "$epsilon" --seed "$seed" \
                      --output-dir "$RESULTS_DIR" >> "$RESULTS_FILE"

                  TIMING_FILE="$RESULTS_DIR/$dataset/seed_$seed/$method/timing_eps_$epsilon.txt"
                  if [ -f "$TIMING_FILE" ]; then
                      INF_TIME=$(grep "Total Inference Time" "$TIMING_FILE" | awk '{print $4}')
                      BETA_VAL=$(grep "Beta Parameter" "$TIMING_FILE" | awk '{print $3}')
                      echo "$dataset,$seed,$method,$epsilon,$BETA_VAL,$TRAIN_TIME,$INF_TIME" >> "$TIMING_CSV"
                  fi
              done
              echo "💾 Saved metrics to $TIMING_CSV" | tee -a "$RESULTS_FILE"
          fi
      done
  done
done

echo -e "\n🎉 ALL EXPERIMENTS COMPLETED SUCESSFULLY! 🎉"