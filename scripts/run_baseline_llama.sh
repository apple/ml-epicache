# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

SCALE=${2} # 3, 7
LEVEL=${3} # pair (head-wise), uniform (uniform head-budget ratios)
KV_BUDGET=${4} 
PREFILL_CHUNK_SIZE=${5}

METHOD=${6}

if [ "$SCALE" = "1" ] || [ "$SCALE" = "3" ]; then
    MODEL=meta-llama/Llama-3.2-${SCALE}B-Instruct
else
    MODEL=meta-llama/Llama-3.1-${SCALE}B-Instruct
fi

DATA=${7} # locomo, realtalk, longmemeval
TARGET_LENGTH=${8} # only applicable to longmemeval

echo "================ EpiCache Configuration ================"
echo "DATA:              $DATA"
echo "MODEL:             $MODEL"
echo "LEVEL:             $LEVEL"
echo "CHUNK_SIZE:        $PREFILL_CHUNK_SIZE"
echo "METHOD:            $METHOD"
echo "KV_BUDGET:         $KV_BUDGET"
echo "TARGET_LENGTH:     $TARGET_LENGTH"
echo "========================================================="

CUDA_VISIBLE_DEVICES=$1 python -B run_baseline.py --model $MODEL --data $DATA --exp_name ${DATA}/baseline_${METHOD}_${KV_BUDGET}_${PREFILL_CHUNK_SIZE} --level $LEVEL \
--prefill_chunk_size $PREFILL_CHUNK_SIZE --kv_budget $KV_BUDGET --scoring_method $METHOD --verbose --target_length $TARGET_LENGTH