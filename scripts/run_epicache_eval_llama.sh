# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

SCALE=${2} # 3, 8
LEVEL=${3} # pair (head-wise), uniform (uniform head-budget ratios)
KV_BUDGET=${4} 
PREFILL_CHUNK_SIZE=${5}

METHOD=clustering 
N_CLUSTER=4 # number of episodes
N_MEDOID=8
EMBEDDING_TYPE=qwen # encoder model selection (sentence, llm, qwen)
CONV_WINDOW=4
DO_SCORE=${6} # whether to use sensitivity-based layer-wise budget allocation

if [ "$SCALE" = "1" ] || [ "$SCALE" = "3" ]; then
    MODEL=meta-llama/Llama-3.2-${SCALE}B-Instruct
    SCORE_PATH=data/layer_scores/booksum_Llama-3.2-${SCALE}B-Instruct_sample0_layer_scores.json
else
    MODEL=meta-llama/Llama-3.1-${SCALE}B-Instruct
    SCORE_PATH=data/layer_scores/booksum_Llama-3.1-${SCALE}B-Instruct_sample0_layer_scores.json
fi

if [ "$DO_SCORE" = "True" ]; then
    SCORE_PATH=$SCORE_PATH
    POWER=1.1
else
    SCORE_PATH=None
    POWER=0
fi

DATA=${7} # locomo, realtalk, longmemeval
TARGET_LENGTH=${8} # only applicable to longmemeval

echo "================ EpiCache Configuration ================"
echo "DATA:              $DATA"
echo "MODEL:             $MODEL"
echo "EMBEDDING_TYPE:    $EMBEDDING_TYPE"
echo "LEVEL:             $LEVEL"
echo "PREFILL_CHUNK_SIZE:$PREFILL_CHUNK_SIZE"
echo "METHOD:            $METHOD"
echo "N_CLUSTER:         $N_CLUSTER"
echo "N_MEDOID:          $N_MEDOID"
echo "CONV_WINDOW:       $CONV_WINDOW"
echo "KV_BUDGET:         $KV_BUDGET"
echo "DO_SCORE:          $DO_SCORE"
echo "POWER:             $POWER"
echo "========================================================="

CUDA_VISIBLE_DEVICES=$1 python -B run_epicache_eval.py --model $MODEL --data $DATA --embedding_type $EMBEDDING_TYPE \
    --exp_name ${DATA}/clustering_${KV_BUDGET}_${PREFILL_CHUNK_SIZE}_${N_CLUSTER}_${EMBEDDING_TYPE}_${POWER} --level $LEVEL --verbose \
    --prefill_chunk_size $PREFILL_CHUNK_SIZE --scoring_method $METHOD --n_cluster $N_CLUSTER --n_medoid $N_MEDOID \
    --conv_window $CONV_WINDOW --kv_budget $KV_BUDGET --score_path $SCORE_PATH --power $POWER --target_length $TARGET_LENGTH

