export HF_ENDPOINT="https://hf-mirror.com"

SGL_HOOK_REQ_INFO_DIR=`pwd`/tmp/server

CASE_NAME="${1:-H20-Deepseek-V4-Flash}"
BASE_OUT_DIR=`pwd`/tmp/L1/$CASE_NAME
mkdir -p $BASE_OUT_DIR

cp ${SGL_HOOK_REQ_INFO_DIR}/server.sh "${BASE_OUT_DIR}/server.sh"
cp "$(readlink -f "$0")" "${BASE_OUT_DIR}/bench.sh"


request_rates=(1 2 4 6 8 12 16 20 24 28 32)
for rate in "${request_rates[@]}"; do
    echo "Running the rate: $rate"

    curl http://localhost:30000/flush_cache
    rm -rf /tmp/hicache/ && mkdir -p /tmp/hicache
    curl http://localhost:30000/start_profile

    OUR_DIR=$BASE_OUT_DIR/data/$rate
    mkdir -p $OUR_DIR

    python3 -m sglang.bench_serving \
            --warmup-requests 0 \
            --dataset-name random \
            --num-prompts 100 \
            --request-rate $rate \
            --random-input-len 5000 \
            --random-output-len 512 \
            --random-range-ratio 0.5 \
            --dataset-path /host/aiconfigurator/ShareGPT_V3_unfiltered_cleaned_split.json \
            --output-file $OUR_DIR/no_cache.metrics.json

    curl http://localhost:30000/start_profile
    mv $SGL_HOOK_REQ_INFO_DIR/TP0.raw_request.jsonl $OUR_DIR/l1.requests.jsonl
    mv $SGL_HOOK_REQ_INFO_DIR/TP0.schedule_batch.jsonl $OUR_DIR/l1.schedule_batch.jsonl

done