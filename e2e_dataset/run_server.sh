export HF_ENDPOINT=https://hf-mirror.com
export SGL_HOOK_REQ_INFO_DIR="$(pwd)/tmp/server"

# Create target directory
mkdir -p "${SGL_HOOK_REQ_INFO_DIR}"
# Copy current script to target directory
cp "$(readlink -f "$0")" "${SGL_HOOK_REQ_INFO_DIR}/server.sh"

SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_DSV4_FP4_EXPERTS=0 \
python3 sgl_launch_server.py \
    --model-path="/nfs/sgl-project/DeepSeek-V4-Flash-FP8/" \
    --trust-remote-code \
    --disable-overlap-schedule \
    --tp-size=8 \
    --cuda-graph-max-bs 256 \
    --max-running-requests 256 \
    --disable-cuda-graph-padding