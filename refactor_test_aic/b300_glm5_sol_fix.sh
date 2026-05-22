#!/bin/bash
# ==============================================================================
# B300 GLM5 SOL 模式 debug 修复脚本
# ==============================================================================
# 诊断结果：
#   Bug1: l3.hisim_config.glm5.json 中 data_type="FP8" 应为 "FP4"
#         → GEMM/MoE SOL sol_math 被高估 3.11x (fp8_tc=4.5PF vs fp4_tc=14PF)
#         → GEMM/MoE SOL sol_mem 被高估 1.78x (fp8 mem=1B vs nvfp4 mem=9/16B)
#   Bug2: l3.hisim_config.glm5.json 中 kv_cache_data_type="FP16" 应为 "FP8"
#         → Attention SOL 的 KV cache 访存量被高估 2x (bf16=2B vs fp8=1B)
# ==============================================================================

CONFIG_DIR="/host/insight_benchmark/test/hisim/mry_debug/one_case"
CONFIG_FILE="${CONFIG_DIR}/l3.hisim_config.glm5.json"

echo "=== 修复 l3.hisim_config.glm5.json ==="
echo "Before:"
cat "${CONFIG_FILE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  data_type: {d[\"scheduler\"][\"data_type\"]}'); print(f'  kv_cache_data_type: {d[\"scheduler\"][\"kv_cache_data_type\"]}')"

# 修复
python3 -c "
import json
with open('${CONFIG_FILE}') as f:
    config = json.load(f)
config['scheduler']['data_type'] = 'FP4'
config['scheduler']['kv_cache_data_type'] = 'FP8'
with open('${CONFIG_FILE}', 'w') as f:
    json.dump(config, f, indent=4)
print('Fixed!')
"

echo "After:"
cat "${CONFIG_FILE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  data_type: {d[\"scheduler\"][\"data_type\"]}'); print(f'  kv_cache_data_type: {d[\"scheduler\"][\"kv_cache_data_type\"]}')"
echo ""
echo "=== Done ==="
