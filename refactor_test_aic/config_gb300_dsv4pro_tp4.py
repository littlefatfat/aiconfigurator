"""
统一配置模块 — DSv4-Pro on gb300 (L20B) tp=4
================================================
原 Flash/h20_sxm 配置已备份到 config.py.flash-orig
"""

import os

from aiconfigurator.sdk.common import (
    CommQuantMode,
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
)

# ============================================================================
# 数据目录
# ============================================================================

DATA_DIR = os.environ.get(
    'AIC_DATA_DIR',
    '/home/admin/maruiyan.mry/batch_info/gb300-deepseek-v4-pro/disable_overlap',
)

SCHEDULE_JSONL_FILENAME = 'TP0_schedule_batch.jsonl'

# ============================================================================
# 各阶段输出子目录
# ============================================================================
SUBDIR_CSV = 'csv'
SUBDIR_ESTIMATION = 'estimation'
SUBDIR_ACCURACY = 'accuracy'
SUBDIR_SIGNED_ERROR = 'signed_error'

# ============================================================================
# 模型 / 后端 — DSv4-Pro
# ============================================================================
MODEL_NAME = 'deepseek-ai/DeepSeek-V4-Pro'
MODEL_PATH = '/home/admin/resource-slow/model/464482ce/DeepSeek-V4-Pro/04242026'
BACKEND_NAME = 'sglang'

# AIC 性能数据库：gb300 + sglang + 我们采集的版本号
# 对应 src/aiconfigurator/systems/data/gb300/sglang/0.5.11/ 下的 *_perf.txt
AIC_SYSTEM = 'gb300'
AIC_BACKEND = 'sglang'
AIC_VERSION = '0.5.11_660b28976_dsv4pro'

# ============================================================================
# ModelConfig — 对应 sglang 启动命令:
#   --tp 4 --disable-overlap-schedule
#   --kv-cache-dtype fp8_e4m3
#   --moe-runner-backend flashinfer_mxfp4
#   (无 --ep / --pp / --dp 显式参数 → 默认 1)
# ============================================================================

MODEL_CONFIG_KWARGS = dict(
    # 并行
    pp_size=1,
    tp_size=4,
    moe_tp_size=4,
    moe_ep_size=1,
    attention_dp_size=1,

    # MoE 扩展（DSv4-Pro 单机 tp=4，不开 wideep/EPLB）
    enable_wideep=False,
    enable_eplb=False,
    wideep_num_slots=None,
    moe_backend=None,                      # wideep only (e.g. "deepep")
    sms=20,                                # wideep partitioning, default 20

    # MTP（DSv4-Pro sglang 启动未开 speculative decoding）
    nextn=0,
    nextn_accept_rates=None,

    # 模型构建
    overwrite_num_layers=0,                # 0 = use model's native layer count
    attention_backend='flashinfer',        # aic 内部按 DSv4 arch dispatch 到 CSA/HCA

    # workload
    workload_distribution='power_law_1.01',

    # 量化：DSv4-Pro 是 FP8 checkpoint
    # gemm/moe 用 fp8_block；缺数据时切到 w4a8_mxfp4_mxfp8 (对应 --moe-runner-backend flashinfer_mxfp4)
    gemm_quant_mode=GEMMQuantMode.fp8_block,
    moe_quant_mode=MoEQuantMode.w4a8_mxfp4_mxfp8,
    kvcache_quant_mode=KVCacheQuantMode.fp8,
    fmha_quant_mode=FMHAQuantMode.bfloat16,
    comm_quant_mode=CommQuantMode.half,
)

# ============================================================================
# SGLang Server 启动命令 (供 nsys_profiler 解析，复用你启动 sgl 的命令)
# ============================================================================
SGLANG_LAUNCH_CMD = '''
python3 /home/admin/maruiyan.mry/refactor_test_aic/hook_dataset_collector/sglang_launch_server.py \\
  --trust-remote-code \\
  --model-path /home/admin/resource-slow/model/464482ce/DeepSeek-V4-Pro/04242026/ \\
  --tp 4 \\
  --disable-overlap-schedule \\
  --cuda-graph-max-bs 128 \\
  --chunked-prefill-size 8192 \\
  --disable-cuda-graph-padding \\
  --max-running-requests 256 \\
  --mem-fraction-static 0.93 \\
  --attention-backend compressed \\
  --moe-runner-backend flashinfer_mxfp4 \\
  --kv-cache-dtype fp8_e4m3 \\
  --tool-call-parser deepseekv4 \\
  --reasoning-parser deepseek-v4 \\
  --disable-flashinfer-autotune \\
  --allow-auto-truncate
'''

# ============================================================================
# 估算校正系数 (原值保留，从 Flash 配置继承)
# ============================================================================
DECODE_CORRECTION_FACTOR = 1.0
PREFILL_CORRECTION_FACTOR = 1.02


def get_output_dir(data_dir: str, subdir: str) -> str:
    path = os.path.join(data_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# Scenario-aware predictor overrides
# ----------------------------------------------------------------------------
# AIC 当前不建模 CPU bubble，短 prefill 时 GPU 时间 << host overhead，
# 实测会出现一个 "地板"。地板大小随 (hw, tp, model) 显著变化，
# 经验值需要按场景标定。这里同时支持 scale_factor 做整体缩放。
#
# Key:   (AIC_SYSTEM, tp_size, model_substring)
# Value: 转发给 stage2 的 knob dict, 字段对齐 hisim-sglang predictor:
#          prefill_scale_factor: float = 1.0
#          decode_scale_factor:  float = 1.0
#          prefill_min_latency:  float = 0.0  # ms, prefill 输出的下限
# ============================================================================
PREDICTOR_OVERRIDES = {
    # H20 tp=8 DeepSeek-V4-Flash: 来源 = 原 stage2 里写死的 118ms 经验值
    ('h20_sxm', 8, 'DeepSeek-V4-Flash'): dict(
        prefill_scale_factor=1.0,
        decode_scale_factor=1.0,
        prefill_min_latency=118.0,
    ),
    # GB300 tp=4 DeepSeek-V4-Pro: 来源 = 本次 disable_overlap bench
    # 实测 short-prefill 的 min/median 分别是 241ms / 295ms
    ('gb300', 4, 'DeepSeek-V4-Pro'): dict(
        prefill_scale_factor=1.0,
        decode_scale_factor=1.0,
        prefill_min_latency=241.0,
    ),
}


def get_predictor_overrides() -> dict:
    """按 (AIC_SYSTEM, tp_size, model_substring) 查 PREDICTOR_OVERRIDES 中的 knob。

    匹配规则: hw / tp 精确相等; model_substring 出现在 MODEL_NAME 或 MODEL_PATH 中即可。
    未命中时返回空 dict, stage2 将退化为 (scale=1, floor=0) — 即关闭新逻辑。
    """
    hw = AIC_SYSTEM
    tp = MODEL_CONFIG_KWARGS.get('tp_size')
    haystack = (MODEL_NAME or '') + '|' + (MODEL_PATH or '')
    for (h, t, m), knobs in PREDICTOR_OVERRIDES.items():
        if h == hw and t == tp and m in haystack:
            return dict(knobs)
    return {}
