"""Print single-phase static per-op data for GLM-5 on vLLM."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import aiconfigurator.sdk.operations as sdk_ops
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.common import (
    CommQuantMode,
    DatabaseMode,
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
)
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database

MODEL_PATH = "/models/Qwen3-235B-A22B-Instruct-2507-FP8"
SYSTEM_NAME = "h20_sxm"
BACKEND_NAME = "sglang"
BACKEND_VERSION = "0.5.9"
PHASE_TO_MODE = {
    "prefill": "static_ctx",
    "decode": "static_gen",
}

phase = sys.argv[1].lower() if len(sys.argv) > 1 else "prefill"
if phase not in PHASE_TO_MODE:
    raise SystemExit(
        f"Usage: python aic_perop3.py [prefill|decode] [past_kv_length]  (got: {phase})"
    )
mode = PHASE_TO_MODE[phase]
past_kv_length = int(sys.argv[2]) if len(sys.argv) > 2 else 0
if phase == "prefill":
    past_kv_length = 0

database = get_database(SYSTEM_NAME, BACKEND_NAME, BACKEND_VERSION)
if database is None:
    raise RuntimeError(f"Failed to load database for {SYSTEM_NAME}/{BACKEND_NAME}/{BACKEND_VERSION}")
database.set_default_database_mode(DatabaseMode.SILICON)

backend = get_backend(BACKEND_NAME)
model_config = ModelConfig(
    tp_size=8,
    pp_size=1,
    moe_tp_size=1,
    moe_ep_size=8,
    gemm_quant_mode=GEMMQuantMode.fp8_block,
    moe_quant_mode=MoEQuantMode.fp8_block,
    # gemm_quant_mode=GEMMQuantMode.float16,
    # moe_quant_mode=MoEQuantMode.float16,
    kvcache_quant_mode=KVCacheQuantMode.float16,
    fmha_quant_mode=FMHAQuantMode.float16,
    comm_quant_mode=CommQuantMode.half,
)
model = get_model(MODEL_PATH, model_config, BACKEND_NAME)

runtime_config = RuntimeConfig(
    batch_size=51,
    beam_width=1,
    isl=3786,
    osl=2,
)

session = InferenceSession(model, database, backend)
summary = session.run_static(runtime_config=runtime_config, mode=mode, stride=32)

context_ops = summary.get_context_latency_dict()
generation_ops = summary.get_generation_latency_dict()
result_dict = summary.get_result_dict() or {}

selected_ops = context_ops if mode == "static_ctx" else generation_ops
selected_label = "Prefill" if mode == "static_ctx" else "Decode"


def _safe_latency(entry):
    if isinstance(entry, dict):
        return float(entry.get("latency", 0.0))
    return float(entry)


def _safe_energy(entry):
    if isinstance(entry, dict):
        return float(entry.get("energy", 0.0))
    return 0.0


def _print_decode_moe_interp_debug():
    if mode != "static_gen":
        return

    # Keep this aligned with BaseBackend._run_generation_phase().
    query_x = runtime_config.batch_size * (getattr(model, "_nextn", 0) + 1) * runtime_config.beam_width
    moe_debug_rows = []

    for op in model.generation_ops:
        if not isinstance(op, sdk_ops.MoE):
            continue

        query_tokens = int(query_x * getattr(op, "_attention_dp_size", 1))
        quant_mode = op._quant_mode
        workload_distribution = op._workload_distribution
        topk = op._topk
        num_experts = op._num_experts
        hidden_size = op._hidden_size
        inter_size = op._inter_size
        moe_tp_size = op._moe_tp_size
        moe_ep_size = op._moe_ep_size
        scale_factor = float(op._scale_factor)
        moe_backend = getattr(op, "_moe_backend", None)

        if database.backend == "sglang" and moe_backend == "deepep_moe":
            moe_data = database._wideep_generation_moe_data
        else:
            moe_data = database._moe_data
        moe_data.raise_if_not_loaded()

        used_workload_distribution = workload_distribution if workload_distribution in moe_data[quant_mode] else "uniform"
        moe_dict = moe_data[quant_mode][used_workload_distribution][topk][num_experts][hidden_size][inter_size][
            moe_tp_size
        ][moe_ep_size]
        token_points = sorted(moe_dict.keys())

        overflow = query_tokens > token_points[-1]
        if overflow:
            left_tok, right_tok = token_points[-2], token_points[-1]
            interp_raw = None
            interp_scaled = None
        else:
            left_tok, right_tok = database._nearest_1d_point_helper(query_tokens, token_points, inner_only=False)
            interp_result = database._interp_1d(
                [left_tok, right_tok],
                [moe_dict[left_tok], moe_dict[right_tok]],
                query_tokens,
            )
            interp_raw = _safe_latency(interp_result)
            interp_scaled = interp_raw * scale_factor

        left_entry = moe_dict[left_tok]
        right_entry = moe_dict[right_tok]
        op_total_latency = float(selected_ops.get(op._name, 0.0))

        moe_debug_rows.append(
            {
                "op_name": op._name,
                "query_num_tokens": query_tokens,
                "used_workload_distribution": used_workload_distribution,
                "left_token": left_tok,
                "right_token": right_tok,
                "left_latency_ms_raw": _safe_latency(left_entry),
                "right_latency_ms_raw": _safe_latency(right_entry),
                "left_energy_wms_raw": _safe_energy(left_entry),
                "right_energy_wms_raw": _safe_energy(right_entry),
                "interp_latency_ms_raw": interp_raw,
                "interp_latency_ms_scaled": interp_scaled,
                "op_scale_factor": scale_factor,
                "op_total_latency_ms_from_summary": op_total_latency,
                "overflow_above_max_collected_tokens": overflow,
                "max_collected_token": token_points[-1],
            }
        )

    if moe_debug_rows:
        print()
        print("=== Decode MoE interpolation debug ===")
        print(json.dumps(moe_debug_rows, indent=2, default=str))


print(f"Database Mode: {DatabaseMode.SILICON.name}")
print(f"Phase: {selected_label} ({mode})")
print(f"past_kv_length(prefix): {past_kv_length}")
print(f"{selected_label} Latency: {sum(selected_ops.values()):.3f} ms")
print()
print(f"=== {selected_label} per-op breakdown ===")
print(json.dumps(selected_ops, indent=2, default=str))
print()
_print_decode_moe_interp_debug()
print()
print("=== Full raw result ===")
print(json.dumps(result_dict, indent=2, default=str))
