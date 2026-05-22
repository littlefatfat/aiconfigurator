"""
GLM-5 B300 SOL Prefill Analysis - Complete per-op SOL breakdown.

Hooks ALL database query methods to capture sol_time/sol_math/sol_mem for every
operator including attention (DSA), GEMM, MoE, etc.

Usage (in container mry-dpsk-v4):
  python3 /host/aiconfigurator/refactor_test_aic/run_glm5_sol_direct.py

Output: ./sol_debug_output/glm5_prefill_sol_breakdown.json
"""
import sys
import os
import json
import functools
from collections import defaultdict

sys.path.insert(0, "/host/aiconfigurator/refactor_test_aic")
sys.path.insert(0, "/host/aiconfigurator/src")

os.makedirs("./sol_debug_output", exist_ok=True)

from aiconfigurator.sdk import models
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
from aiconfigurator.sdk.perf_database import get_database


# =============================================================================
# SOLFullResult: wrapper that lets op.query() work with SOL_FULL tuple data
# =============================================================================
class SOLFullResult:
    """Wraps (sol_time, sol_math, sol_mem) tuple so float()/energy/source work."""

    def __init__(self, sol_time, sol_math, sol_mem):
        self.sol_time = sol_time
        self.sol_math = sol_math
        self.sol_mem = sol_mem
        self.energy = 0.0
        self.source = "sol"

    def __float__(self):
        return self.sol_time

    def __mul__(self, other):
        return float(self) * other

    def __rmul__(self, other):
        return other * float(self)

    def __add__(self, other):
        return float(self) + float(other)

    def __radd__(self, other):
        return float(other) + float(self)

    def __repr__(self):
        return f"SOLFullResult(time={self.sol_time:.4f}, math={self.sol_math:.4f}, mem={self.sol_mem:.4f})"


# =============================================================================
# Global log for capturing per-query SOL details
# =============================================================================
sol_call_log = []  # list of dicts: {method, sol_time, sol_math, sol_mem, args_summary}


def _patch_database_for_sol_full(database):
    """
    Monkey-patch all query_* methods on the database instance to:
    1. Call the original with explicit database_mode=SOL_FULL
    2. Wrap the returned tuple in SOLFullResult (supports float() and .energy)
    3. Log the SOL details to sol_call_log
    """
    query_methods = [
        "query_gemm",
        "query_compute_scale",
        "query_scale_matrix",
        "query_context_attention",
        "query_generation_attention",
        "query_context_mla",
        "query_generation_mla",
        "query_context_mla_module",
        "query_generation_mla_module",
        "query_wideep_generation_mla",
        "query_wideep_context_mla",
        "query_custom_allreduce",
        "query_nccl",
        "query_moe",
        "query_mla_bmm",
        "query_context_dsa_module",
        "query_generation_dsa_module",
    ]

    originals = {}
    for name in query_methods:
        if hasattr(database, name):
            # Get the bound method (includes lru_cache)
            originals[name] = getattr(database, name)

    def _make_wrapper(orig_fn, method_name):
        @functools.wraps(orig_fn)
        def wrapper(*args, **kwargs):
            # Force SOL_FULL mode explicitly (different lru_cache key)
            full_kwargs = dict(kwargs)
            full_kwargs["database_mode"] = DatabaseMode.SOL_FULL
            try:
                result = orig_fn(*args, **full_kwargs)
            except Exception as e:
                # Fallback: try SOL mode
                sol_kwargs = dict(kwargs)
                sol_kwargs["database_mode"] = DatabaseMode.SOL
                result = orig_fn(*args, **sol_kwargs)
                val = float(result)
                sol_call_log.append({
                    "method": method_name,
                    "sol_time": val,
                    "sol_math": val,
                    "sol_mem": 0.0,
                    "bottleneck": "UNKNOWN",
                    "error": str(e),
                })
                return result

            if isinstance(result, tuple) and len(result) >= 3:
                sol_time, sol_math, sol_mem = result[0], result[1], result[2]
                bottleneck = "MATH" if sol_math >= sol_mem else "MEM"
                sol_call_log.append({
                    "method": method_name,
                    "sol_time": sol_time,
                    "sol_math": sol_math,
                    "sol_mem": sol_mem,
                    "bottleneck": bottleneck,
                })
                return SOLFullResult(sol_time, sol_math, sol_mem)
            else:
                # Not a tuple - just return as-is
                val = float(result) if hasattr(result, "__float__") else 0.0
                sol_call_log.append({
                    "method": method_name,
                    "sol_time": val,
                    "sol_math": val,
                    "sol_mem": 0.0,
                    "bottleneck": "N/A",
                })
                return result
        return wrapper

    for name, orig in originals.items():
        setattr(database, name, _make_wrapper(orig, name))

    return originals


def _unpatch_database(database, originals):
    """Restore original methods."""
    for name, orig in originals.items():
        # Delete instance attribute to restore class-level method
        if hasattr(database, name):
            try:
                delattr(database, name)
            except AttributeError:
                pass


# =============================================================================
# Configuration: GLM-5 NVFP4 on B300 x 8 TP
# =============================================================================
MODEL_PATH = "/nfs/ZhipuAI/GLM-5.1-FP8/"
DEVICE_NAME = "b300_sxm"
BACKEND_NAME = "sglang"
BACKEND_VERSION = "0.5.10"
DATABASE_PATH = "/host/aiconfigurator/src/aiconfigurator/systems/"

model_config = ModelConfig(
    pp_size=1,
    tp_size=8,
    moe_tp_size=8,
    moe_ep_size=1,
    attention_dp_size=1,
    gemm_quant_mode=GEMMQuantMode.nvfp4,       # quantization=modelopt_fp4
    moe_quant_mode=MoEQuantMode.nvfp4,         # quantization=modelopt_fp4
    kvcache_quant_mode=KVCacheQuantMode.fp8,   # kv_cache_dtype=fp8_e4m3
    fmha_quant_mode=FMHAQuantMode.fp8,         # kv_cache_dtype=fp8_e4m3
    comm_quant_mode=CommQuantMode.fp8,
    workload_distribution="power_law_1.01",
)


def main():
    print("=" * 90)
    print("GLM-5 B300 SOL Prefill Analysis")
    print("=" * 90)
    print(f"Config: gemm={model_config.gemm_quant_mode.name}, "
          f"moe={model_config.moe_quant_mode.name}, "
          f"kvcache={model_config.kvcache_quant_mode.name}, "
          f"fmha={model_config.fmha_quant_mode.name}")
    print(f"  nvfp4: memory_factor={GEMMQuantMode.nvfp4.value.memory}, "
          f"compute_factor={GEMMQuantMode.nvfp4.value.compute}")
    print(f"  kvcache fp8: memory_factor={KVCacheQuantMode.fp8.value.memory}")
    print()

    # Load model
    perf_model = models.get_model(
        model_path=MODEL_PATH,
        model_config=model_config,
        backend_name=BACKEND_NAME,
    )
    print(f"Model: {perf_model.model_path}")
    print(f"Context ops ({len(perf_model.context_ops)}):")
    for i, op in enumerate(perf_model.context_ops):
        print(f"  [{i:2d}] {op._name}")
    print()

    # Load database
    database = get_database(
        system=DEVICE_NAME,
        backend=BACKEND_NAME,
        version=BACKEND_VERSION,
        systems_paths=[DATABASE_PATH],
        allow_missing_data=True,
    )
    assert database is not None, "Failed to load database"

    gpu = database.system_spec["gpu"]
    print(f"B300 SXM GPU specs:")
    print(f"  mem_bw:         {gpu['mem_bw']/1e12:.2f} TB/s")
    print(f"  bfloat16_tc:    {gpu['bfloat16_tc_flops']/1e15:.2f} PFLOPS")
    print(f"  fp8_tc:         {gpu['fp8_tc_flops']/1e15:.2f} PFLOPS")
    print(f"  fp4_tc:         {gpu['fp4_tc_flops']/1e15:.2f} PFLOPS")
    print()

    # Get backend
    backend = get_backend(BACKEND_NAME)

    # =========================================================================
    # Test cases: Prefill ISL = 4k, 16k, 32k, 64k, 128k, 256k
    # =========================================================================
    isl_values = [4096, 16384, 32768, 65536, 131072, 262144]
    batch_size = 1
    prefix = 0

    all_results = []

    for isl in isl_values:
        print(f"\n{'=' * 90}")
        print(f"PREFILL: bs={batch_size}, isl={isl} ({isl//1024}K), prefix={prefix}")
        print(f"{'=' * 90}")

        runtime_config = RuntimeConfig(batch_size=batch_size, isl=isl, prefix=prefix, osl=1)

        # --- Pass 1: Normal SOL mode for total latencies ---
        database.set_default_database_mode(DatabaseMode.SOL)
        results = backend._run_static_breakdown(
            perf_model, database, runtime_config, mode="static_ctx"
        )
        latency_dict = results[0]

        # --- Pass 2: SOL_FULL mode with hook to get per-op details ---
        global sol_call_log
        sol_call_log = []
        originals = _patch_database_for_sol_full(database)

        try:
            # Re-run context phase with patched database
            effective_isl = isl - prefix
            ctx_latency = defaultdict(float)
            op_details = []

            for op in perf_model.context_ops:
                x = batch_size * effective_isl if "logits_gemm" not in op._name else batch_size
                log_before = len(sol_call_log)

                result = op.query(
                    database,
                    x=x,
                    batch_size=batch_size,
                    beam_width=1,
                    s=effective_isl,
                    prefix=prefix,
                    model_name=getattr(perf_model, "model_name", ""),
                    seq_imbalance_correction_scale=runtime_config.seq_imbalance_correction_scale,
                )
                latency_ms = float(result)
                ctx_latency[op._name] += latency_ms

                # Collect log entries produced by this op
                log_after = len(sol_call_log)
                op_sol_entries = sol_call_log[log_before:log_after]

                if op_sol_entries:
                    # If op made multiple sub-calls, aggregate raw values
                    raw_sol_time = sum(e["sol_time"] for e in op_sol_entries)
                    raw_sol_math = sum(e["sol_math"] for e in op_sol_entries)
                    raw_sol_mem = sum(e["sol_mem"] for e in op_sol_entries)
                    # Apply scale: latency_ms = raw_sol_time * scale_factor
                    # (scale_factor = num_layers for most ops)
                    if raw_sol_time > 0:
                        scale = latency_ms / raw_sol_time
                    else:
                        scale = 1.0
                    scaled_math = raw_sol_math * scale
                    scaled_mem = raw_sol_mem * scale
                    bottleneck = "MATH" if scaled_math >= scaled_mem else "MEM"
                    op_details.append({
                        "op_name": op._name,
                        "latency_ms": round(latency_ms, 6),
                        "sol_time": round(latency_ms, 6),
                        "sol_math": round(scaled_math, 6),
                        "sol_mem": round(scaled_mem, 6),
                        "bottleneck": bottleneck,
                        "scale_factor": round(scale, 2),
                        "db_calls": op_sol_entries,
                    })
                else:
                    # Op didn't call database (e.g., ElementWise - pure bandwidth)
                    op_details.append({
                        "op_name": op._name,
                        "latency_ms": round(latency_ms, 6),
                        "sol_time": round(latency_ms, 6),
                        "sol_math": 0.0,
                        "sol_mem": round(latency_ms, 6),
                        "bottleneck": "MEM (bandwidth-only)",
                        "scale_factor": 1.0,
                        "db_calls": [],
                    })
        finally:
            _unpatch_database(database, originals)

        # --- Print results ---
        total = sum(d["latency_ms"] for d in op_details)
        print(f"\n  {'Op Name':<45} {'lat(ms)':>8} {'sol_math':>9} {'sol_mem':>9} {'bottleneck':>12} {'pct':>6}")
        print(f"  {'=' * 45} {'=' * 8} {'=' * 9} {'=' * 9} {'=' * 12} {'=' * 6}")

        for d in sorted(op_details, key=lambda x: -x["latency_ms"]):
            pct = d["latency_ms"] / total * 100 if total > 0 else 0
            print(f"  {d['op_name']:<45} {d['latency_ms']:>8.4f} "
                  f"{d['sol_math']:>9.4f} {d['sol_mem']:>9.4f} "
                  f"{d['bottleneck']:>12} {pct:>5.1f}%")

        print(f"  {'=' * 45} {'=' * 8}")
        print(f"  {'TOTAL':<45} {total:>8.4f} ms")

        # --- Store result ---
        case_result = {
            "isl": isl,
            "isl_k": f"{isl // 1024}K",
            "batch_size": batch_size,
            "prefix": prefix,
            "total_latency_ms": round(total, 4),
            "ops": op_details,
        }
        all_results.append(case_result)

    # =========================================================================
    # Summary table
    # =========================================================================
    print(f"\n\n{'=' * 90}")
    print(f"SUMMARY: GLM-5 NVFP4 Prefill SOL on B300 SXM x 8 TP")
    print(f"{'=' * 90}")
    print(f"  {'ISL':<10} {'Total(ms)':>10} {'Attn(ms)':>10} {'Attn%':>7} "
          f"{'MoE(ms)':>10} {'MoE%':>7} {'GEMM(ms)':>10} {'Other(ms)':>10}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 7} "
          f"{'-' * 10} {'-' * 7} {'-' * 10} {'-' * 10}")

    for r in all_results:
        total = r["total_latency_ms"]
        attn = sum(d["latency_ms"] for d in r["ops"] if "attention" in d["op_name"])
        moe_lat = sum(d["latency_ms"] for d in r["ops"]
                      if "moe" in d["op_name"] and "dispatch" not in d["op_name"])
        gemm = sum(d["latency_ms"] for d in r["ops"]
                   if "gemm" in d["op_name"] and "router" not in d["op_name"])
        other = total - attn - moe_lat - gemm
        attn_pct = attn / total * 100 if total > 0 else 0
        moe_pct = moe_lat / total * 100 if total > 0 else 0
        print(f"  {r['isl_k']:<10} {total:>10.2f} {attn:>10.2f} {attn_pct:>6.1f}% "
              f"{moe_lat:>10.2f} {moe_pct:>6.1f}% {gemm:>10.2f} {other:>10.2f}")

    # =========================================================================
    # Save to JSON
    # =========================================================================
    output_file = "./sol_debug_output/glm5_prefill_sol_breakdown.json"
    output = {
        "description": "GLM-5 NVFP4 Prefill SOL Breakdown on B300 SXM x8 TP",
        "config": {
            "model": "GLM-5.1-FP8 (nvfp4 quantized)",
            "device": "B300 SXM",
            "tp_size": 8,
            "gemm_quant": "nvfp4 (mem=9/16, compute=4x, fp4_tc=14PFLOPS)",
            "moe_quant": "nvfp4",
            "kvcache": "fp8 (mem=1B, fp8_tc=4.5PFLOPS)",
            "fmha": "fp8",
            "mem_bw": f"{gpu['mem_bw']/1e12:.2f} TB/s",
        },
        "cases": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")
    print("Done!")


if __name__ == "__main__":
    main()
