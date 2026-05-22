"""
B300 GLM5 SOL Debug 补丁 —— 为 AIC predictor 增加详细的 per-operator SOL 日志。
将此文件放到容器中，用作 monkey-patch 或直接替换原文件的 predict_infer_latency_dict 方法。

使用方法（在 simulate_one_case.py 的 import 之后加入）：
    from b300_glm5_sol_debug_patch import patch_aiconfigurator_predictor
    patch_aiconfigurator_predictor()

或者直接把 patched 的 predict_infer_latency_dict / predict_infer_time 拷贝到
/host/hisim-sglang/sglang/tools/sglang-simulator/src/sglang_simulator/time_predictor/aiconfigurator.py
"""

import os
import json
import time
from collections import defaultdict
from typing import Optional

import numpy as np


# ============================================================
# 详细 SOL 日志输出
# ============================================================

SOL_DEBUG_OUTPUT_DIR = os.environ.get("SOL_DEBUG_OUTPUT_DIR", "./sol_debug_output")
os.makedirs(SOL_DEBUG_OUTPUT_DIR, exist_ok=True)

_batch_counter = 0
_all_batch_results = []


def _get_sol_details_for_operator(database, op, **query_kwargs):
    """
    用 SOL_FULL 模式查询单个算子，获取 (sol_time, sol_math, sol_mem) 三元组。
    """
    from aiconfigurator.sdk.common import DatabaseMode

    # 临时切换到 SOL_FULL 获取详细信息
    orig_mode = database.get_default_database_mode()
    database.set_default_database_mode(DatabaseMode.SOL_FULL)
    try:
        result = op.query(database, **query_kwargs)
        if isinstance(result, tuple) and len(result) == 3:
            sol_time, sol_math, sol_mem = result
        else:
            sol_time = float(result)
            sol_math = sol_time
            sol_mem = 0.0
    except Exception as e:
        sol_time = sol_math = sol_mem = -1.0
        print(f"  [SOL_FULL query failed for {op._name}]: {e}")
    finally:
        database.set_default_database_mode(orig_mode)

    return sol_time, sol_math, sol_mem


def patched_predict_infer_latency_dict(self, batch):
    """
    增强版 predict_infer_latency_dict，输出每个算子的 SOL 详细分解。
    """
    from aiconfigurator.sdk.config import RuntimeConfig
    from aiconfigurator.sdk.common import DatabaseMode

    global _batch_counter, _all_batch_results
    _batch_counter += 1
    batch_id = _batch_counter

    is_decode = batch.is_decode()
    phase = "decode" if is_decode else "prefill"

    # 构造 RuntimeConfig（与原逻辑一致）
    if is_decode:
        isl = int(np.mean([req.past_kv_length for req in batch.reqs]))
        runtime_config = RuntimeConfig(batch_size=batch.batch_size, isl=isl, osl=2)
    else:
        mean_past = np.mean([req.past_kv_length for req in batch.reqs])
        mean_input = np.mean([req.extend_length for req in batch.reqs])
        isl = int(mean_past + mean_input)
        prefix = int(mean_past)
        seq_imbalance_correction_scale = self.ctx_attn_flops_ratio_with_avg(batch.reqs)
        if seq_imbalance_correction_scale >= 0.4:
            runtime_config = RuntimeConfig(
                batch_size=batch.batch_size, isl=isl, prefix=prefix, osl=1,
                seq_imbalance_correction_scale=seq_imbalance_correction_scale,
            )
        else:
            runtime_config = RuntimeConfig(
                batch_size=batch.batch_size, isl=isl, prefix=prefix, osl=1
            )

    # 获取常规 latency_dict（用 SOL 模式）
    results = self._session._backend._run_static_breakdown(
        self._session._model,
        self._session._database,
        runtime_config,
        mode="static_gen" if is_decode else "static_ctx",
    )
    latency_dict = results[2] if is_decode else results[0]

    # === 详细 SOL 信息采集 ===
    model = self._session._model
    database = self._session._database
    ops = model.generation_ops if is_decode else model.context_ops
    batch_size = runtime_config.batch_size
    effective_isl = runtime_config.isl - runtime_config.prefix if not is_decode else runtime_config.isl

    batch_detail = {
        "batch_id": batch_id,
        "phase": phase,
        "batch_size": batch.batch_size,
        "isl": int(isl),
        "prefix": int(runtime_config.prefix) if not is_decode else 0,
        "effective_isl": int(effective_isl) if not is_decode else int(isl),
        "total_latency_ms": sum(latency_dict.values()),
        "model_config": {
            "gemm_quant_mode": model.config.gemm_quant_mode.name,
            "moe_quant_mode": model.config.moe_quant_mode.name,
            "kvcache_quant_mode": model.config.kvcache_quant_mode.name,
            "fmha_quant_mode": model.config.fmha_quant_mode.name,
            "tp_size": model.config.tp_size,
            "moe_tp_size": model.config.moe_tp_size,
            "moe_ep_size": model.config.moe_ep_size,
        },
        "operators": [],
    }

    print(f"\n{'='*80}")
    print(f"[SOL DEBUG] Batch #{batch_id} | {phase} | bs={batch.batch_size} | isl={isl}")
    if not is_decode:
        print(f"  prefix={runtime_config.prefix} | effective_isl={effective_isl}")
    print(f"  quant: gemm={model.config.gemm_quant_mode.name}, "
          f"moe={model.config.moe_quant_mode.name}, "
          f"kvcache={model.config.kvcache_quant_mode.name}, "
          f"fmha={model.config.fmha_quant_mode.name}")
    print(f"  parallel: tp={model.config.tp_size}, moe_tp={model.config.moe_tp_size}, "
          f"moe_ep={model.config.moe_ep_size}")
    print(f"{'─'*80}")
    print(f"  {'Op Name':<45} {'sol_time':>10} {'sol_math':>10} {'sol_mem':>10} {'bottleneck':>12}")
    print(f"  {'─'*45} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")

    for op in ops:
        if is_decode:
            x = batch_size
            query_kwargs = dict(
                x=x, batch_size=batch_size, beam_width=1,
                s=isl + 1,
                model_name=getattr(model, "model_name", ""),
            )
        else:
            x = batch_size * effective_isl if "logits_gemm" not in op._name else batch_size
            query_kwargs = dict(
                x=x, batch_size=batch_size, beam_width=1,
                s=effective_isl, prefix=runtime_config.prefix,
                model_name=getattr(model, "model_name", ""),
                seq_imbalance_correction_scale=runtime_config.seq_imbalance_correction_scale,
            )

        sol_time, sol_math, sol_mem = _get_sol_details_for_operator(
            database, op, **query_kwargs
        )

        bottleneck = "MATH" if sol_math >= sol_mem else "MEM"
        op_latency = latency_dict.get(op._name, 0.0)

        print(f"  {op._name:<45} {sol_time:>10.4f} {sol_math:>10.4f} {sol_mem:>10.4f} {bottleneck:>12}")

        batch_detail["operators"].append({
            "name": op._name,
            "sol_time_ms": round(sol_time, 6),
            "sol_math_ms": round(sol_math, 6),
            "sol_mem_ms": round(sol_mem, 6),
            "latency_ms": round(op_latency, 6),
            "bottleneck": bottleneck,
        })

    total = sum(latency_dict.values())
    print(f"  {'─'*45} {'─'*10}")
    print(f"  {'TOTAL':<45} {total:>10.4f} ms")
    print(f"{'='*80}\n")

    _all_batch_results.append(batch_detail)

    # 每 50 个 batch 或每次 prefill 都写一次文件
    if batch_id % 50 == 0 or phase == "prefill":
        output_file = os.path.join(SOL_DEBUG_OUTPUT_DIR, "sol_debug_all_batches.jsonl")
        with open(output_file, "w") as f:
            for item in _all_batch_results:
                f.write(json.dumps(item) + "\n")

    return latency_dict


def patched_predict_infer_time(self, batch):
    """与原逻辑一致，仅调用 patched 版的 predict_infer_latency_dict"""
    latency_dict = patched_predict_infer_latency_dict(self, batch)
    infer_time = sum(latency_dict.values())

    if getattr(self, '_is_oom', False):
        infer_time = -infer_time
    if batch.is_decode():
        infer_time *= self.decode_scale_factor
    else:
        infer_time *= self.prefill_scale_factor

    if not batch.is_decode():
        infer_time = (
            max(infer_time, self.prefill_min_latency)
            if infer_time > 0
            else infer_time
        )

    return infer_time / 1e3


def flush_sol_debug_results():
    """手动刷写所有累积的 batch 结果到文件"""
    output_file = os.path.join(SOL_DEBUG_OUTPUT_DIR, "sol_debug_all_batches.jsonl")
    with open(output_file, "w") as f:
        for item in _all_batch_results:
            f.write(json.dumps(item) + "\n")
    print(f"[SOL DEBUG] Flushed {len(_all_batch_results)} batch results to {output_file}")

    # 同时输出 summary
    summary_file = os.path.join(SOL_DEBUG_OUTPUT_DIR, "sol_debug_summary.json")
    prefill_batches = [b for b in _all_batch_results if b["phase"] == "prefill"]
    decode_batches = [b for b in _all_batch_results if b["phase"] == "decode"]
    summary = {
        "total_batches": len(_all_batch_results),
        "prefill_batches": len(prefill_batches),
        "decode_batches": len(decode_batches),
        "avg_prefill_latency_ms": np.mean([b["total_latency_ms"] for b in prefill_batches]) if prefill_batches else 0,
        "avg_decode_latency_ms": np.mean([b["total_latency_ms"] for b in decode_batches]) if decode_batches else 0,
        "model_config": _all_batch_results[0]["model_config"] if _all_batch_results else {},
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SOL DEBUG] Summary saved to {summary_file}")


def patch_aiconfigurator_predictor():
    """
    Monkey-patch AIConfiguratorTimePredictor 以启用详细 SOL 日志。
    在 simulate_one_case.py 的 import 之后调用即可。
    """
    from sglang_simulator.time_predictor.aiconfigurator import AIConfiguratorTimePredictor

    AIConfiguratorTimePredictor.predict_infer_latency_dict = patched_predict_infer_latency_dict
    AIConfiguratorTimePredictor.predict_infer_time = patched_predict_infer_time

    # 注册退出时自动 flush
    import atexit
    atexit.register(flush_sol_debug_results)

    print(f"[SOL DEBUG] Patch applied! Output dir: {SOL_DEBUG_OUTPUT_DIR}")
    print(f"[SOL DEBUG] Set SOL_DEBUG_OUTPUT_DIR env to change output location.")


if __name__ == "__main__":
    print("This module provides a monkey-patch for AIC SOL debug logging.")
    print("Usage: from b300_glm5_sol_debug_patch import patch_aiconfigurator_predictor")
    print("       patch_aiconfigurator_predictor()")
