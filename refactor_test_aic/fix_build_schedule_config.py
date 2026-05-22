"""
修复 simulate_one_case.py 中 _build_schedule_config 的量化类型推导逻辑。

问题：原始逻辑把任何非空 quantization 都映射为 "FP8"，无法区分 fp4/nvfp4。
修复：根据 quantization 字段的实际值，正确推导 data_type。

将此函数替换原始 simulate_one_case.py 中的 _build_schedule_config 函数。
路径: /host/insight_benchmark/test/hisim/mry_debug/one_case/simulate_one_case.py
"""


from typing import Optional


def _build_schedule_config(server_args: dict, backend_version: Optional[str] = None) -> dict:
    """根据 server_args 推导 scheduler 段配置。

    量化模式映射规则:
      - quantization 含 "fp4" / "nvfp4" / "w4a8" / "mxfp4" -> data_type = "FP4"
      - quantization 含 "fp8" / "int8" / "sq" 或 model_path 含 "FP8" -> data_type = "FP8"
      - quantization 为其他非空值（未知新模式）-> 默认 "FP8" + 打印 warning
      - quantization 为 None 且 model_path 不含 "FP8" -> data_type = "FP16"(bfloat16)

    KV cache 推导规则:
      - kv_cache_dtype 含 "fp8" / "e4m3" / "e5m2" -> "FP8"
      - kv_cache_dtype 为 "auto" / "bfloat16" / "float16" 或缺失 -> "FP16"
      - kv_cache_dtype 含 "int8" -> "FP8" (通常 int8 kv-cache 与 fp8 开销近似)
    """
    quant = (server_args.get("quantization") or "").lower()
    model_path = server_args.get("model_path", "").upper()

    # ---- 推导 data_type ----
    if quant:
        # fp4 系列
        if any(kw in quant for kw in ("fp4", "nvfp4", "w4a8", "mxfp4")):
            dtype = "FP4"
        # fp8 系列
        elif any(kw in quant for kw in ("fp8", "int8", "sq", "smooth")):
            dtype = "FP8"
        # int4 weight-only (activation 仍是 bf16)
        elif "int4" in quant or "w4a16" in quant or "gptq" in quant or "awq" in quant:
            dtype = "FP16"  # weight-only 量化，compute 用 bf16 TC
        else:
            print(f"[WARNING] Unknown quantization '{quant}', defaulting to FP8")
            dtype = "FP8"
    elif "FP8" in model_path:
        dtype = "FP8"
    else:
        dtype = "FP16"

    # ---- 推导 kv_cache_data_type ----
    kv_cache_dtype = (server_args.get("kv_cache_dtype") or "auto").lower()
    if any(kw in kv_cache_dtype for kw in ("fp8", "e4m3", "e5m2")):
        kv_dtype = "FP8"
    elif "int8" in kv_cache_dtype:
        kv_dtype = "FP8"  # int8 kv 与 fp8 开销近似
    else:
        kv_dtype = "FP16"

    return {
        "tp_size": server_args["tp_size"],
        "ep_size": server_args["ep_size"],
        "dp_size": server_args["dp_size"],
        "data_type": dtype,
        "kv_cache_data_type": kv_dtype,
        "backend_name": "sglang",
        "backend_version": backend_version or server_args["version"],
    }


# ============================================================
# 验证测试
# ============================================================
if __name__ == "__main__":
    # GLM5 NVFP4 + fp8_e4m3 kv-cache 场景
    sa_glm5 = {
        "quantization": "modelopt_fp4",
        "model_path": "/nfs/ZhipuAI/GLM-5.1-FP8/",
        "kv_cache_dtype": "fp8_e4m3",
        "version": "0.5.10",
        "tp_size": 8,
        "ep_size": 1,
        "dp_size": 1,
    }
    result = _build_schedule_config(sa_glm5, "0.5.10")
    print(f"GLM5 NVFP4: {result}")
    assert result["data_type"] == "FP4", f"Expected FP4, got {result['data_type']}"
    assert result["kv_cache_data_type"] == "FP8", f"Expected FP8, got {result['kv_cache_data_type']}"

    # DeepSeek-V4 FP8 场景
    sa_dsv4 = {
        "quantization": "fp8",
        "model_path": "/models/DeepSeek-V4-FP8/",
        "kv_cache_dtype": "fp8_e4m3",
        "version": "0.5.9",
        "tp_size": 8,
        "ep_size": 8,
        "dp_size": 1,
    }
    result = _build_schedule_config(sa_dsv4, None)
    print(f"DSV4 FP8:   {result}")
    assert result["data_type"] == "FP8"
    assert result["kv_cache_data_type"] == "FP8"

    # BF16 无量化场景
    sa_bf16 = {
        "quantization": None,
        "model_path": "/models/Llama-3-70B/",
        "kv_cache_dtype": "auto",
        "version": "0.5.10",
        "tp_size": 4,
        "ep_size": 1,
        "dp_size": 1,
    }
    result = _build_schedule_config(sa_bf16, None)
    print(f"BF16:       {result}")
    assert result["data_type"] == "FP16"
    assert result["kv_cache_data_type"] == "FP16"

    # GPTQ int4 weight-only 场景
    sa_gptq = {
        "quantization": "gptq",
        "model_path": "/models/Llama-3-70B-GPTQ/",
        "kv_cache_dtype": "bfloat16",
        "version": "0.5.10",
        "tp_size": 2,
        "ep_size": 1,
        "dp_size": 1,
    }
    result = _build_schedule_config(sa_gptq, None)
    print(f"GPTQ:       {result}")
    assert result["data_type"] == "FP16"  # weight-only, compute still bf16
    assert result["kv_cache_data_type"] == "FP16"

    print("\nAll tests passed!")
