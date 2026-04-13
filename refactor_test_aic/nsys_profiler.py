#!/usr/bin/env python3
"""
Nsys Profile 工具 — 对指定 batch 进行 Nvidia Nsight Systems 性能分析
====================================================================

功能:
  - 从 schedule_batch.jsonl 中按 case_id 或 (batch_size, seq_len, forward_mode) 查找 batch
  - 构建 prefix + extend 的 prompt，利用 SGLang Engine 的 prefix cache 还原真实 KV 缓存
  - 集成 NVTX range 标记，便于 nsys 时间线对齐
  - 支持 CUDA Profiler API（cudaProfilerStart/Stop），配合 nsys 的 --capture-range=cudaProfilerApi

使用方法:
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ 基础 profiling（仅 GPU kernel + NVTX）:                                     │
  │   nsys profile -o report --capture-range=cudaProfilerApi \\                 │
  │     python3 -m refactor_test_aic.nsys_profiler --case-id 42                │
  │                                                                             │
  │ 带 Python 调用栈（推荐，可看到每个 kernel 对应的 Python 代码位置）:            │
  │   nsys profile -o report \\                                                 │
  │     --capture-range=cudaProfilerApi \\                                      │
  │     --python-backtrace=cuda \\                                              │
  │     --python-sampling=true \\                                               │
  │     --cpuctxsw=process-tree \\                                              │
  │     python3 -m refactor_test_aic.nsys_profiler --case-id 42                │
  │                                                                             │
  │ dry-run 验证（不需要 GPU）:                                                  │
  │   python3 -m refactor_test_aic.nsys_profiler --case-id 42 --dry-run        │
  └──────────────────────────────────────────────────────────────────────────────┘

nsys 参数说明:
  --python-backtrace=cuda   在每次 CUDA API 调用时采集 Python 调用栈
  --python-sampling=true    启用 Python 采样分析器（按时间间隔采样 Python 栈帧）
  --cpuctxsw=process-tree   采集 CPU 上下文切换，关联 Python 线程与 GPU 活动

依赖:
  - sglang (Engine, ServerArgs)
  - torch (NVTX, CUDA Profiler)
"""

import argparse
import contextlib
import json
import os
import sys
from dataclasses import asdict
from typing import Any

import shlex

from .config import DATA_DIR, SCHEDULE_JSONL_FILENAME, SGLANG_LAUNCH_CMD


# ============================================================================
# JSONL 读取函数
# ============================================================================

def load_case_by_id(schedule_jsonl: str, case_id: int) -> tuple[str, list[dict[str, Any]]]:
    """
    按 0-based 行号从 schedule_batch.jsonl 加载一条 batch。

    Returns:
        (batch_type, request_infos)  其中 request_infos 已标准化为
        [{"input_length": .., "past_kv_length": ..}, ...]
    """
    with open(schedule_jsonl, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx != case_id:
                continue
            rec = json.loads(line)
            forward_mode = int(rec["forward_mode"])
            batch_type = "prefill" if forward_mode == 1 else "decode"
            out: list[dict[str, Any]] = []
            for req in rec["request_infos"]:
                if forward_mode == 1:
                    seq_len = int(req["extend_input_len"])
                    past_kv_len = int(req["prefix_indices_len"])
                else:
                    seq_len = 1
                    past_kv_len = int(req["prefix_indices_len"]) + int(req["output_ids_len"])
                out.append({"input_length": seq_len, "past_kv_length": past_kv_len})
            return batch_type, out

    raise FileNotFoundError(f"case_id={case_id} not found in {schedule_jsonl}")


def find_case_by_match(
    schedule_jsonl: str,
    *,
    batch_size: int,
    seq_len: int,
    forward_mode: int = 1,
    match_index: int = 0,
) -> tuple[int, dict[str, Any]]:
    """
    按 (batch_size, seq_len, forward_mode) 在 JSONL 中查找匹配 batch。

    Returns:
        (line_index, record_dict)
    """
    seen = 0
    with open(schedule_jsonl, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if int(rec.get("forward_mode", -1)) != forward_mode:
                continue
            reqs = rec.get("request_infos") or []
            if len(reqs) != batch_size:
                continue

            if forward_mode == 1:
                if all(int(r.get("extend_input_len", -1)) == seq_len for r in reqs):
                    if seen == match_index:
                        return idx, rec
                    seen += 1
            else:
                if all(int(r.get("output_ids_len", -1)) == seq_len for r in reqs):
                    if seen == match_index:
                        return idx, rec
                    seen += 1

    raise FileNotFoundError(
        f"No matching record in {schedule_jsonl} for "
        f"batch_size={batch_size}, seq_len={seq_len}, forward_mode={forward_mode}"
    )


# ============================================================================
# Prompt 构建
# ============================================================================

def build_prompts(
    request_infos: list[dict[str, Any]], token_id: int = 325
) -> tuple[list[list[int]], list[list[int]]]:
    """
    根据 request_infos 构建 prefix_prompts 和 full_prompts。

    思路:
      prefix_prompt = [token_id] * past_kv_length   (先发送以填充 prefix cache)
      full_prompt   = prefix_prompt + [token_id] * input_length

    这样 SGLang 会复用 prefix cache，使 past_kv_length 对应真实 cached KV，
    而非重新计算整个前缀。
    """
    prefix_prompts: list[list[int]] = []
    full_prompts: list[list[int]] = []

    for ri in request_infos:
        seq_len = int(ri["input_length"])
        past_kv_len = int(ri["past_kv_length"])
        prefix = [token_id] * max(past_kv_len, 0)
        extend = [token_id] * max(seq_len, 1)
        prefix_prompts.append(prefix if prefix else [token_id])
        full_prompts.append(prefix + extend)

    return prefix_prompts, full_prompts


# ============================================================================
# NVTX 辅助
# ============================================================================

@contextlib.contextmanager
def nvtx_range(msg: str):
    """Best-effort NVTX range marker for Nsight Systems profiling."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(msg)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
        else:
            yield
    except Exception:
        yield


# ============================================================================
# SGLang 启动命令解析
# ============================================================================

def parse_launch_cmd(cmd: str) -> dict[str, Any]:
    """
    将 config.py 中的 SGLANG_LAUNCH_CMD 解析为可传给 ServerArgs 的 kwargs dict。

    解析规则:
      1. 用 shlex 按 shell 语义切分（支持 \\ 续行、引号）
      2. 跳过第一个 token（python3）和第二个 token（脚本路径）
      3. 将 --foo-bar 转换为 foo_bar，自动识别 bool flag / 数值 / 字符串

    Returns:
        dict, 例如 {"model_path": "/models/xxx", "tp_size": 8, "disable_overlap_schedule": True, ...}
    """
    tokens = shlex.split(cmd.strip())

    # 跳过 "python3 script.py" 前缀
    arg_tokens = []
    for i, tok in enumerate(tokens):
        if tok.startswith("--"):
            arg_tokens = tokens[i:]
            break

    kwargs: dict[str, Any] = {}
    i = 0
    while i < len(arg_tokens):
        tok = arg_tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue

        key = tok.lstrip("-").replace("-", "_")

        # 看下一个 token 是否为值
        if i + 1 < len(arg_tokens) and not arg_tokens[i + 1].startswith("--"):
            raw_val = arg_tokens[i + 1]
            # 尝试转换数值
            try:
                val: Any = int(raw_val)
            except ValueError:
                try:
                    val = float(raw_val)
                except ValueError:
                    val = raw_val
            kwargs[key] = val
            i += 2
        else:
            # bool flag，如 --disable-overlap-schedule, --trust-remote-code
            kwargs[key] = True
            i += 1

    return kwargs


def build_server_args_from_config(
    *,
    load_format: str = "dummy",
    override_model_path: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
):
    """
    从 SGLANG_LAUNCH_CMD 解析出 ServerArgs，可覆写部分参数。

    nsys profiling 时会强制设置:
      - load_format="dummy" (nsys 时不需要加载真实权重)
      - disable_cuda_graph=True (避免 cuda graph 干扰 profiling)

    Returns:
        ServerArgs 实例
    """
    from sglang.srt.server_args import ServerArgs

    parsed = parse_launch_cmd(SGLANG_LAUNCH_CMD)
    print(f"[nsys_profiler] 从 SGLANG_LAUNCH_CMD 解析参数: {parsed}")

    # profiling 专用覆写
    parsed["load_format"] = load_format
    parsed["disable_cuda_graph"] = True
    parsed.setdefault("base_gpu_id", 0)

    if override_model_path:
        parsed["model_path"] = override_model_path

    if extra_kwargs:
        parsed.update(extra_kwargs)

    print(f"[nsys_profiler] ServerArgs 参数: {parsed}")
    
    return ServerArgs(**parsed)


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Nsys Profile 工具: 对 schedule_batch.jsonl 中的指定 batch 进行性能分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
使用示例:
  # dry-run 验证 prompt 构建:
  python3 -m refactor_test_aic.nsys_profiler --case-id 42 --dry-run

  # 基础 profiling:
  nsys profile -o report --capture-range=cudaProfilerApi \\
    python3 -m refactor_test_aic.nsys_profiler --case-id 42

  # 带 Python 调用栈（推荐，可看到 kernel 对应的 Python 代码）:
  nsys profile -o report \\
    --capture-range=cudaProfilerApi \\
    --python-backtrace=cuda \\
    --python-sampling=true \\
    --cpuctxsw=process-tree \\
    python3 -m refactor_test_aic.nsys_profiler --case-id 42

  # 按 batch_size + seq_len 匹配:
  nsys profile -o report --capture-range=cudaProfilerApi \\
    python3 -m refactor_test_aic.nsys_profiler --batch-size 16 --seq-len 128
""",
    )

    # 数据源
    ap.add_argument(
        "--schedule-jsonl", type=str, default=None,
        help="schedule_batch.jsonl 路径 (默认: {data_dir}/{SCHEDULE_JSONL_FILENAME})",
    )
    ap.add_argument("--data-dir", type=str, default=DATA_DIR, help="数据根目录")

    # Case 选择 — 方式 1: case_id
    ap.add_argument("--case-id", type=int, default=None, help="直接指定 JSONL 行号 (0-based)")

    # Case 选择 — 方式 2: 匹配
    ap.add_argument("--batch-size", type=int, default=None, help="要匹配的 batch_size")
    ap.add_argument("--seq-len", type=int, default=None, help="要匹配的 seq_len")
    ap.add_argument("--forward-mode", type=int, default=1, choices=[1, 2], help="1=prefill, 2=decode")
    ap.add_argument("--match-index", type=int, default=0, help="多条匹配时取第 N 条 (0-based)")

    # 模型参数（可通过命令行覆盖 config.py 中的 SGLANG_LAUNCH_CMD 解析值）
    ap.add_argument(
        "--model", type=str, default=None,
        help="覆盖 model_path（默认从 SGLANG_LAUNCH_CMD 解析）",
    )
    ap.add_argument("--load-format", type=str, default="dummy", help="权重加载格式，profiling 时一般用 dummy")
    ap.add_argument("--token-id", type=int, default=325, help="构建 prompt 的 dummy token id")
    ap.add_argument("--max-new-tokens", type=int, default=2)

    # 调试
    ap.add_argument("--dry-run", action="store_true", help="只打印 prompt 信息，不启动 Engine")

    args = ap.parse_args()

    # 确定 JSONL 路径
    schedule_jsonl = args.schedule_jsonl or os.path.join(args.data_dir, SCHEDULE_JSONL_FILENAME)

    # 确定 case
    if args.case_id is not None:
        case_id = args.case_id
        batch_type, request_infos = load_case_by_id(schedule_jsonl, case_id)
        print(f"[nsys_profiler] 使用 case_id={case_id}, batch_type={batch_type}, requests={len(request_infos)}")
    elif args.batch_size is not None and args.seq_len is not None:
        case_id, rec = find_case_by_match(
            schedule_jsonl,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            forward_mode=args.forward_mode,
            match_index=args.match_index,
        )
        iter_latency_s = rec.get("iter_latency", None)
        iter_latency_ms = float(iter_latency_s) * 1000.0 if iter_latency_s is not None else None
        batch_type, request_infos = load_case_by_id(schedule_jsonl, case_id)
        print(
            f"[nsys_profiler] 匹配到 case_id={case_id}, batch_type={batch_type}, "
            f"requests={len(request_infos)}"
            + (f", gt_latency={iter_latency_ms:.3f}ms" if iter_latency_ms else "")
        )
    else:
        print("错误: 需要指定 --case-id 或 (--batch-size + --seq-len)")
        print("运行 --help 查看用法")
        sys.exit(1)

    if not request_infos:
        raise ValueError("Empty request_infos for selected case.")

    # 构建 prompts
    prefix_prompts, full_prompts = build_prompts(request_infos, token_id=args.token_id)

    print(f"\n[nsys_profiler] Prompt 详情:")
    for i, ri in enumerate(request_infos):
        print(
            f"  req[{i}] batch_type={batch_type} "
            f"input_len={ri['input_length']} past_kv_len={ri['past_kv_length']} "
            f"prefix_prompt_len={len(prefix_prompts[i])} full_prompt_len={len(full_prompts[i])}"
        )

    if args.dry_run:
        print("\n[nsys_profiler] --dry-run 模式，不启动 Engine。")
        return

    # ---- 启动 SGLang Engine (参数自动从 SGLANG_LAUNCH_CMD 解析) ----
    import torch
    from sglang.srt.entrypoints.engine import Engine

    server_args = build_server_args_from_config(
        load_format=args.load_format,
        override_model_path=args.model,
    )
    print(f"\n[nsys_profiler] ServerArgs 关键参数:")
    print(f"  model_path = {server_args.model_path}")
    print(f"  tp_size = {server_args.tp_size}")
    print(f"  load_format = {server_args.load_format}")
    print(f"  disable_cuda_graph = {server_args.disable_cuda_graph}")

    llm = Engine(**asdict(server_args))

    sampling_params = {
        "temperature": 0,
        "top_p": 1,
        "max_new_tokens": args.max_new_tokens,
    }

    # Step 1: 填充 prefix cache
    print("[nsys_profiler] 填充 prefix cache ...")
    with nvtx_range("prime_prefix_cache"):
        _ = llm.generate(
            input_ids=prefix_prompts,
            sampling_params={"temperature": 0, "top_p": 1, "max_new_tokens": 1},
        )
    print("[nsys_profiler] prefix cache 已填充")

    # Step 2: 在 CUDA Profiler 范围内运行 full prompts
    print("[nsys_profiler] 开始 profiling generate() ...")
    torch.cuda.cudart().cudaProfilerStart()
    with nvtx_range("generate_full_prompts"):
        outputs = llm.generate(input_ids=full_prompts, sampling_params=sampling_params)
    torch.cuda.cudart().cudaProfilerStop()

    print(f"[nsys_profiler] generate() 完成, outputs_type={type(outputs)}")

    llm.shutdown()
    print("[nsys_profiler] Engine 已关闭")


if __name__ == "__main__":
    main()
