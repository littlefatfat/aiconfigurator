#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
from dataclasses import asdict
from typing import Any


DEFAULT_SCHEDULE_JSONL = (
    "/raid/kimi/aic_trtllm/ali_qwen/for_nv/h20e_qwen3_32b_fp8/schedule_batch.jsonl"
)


def _load_case_from_schedule_jsonl(schedule_jsonl: str, case_id: int) -> tuple[str, list[dict[str, Any]]]:
    """
    Read `schedule_batch.jsonl` and treat the 0-based line index as `case_id`.

    This file is written by `ali_qwen/for_nv/hook.py` and contains one JSON record per scheduler batch:
      {
        "forward_mode": 1|2,
        "request_infos": [{"extend_input_len":.., "prefix_indices_len":.., "output_ids_len":..}, ...]
      }

    We convert it into normalized request_infos:
      [{"input_length": seq_len, "past_kv_length": past_kv_len}, ...]

    Semantics (matching `convert_batch_log.py`):
    - prefill (forward_mode==1): input_length = extend_input_len, past_kv_length = prefix_indices_len
    - decode  (forward_mode==2): input_length = 1, past_kv_length = prefix_indices_len + output_ids_len
    """
    with open(schedule_jsonl, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx != int(case_id):
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


def _find_case_in_schedule_jsonl(
    schedule_jsonl: str,
    *,
    batch_size: int,
    seq_len: int,
    forward_mode: int = 1,
    match_index: int = 0,
) -> tuple[int, dict[str, Any]]:
    """
    Find the Nth matching record from `schedule_batch.jsonl`.

    Matching rule:
    - `len(request_infos) == batch_size`
    - `forward_mode` equals the requested mode (1=prefill, 2=decode)
    - For prefill: all req.extend_input_len == seq_len
    - For decode: all req.output_ids_len == seq_len (typically 1)

    Returns:
      (line_index, record_dict)
    """
    want_bs = int(batch_size)
    want_seq = int(seq_len)
    want_mode = int(forward_mode)
    want_match = int(match_index)
    if want_bs <= 0:
        raise ValueError("batch_size must be > 0")
    if want_seq <= 0:
        raise ValueError("seq_len must be > 0")
    if want_mode not in (1, 2):
        raise ValueError("forward_mode must be 1 (prefill) or 2 (decode)")
    if want_match < 0:
        raise ValueError("match_index must be >= 0")

    seen = 0
    with open(schedule_jsonl, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if int(rec.get("forward_mode", -1)) != want_mode:
                continue
            reqs = rec.get("request_infos") or []
            if len(reqs) != want_bs:
                continue

            if want_mode == 1:
                if all(int(r.get("extend_input_len", -1)) == want_seq for r in reqs):
                    if seen == want_match:
                        return idx, rec
                    seen += 1
            else:
                if all(int(r.get("output_ids_len", -1)) == want_seq for r in reqs):
                    if seen == want_match:
                        return idx, rec
                    seen += 1

    raise FileNotFoundError(
        f"No matching record found in {schedule_jsonl} for "
        f"batch_size={want_bs}, seq_len={want_seq}, forward_mode={want_mode}, match_index={want_match}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Find a scheduler batch by (batch_size, seq_len, forward_mode) from schedule_batch.jsonl, "
            "construct per-request seq_len/past_kv_len, then run SGLang with prefix-cache reuse so "
            "past_kv_length corresponds to real cached KV."
        )
    )
    ap.add_argument("--schedule-jsonl", type=str, default=DEFAULT_SCHEDULE_JSONL)
    ap.add_argument("--batch-size", type=int, required=True, help="Batch size to match (len(request_infos)).")
    ap.add_argument(
        "--seq-len",
        type=int,
        required=True,
        help="Seq len to match (prefill: extend_input_len, decode: output_ids_len).",
    )
    ap.add_argument("--forward-mode", type=int, default=1, choices=[1, 2], help="1=prefill, 2=decode")
    ap.add_argument(
        "--match-index",
        type=int,
        default=0,
        help="If multiple records match (batch_size, seq_len, forward_mode), pick the Nth match (0-based).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL", os.environ.get("MODEL_PATH", "Qwen2.5-3B-Instruct")),
        help=(
            "Model name / model id / local path. Examples: 'Qwen2.5-3B-Instruct', "
            "'Qwen/Qwen2.5-3B-Instruct', '/host/models/modelscope/Qwen2.5-3B-Instruct/'."
        ),
    )
    ap.add_argument(
        "--model-root",
        type=str,
        default=os.environ.get("MODEL_ROOT", "/host/models/modelscope"),
        help="If --model is a bare name and exists under this directory, use that local path.",
    )
    ap.add_argument("--load-format", type=str, default=os.environ.get("SGLANG_LOAD_FORMAT", "dummy"))
    ap.add_argument("--token-id", type=int, default=325, help="Dummy token id to build input_ids.")
    ap.add_argument("--max-new-tokens", type=int, default=2)
    args = ap.parse_args()

    # case_id, _rec = _find_case_in_schedule_jsonl(
    #     args.schedule_jsonl,
    #     batch_size=int(args.batch_size),
    #     seq_len=int(args.seq_len),
    #     forward_mode=int(args.forward_mode),
    #     match_index=int(args.match_index),
    # )
    # iter_latency_s = _rec.get("iter_latency", None)
    # iter_latency_ms = (float(iter_latency_s) * 1000.0) if iter_latency_s is not None else None
    # batch_type, request_infos = _load_case_from_schedule_jsonl(args.schedule_jsonl, case_id)
    # if not request_infos:
    #     raise ValueError(f"Empty request_infos for selected case.")

    # # Build prompts as (prefix + extend). We will prime the prefix cache by issuing prefix-only prompts
    # # first, and then issuing the full prompts which share the same prefix. This makes `past_kv_length`
    # # correspond to cached KV (radix/chunk cache), rather than recomputing the full prefix each time.
    # prefix_prompts: list[list[int]] = []
    # full_prompts: list[list[int]] = []
    # for i, ri in enumerate(request_infos):
    #     seq_len = int(ri["input_length"])
    #     past_kv_len = int(ri["past_kv_length"])
    #     prefix = [int(args.token_id)] * max(past_kv_len, 0)
    #     extend = [int(args.token_id)] * max(seq_len, 1)
    #     prefix_prompts.append(prefix if prefix else [int(args.token_id)])
    #     full_prompts.append(prefix + extend)
    #     print(
    #         f"[case] req[{i}] batch_type={batch_type} "
    #         f"seq_len={seq_len} past_kv_len={past_kv_len} "
    #         f"prefix_prompt_len={len(prefix)} full_prompt_len={len(prefix + extend)} "
    #         + (f"iter_latency_ms={iter_latency_ms:.3f}" if iter_latency_ms is not None else "iter_latency_ms=NA")
    #     )
    
    # insight
    prefix_prompts = [
        [10000] * 100
    ]

    full_prompts = [
        [10000] * 100 + [231] * 10
    ]

    # Run SGLang
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.server_args import ServerArgs

    import torch

    @contextlib.contextmanager
    def nvtx_range(msg: str):
        """Best-effort NVTX range marker for profiling (Nsight Systems/Compute)."""
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push(msg)
                try:
                    yield
                finally:
                    torch.cuda.nvtx.range_pop()
            else:
                yield
        except Exception:
            # NVTX not available / disabled; do not break the run.
            yield

    model_value = str(args.model)
    model_path = model_value
    # If user provides a bare name (no slashes) and it exists under model_root, prefer local dir.
    if not os.path.exists(model_value) and ("/" not in model_value) and ("\\" not in model_value):
        candidate = os.path.join(str(args.model_root), model_value)
        if os.path.exists(candidate):
            model_path = candidate
    print(f"Using model_path={model_path!r} (from --model={model_value!r})")

    server_args = ServerArgs(
        model_path=model_path,
        load_format=args.load_format,
        disable_cuda_graph=True,
        base_gpu_id=0,
        # enable_piecewise_cuda_graph=True,
        # piecewise_cuda_graph_max_tokens=4096,
        # piecewise_cuda_graph_tokens=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133]
    )
    llm = Engine(**asdict(server_args))

    sampling_params = {
        "temperature": 0,
        "top_p": 1,
        "max_new_tokens": int(args.max_new_tokens),
    }

    # Prime: run prefix-only prompts to populate radix/chunk cache.
    # This is required to make `past_kv_length` correspond to real cached KV.
    with nvtx_range("sglang_prime_prefix_cache"):
        _ = llm.generate(
            input_ids=prefix_prompts,
            sampling_params={"temperature": 0, "top_p": 1, "max_new_tokens": 1},
        )
    print("prefix-cache primed.")

    torch.cuda.cudart().cudaProfilerStart()
    with nvtx_range("sglang_generate_full_prompts"):
        outputs = llm.generate(input_ids=full_prompts, sampling_params=sampling_params)
    torch.cuda.cudart().cudaProfilerStop()

    print(f"generate() done. outputs_type={type(outputs)}")

    llm.shutdown()


if __name__ == "__main__":
    main()

# nsys profile  -c cudaProfilerApi  --trace-fork-before-exec=true   --cuda-graph-trace=node   -t cuda,nvtx,osrt   -o sglang_bs1is15os8_piece_cg2 -f true python3 main.py --model Qwen/Qwen3-8B --batch-size 1 --seq-len 4