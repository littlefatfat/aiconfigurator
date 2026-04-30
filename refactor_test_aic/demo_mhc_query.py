"""
Quick demo: query DeepSeek-V4 mHC module latency via AIC SDK.
===============================================================
最小验证脚本，直接调 perf_database.query_mhc_module，分别在 SOL /
EMPIRICAL / SILICON 三种模式下打印 pre/post 的时延曲线。

SILICON 模式需要把采集产物 ``mhc_module_perf.txt`` 放到：
    src/aiconfigurator/systems/data/<system>/<backend>/<version>/

用法：
    python3 -m refactor_test_aic.demo_mhc_query
或：
    python3 refactor_test_aic/demo_mhc_query.py
"""

from __future__ import annotations

from aiconfigurator.sdk.common import DatabaseMode, GEMMQuantMode
from aiconfigurator.sdk.perf_database import get_database, get_systems_paths


# ---- 数据库选择（按实际采集的 system/backend/version 调整） --------------------
AIC_SYSTEM = "h20_sxm"
AIC_BACKEND = "sglang"
AIC_VERSION = "0.5.9"

# ---- DeepSeek-V4-Flash-FP8 的 mHC 相关参数（见 model_configs/*.json） --------
HIDDEN_SIZE = 4096
HC_MULT = 4
SINKHORN_ITERS = 20
QUANT_MODE = GEMMQuantMode.bfloat16

# 采样几个代表性 num_tokens（覆盖 decode/prefill 常见区间）
TOKEN_CASES = [1, 8, 64, 256, 1024, 4096, 16384]


def _query_one(db, *, op: str, num_tokens: int) -> float | None:
    try:
        r = db.query_mhc_module(
            num_tokens=num_tokens,
            hidden_size=HIDDEN_SIZE,
            hc_mult=HC_MULT,
            sinkhorn_iters=SINKHORN_ITERS,
            op=op,
            quant_mode=QUANT_MODE,
        )
        return float(r)
    except Exception as e:
        print(f"  [{op:>4}] num_tokens={num_tokens:>6}  ERROR: {e}")
        return None


def main() -> None:
    db = get_database(
        system=AIC_SYSTEM,
        backend=AIC_BACKEND,
        version=AIC_VERSION,
        systems_paths=get_systems_paths(),
    )
    print(f"database: system={AIC_SYSTEM}, backend={AIC_BACKEND}, version={AIC_VERSION}")
    print(
        f"params  : hidden_size={HIDDEN_SIZE}, hc_mult={HC_MULT}, "
        f"sinkhorn_iters={SINKHORN_ITERS}, quant_mode={QUANT_MODE.name}"
    )

    for mode in (DatabaseMode.SOL, DatabaseMode.EMPIRICAL, DatabaseMode.SILICON):
        db.set_default_database_mode(mode)
        print(f"\n=== database_mode = {mode.name} ===")
        print(f"{'op':>5} {'num_tokens':>12} {'latency(ms)':>14}")
        for op in ("pre", "post"):
            for n in TOKEN_CASES:
                lat = _query_one(db, op=op, num_tokens=n)
                if lat is not None:
                    print(f"{op:>5} {n:>12} {lat:>14.4f}")


if __name__ == "__main__":
    main()
