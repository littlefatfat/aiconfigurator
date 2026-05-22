"""B300 GLM5 SOL Debug runner - inject SOL debug patch then run simulate_one_case.
Run in container mry-dpsk-v4:
  cd /host/insight_benchmark/test/hisim/mry_debug/one_case
  python3 /host/aiconfigurator/refactor_test_aic/run_glm5_sol_debug.py
"""

import sys
import os
import multiprocessing


def run():
    # 设置输出目录为当前工作目录下
    os.environ["SOL_DEBUG_OUTPUT_DIR"] = os.path.join(os.getcwd(), "sol_debug_output")

    # 确保容器里的代码路径在 sys.path 中
    sys.path.insert(0, "/host/aiconfigurator/refactor_test_aic")
    sys.path.insert(0, "/host/insight_benchmark/test/hisim/mry_debug/one_case")

    # 必须在 import sglang 前设置
    os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")
    os.environ.setdefault("HISIM_LOG_LEVEL", "INFO")

    # 导入 patch
    from b300_glm5_sol_debug_patch import patch_aiconfigurator_predictor

    # 在 predictor 被构造之前就 monkey-patch
    patch_aiconfigurator_predictor()

    # 手动构造参数
    sys.argv = [
        "simulate_one_case.py",
        "--server-args", "/host/insight_benchmark/test/hisim/mry_debug/one_case/l3.server_args.glm5.json",
        "--hisim-config", "/host/insight_benchmark/test/hisim/mry_debug/one_case/l3.hisim_config.glm5.json",
        "--requests", "/host/bl_data_trace/multi_node_trace_combine_glm-5/hisim-num-node-5-glm-5-blksz-256-bucket-80-144-cnt-8952-time-60min.jsonl",
        "--backend-version", "0.5.10",
        "--device-name", "b300_sxm",
    ]

    # 执行
    from simulate_one_case import main
    main()


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    run()
