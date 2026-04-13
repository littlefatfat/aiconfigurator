# SGLang Qwen3-235B Profiling 执行全流程记录

**执行日期**: 2026-03-19
**容器**: mry-hisim
**模型**: Qwen3-235B-A22B-Instruct-2507-FP8
**GPU**: 8 × NVIDIA H20 (97871 MiB each)
**SGLang 版本**: 0.5.9

---

## 1. 环境确认

### 1.1 确认容器状态

```bash
docker ps -a --filter "name=mry-hisim" --format "{{.ID}}\t{{.Names}}\t{{.Status}}"
```

输出:
```
0bf32af04910    mry-hisim       Up 2 hours
```

### 1.2 确认模型路径

```bash
docker exec mry-hisim ls /models/Qwen3-235B-A22B-Instruct-2507-FP8/ | head -20
```

确认模型文件存在 (24 个 safetensors 分片 + config.json 等)。

### 1.3 确认 SGLang 版本和 GPU

```bash
docker exec mry-hisim bash -c "python3 -c 'import sglang; print(sglang.__version__)'"
# 输出: 0.5.9

docker exec mry-hisim nvidia-smi --query-gpu=index,name,memory.total --format=csv
# 输出: 8 × NVIDIA H20, 97871 MiB
```

### 1.4 确认容器挂载

```bash
docker inspect mry-hisim --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}'
```

输出:
```
/data2/maruiyan.mry -> /host
/data2/models -> /models
```

> **注意**: 宿主机 `/data2/maruiyan.mry` 映射到容器 `/host`，
> 所以宿主机 `/data2/maruiyan.mry/sglang/mry_debug/` 在容器内是 `/host/sglang/mry_debug/`

---

## 2. 启动 SGLang 服务

### 2.1 后台启动服务

```bash
docker exec -d mry-hisim bash -c "\
  export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_profile && \
  python3 -m sglang.launch_server \
    --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --disable-radix-cache \
    --mem-fraction-static 0.7 \
    --tp-size 8 \
    --ep-size 8 \
    --enable-dp-attention --dp-size 8 \
    --chunked-prefill-size -1 \
    --cuda-graph-max-bs 32 \
    --trust-remote-code \
    --enable-dp-lm-head \
    --moe-a2a-backend deepep \
    --max-running-requests 300 \
    > /tmp/sglang_server.log 2>&1"
```

> **关键**: 必须设置 `SGLANG_TORCH_PROFILER_DIR` 环境变量，否则 trace 文件可能不会正确生成。

### 2.2 查看启动日志

```bash
docker exec mry-hisim bash -c "cat /tmp/sglang_server.log | tail -30"
```

### 2.3 等待服务就绪

模型加载 + CUDA graph capture 大约需要 1-2 分钟。检查方法:

```bash
# 检查进程
docker exec mry-hisim bash -c "ps aux | grep sglang | head -10"

# 检查健康
docker exec mry-hisim bash -c "curl -s -w '%{http_code}' http://localhost:30000/health"
# 返回 200 表示就绪
```

服务完全就绪的日志标志:
```
INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit)
```

启动后的进程列表:
```
sglang::data_parallel_controller
sglang::detokenizer
sglang::scheduler_DP0_TP0_EP0
sglang::scheduler_DP1_TP1_EP1
... (共 8 个 scheduler, 对应 8 个 DP rank)
```

---

## 3. 触发 Profiling

### 3.1 启动 Torch Profiler (后台)

通过 `/start_profile` HTTP API 启动 profiling，`profile_by_stage=true` 会分别抓取 prefill 和 decode 阶段:

```bash
docker exec mry-hisim bash -c "\
  mkdir -p /tmp/sglang_profile && \
  curl -s -X POST http://localhost:30000/start_profile \
    -H 'Content-Type: application/json' \
    -d '{
      \"output_dir\": \"/tmp/sglang_profile\",
      \"num_steps\": 5,
      \"activities\": [\"CPU\", \"GPU\"],
      \"profile_by_stage\": true,
      \"with_stack\": true,
      \"record_shapes\": true
    }' &"
```

参数说明:
| 参数 | 值 | 说明 |
|------|------|------|
| `output_dir` | `/tmp/sglang_profile` | trace 文件输出目录 |
| `num_steps` | 5 | 每个阶段 (prefill/decode) 抓取 5 个 forward step |
| `activities` | `["CPU", "GPU"]` | 同时抓取 CPU 和 GPU 活动 |
| `profile_by_stage` | `true` | 分别抓取 prefill 和 decode 的 trace |
| `with_stack` | `true` | 记录 Python 调用栈信息 |
| `record_shapes` | `true` | 记录算子输入 shape 信息 |

> **注意**: `/start_profile` 是阻塞式调用，会等到指定的 num_steps 执行完毕后才返回。
> 所以需要在后台运行，同时发送请求来产生 forward step。

### 3.2 发送虚假请求

Profiling 启动后需要实际请求来触发 forward step。发送 3 个简单的 completion 请求:

```bash
docker exec mry-hisim bash -c "\
  echo '=== Request 1 ===' && \
  curl -s -X POST http://localhost:30000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
      \"model\": \"default\",
      \"prompt\": \"Hello world, this is a simple test for profiling the inference pipeline of a large language model. Please write a short paragraph about artificial intelligence.\",
      \"max_tokens\": 64,
      \"temperature\": 0.0
    }' && echo '' && \
  echo '=== Request 2 ===' && \
  curl -s -X POST http://localhost:30000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
      \"model\": \"default\",
      \"prompt\": \"What is the meaning of life? Please provide a thoughtful philosophical answer with multiple perspectives.\",
      \"max_tokens\": 64,
      \"temperature\": 0.0
    }' && echo '' && \
  echo '=== Request 3 ===' && \
  curl -s -X POST http://localhost:30000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
      \"model\": \"default\",
      \"prompt\": \"Explain the concept of machine learning in simple terms that a beginner can understand.\",
      \"max_tokens\": 64,
      \"temperature\": 0.0
    }'"
```

请求结果示例:
```json
{
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 64,
    "total_tokens": 92
  }
}
```

### 3.3 确认 Profiling 完成

查看服务器日志中的 profiling 信息:

```bash
docker exec mry-hisim bash -c "cat /tmp/sglang_server.log | grep -i 'profil'"
```

正常输出应包含:
```
Profiling starts for EXTEND. Traces will be saved to: /tmp/sglang_profile
Stop profiling-EXTEND...
Profiling done. Traces are saved to: /tmp/sglang_profile
Profiling starts for DECODE. Traces will be saved to: /tmp/sglang_profile
Stop profiling-DECODE...
Profiling done. Traces are saved to: /tmp/sglang_profile
```

> 每个 DP rank 都会独立输出上述日志。

---

## 4. 查看 Trace 文件

### 4.1 列出生成的 trace 文件

```bash
docker exec mry-hisim bash -c "\
  find /tmp/sglang_profile -type f | sort && \
  echo '---' && \
  ls -lhS /tmp/sglang_profile/"
```

本次生成的文件:
```
1773920166.5851634-TP-2-DP-2-EP-2-DECODE.trace.json.gz    (472K)
1773920166.5851634-TP-2-DP-2-EP-2-EXTEND.trace.json.gz    (4.5M)
1773920166.5851634-TP-3-DP-3-EP-3-DECODE.trace.json.gz    (472K)
1773920166.5851634-TP-3-DP-3-EP-3-EXTEND.trace.json.gz    (4.5M)
1773920166.5851634-TP-4-DP-4-EP-4-DECODE.trace.json.gz    (4.4M)
1773920166.5851634-TP-4-DP-4-EP-4-EXTEND.trace.json.gz    (472K)
```

文件名格式: `{profile_id}-TP-{tp_rank}-DP-{dp_rank}-EP-{ep_rank}-{STAGE}.trace.json.gz`
- EXTEND = prefill 阶段
- DECODE = decode 阶段

> **注意**: 只有正好处理到请求的 DP rank 才会生成 trace 文件，并非所有 8 个 rank 都有。

### 4.2 复制到宿主机

```bash
docker exec mry-hisim bash -c "\
  cp -r /tmp/sglang_profile /host/sglang/mry_debug/sglang_profile"
```

宿主机路径: `/data2/maruiyan.mry/sglang/mry_debug/sglang_profile/`

---

## 5. 分析 Trace 文件

### 5.1 运行分析脚本

```bash
cd /data2/maruiyan.mry/sglang
python3 mry_debug/analyze_profile.py \
  --trace-dir mry_debug/sglang_profile \
  --output-dir mry_debug/profile_results \
  --top-n 50
```

### 5.2 生成的分析结果

| 文件 | 说明 |
|------|------|
| `profile_results/profile_analysis_report.md` | 完整的 Markdown 格式分析报告 |
| `profile_results/ops_prefill.csv` | Prefill 阶段所有算子明细 (CSV) |
| `profile_results/ops_decode.csv` | Decode 阶段所有算子明细 (CSV) |

### 5.3 分析结果摘要

#### Prefill 阶段

| 指标 | 值 |
|------|------|
| GPU 总耗时 | 1469.9 ms |
| CPU 总耗时 | 191.2 ms |
| 通信总耗时 | 261.2 ms (占 GPU 17.8%) |
| GPU kernel 事件数 | 43,868 |
| CPU 算子事件数 | 22,555 |

GPU 算子分类汇总:
| 分类 | 总耗时 (ms) | 占比 |
|------|------------|------|
| MoE | 547.7 | 37.26% |
| GEMM/Linear | 421.1 | 28.65% |
| Communication | 261.2 | 17.77% |
| Other | 195.1 | 13.27% |
| Elementwise/Activation | 18.0 | 1.22% |
| Embedding/Rotary | 12.5 | 0.85% |
| Attention | 9.7 | 0.66% |

Top 5 耗时算子:
1. MoE forward_impl — 367.2ms (25.0%)
2. DeepEP cached_notify_combine — 199.0ms (13.5%)
3. MoE run_moe_core — 127.3ms (8.7%)
4. DeepGEMM run — 66.8ms (4.6%)
5. DeepGEMM _run_contiguous_gemm — 66.0ms (4.5%)

#### Decode 阶段

| 指标 | 值 |
|------|------|
| GPU 总耗时 | 159.1 ms |
| CPU 总耗时 | 43.8 ms |
| 通信总耗时 | 17.1 ms (占 GPU 10.8%) |

GPU 算子分类汇总:
| 分类 | 总耗时 (ms) | 占比 |
|------|------------|------|
| Other (含 cudaGraphLaunch) | 83.3 | 52.34% |
| GEMM/Linear | 47.2 | 29.68% |
| Communication | 17.1 | 10.76% |
| Elementwise/Activation | 7.0 | 4.42% |

> Decode 阶段使用了 CUDA Graph，因此大部分 kernel 被打包在 `cudaGraphLaunch` 中。

---

## 6. 可视化 Trace

Trace 文件 (.trace.json.gz) 可以用以下工具打开:

1. **Perfetto UI**: https://ui.perfetto.dev/ (推荐，任何浏览器)
2. **Chrome Tracing**: 在 Chrome 浏览器地址栏输入 `chrome://tracing`

直接将 `.trace.json.gz` 文件拖入页面即可查看时间线。

---

## 7. 脚本说明

### 7.1 profile_and_request.py — 一键 Profiling

```bash
# 基本用法
python3 mry_debug/profile_and_request.py \
  --url http://localhost:30000 \
  --output-dir /tmp/sglang_profile

# 自定义参数
python3 mry_debug/profile_and_request.py \
  --url http://localhost:30000 \
  --output-dir /tmp/sglang_profile \
  --num-steps 10 \
  --input-len 256 \
  --output-len 128 \
  --num-requests 5
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--url` | `http://localhost:30000` | SGLang 服务地址 |
| `--output-dir` | `/tmp/sglang_profile` | Trace 输出目录 |
| `--num-steps` | 5 | 每阶段抓取的 forward step 数 |
| `--input-len` | 128 | 虚假请求的输入 token 数 |
| `--output-len` | 64 | 虚假请求的最大输出 token 数 |
| `--num-requests` | 3 | 发送的请求数量 |

### 7.2 analyze_profile.py — 分析 Trace

```bash
# 分析整个目录
python3 mry_debug/analyze_profile.py \
  --trace-dir /tmp/sglang_profile \
  --output-dir mry_debug/profile_results \
  --top-n 50

# 分析指定文件
python3 mry_debug/analyze_profile.py \
  --trace-file /tmp/sglang_profile/xxx-EXTEND.trace.json.gz \
  --output-dir mry_debug/profile_results
```

| 参数 | 说明 |
|------|------|
| `--trace-dir` | 包含 trace 文件的目录 (递归搜索) |
| `--trace-file` | 直接指定 trace 文件 (可多个) |
| `--output-dir` | 分析结果输出目录 |
| `--top-n` | 每阶段显示 Top N 算子 (默认 50) |
| `--no-report` | 不生成 Markdown 报告 |

---

## 8. 停止服务

如果需要停止 SGLang 服务:

```bash
docker exec mry-hisim bash -c "pkill -f 'sglang.launch_server'"
```

---

## 附录: 文件清单

```
mry_debug/
├── profile_and_request.py          # 一键 profiling 脚本
├── analyze_profile.py              # trace 分析脚本
├── sglang_profile/                 # 原始 trace 文件 (从容器复制)
│   ├── *-EXTEND.trace.json.gz      # prefill 阶段 trace
│   └── *-DECODE.trace.json.gz      # decode 阶段 trace
└── profile_results/                # 分析结果
    ├── profile_analysis_report.md  # Markdown 分析报告
    ├── ops_prefill.csv             # prefill 算子明细
    └── ops_decode.csv              # decode 算子明细
```
