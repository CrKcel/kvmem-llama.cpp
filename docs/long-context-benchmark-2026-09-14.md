# 任务二：256K 多轮工具测试

本页保存 2026-09-14 的历史测量与可复用测试方法。**表中 IQ4 使用旧 Python 量化 MTP 权重；当前默认的 llama-quantize 模型尚未完成任务二复测。** 当前配置及图文任务见[性能文档](recommended-config-performance.md)。

## 固定测试方法

RTX 5060 Ti；context 262144；batch 512；MTP 2；草稿 KV F16；auto replay + user Q。IQ3 为 Q8 主 KV、检索 36864、生成预留 16384、GPU Q8_0 视觉头；IQ4 为 Q5 主 KV、检索 32768、生成预留 12288、CPU BF16 视觉头。长程任务不发送图片，视觉头仍按推荐配置加载。

全程 thinking，思考预算 128，每请求最多生成 512 token。采样为 temperature=1、top_p=.95、top_k=20、min_p=0、presence/frequency penalty=0、repetition penalty=1、seed=42。

固定用户问题要求依次确认源文件的 batch ID 和 checksum，最后根据 `final.py` 生成 Python 工具。测试程序构造 `read_file` 调用及返回结果，每轮增加约 8K token；文件内容是带批次标记、校验字段和重复增量语句的固定数据。模型输出不加入后续回放历史，因此两模型的输入序列可保持一致。此流程是受控工具结果回放，不执行模型提出的额外工具调用。

最后一轮按真实 API `prompt_tokens` 计算剩余容量，保留 512 token 生成空间和小幅模板边界余量。断言最终 prompt 位于 `ctx-1024` 至 `ctx-512` 之间、各轮上下文递增、后续请求有缓存命中；HTTP/SSE 必须完整结束。最终 prompt 加实际生成量用于报告 context 填充程度。

这些断言检查协议和缓存推进，不代表长程语义质量合格。例如模型可能主动猜测未要求的源码行数；生成代码达到 512 token 上限时不检查完整程序功能。全部回答保留，便于后续另做质量评估。

## 历史测量结果

两组均完成 33 个请求，输入文件逐字一致。服务二进制 SHA-256 为 `b0f0a9ad776395f4133bdf2e850bbd238e18eb8877de6c3a5758c62fc6717b5c`。旧 IQ4 模型 SHA-256 为 `4bfcd8385362ff0fdc6b9eeba8a9f5547fbeef7748e0128549fa2524d0d07d0b`，该本地文件已被新版本替换。

| 指标 | IQ3 | IQ4（旧 Python MTP） |
|---|---:|---:|
| 最终 prompt | 261546 token | 261546 token |
| 加上最终实际生成量 | 262058 / 262144（99.97%） | 262058 / 262144（99.97%） |
| 全任务聚合 prefill，首遍 | 436.72 token/s | 465.61 token/s |
| 全任务聚合 prefill，有效 | 243.16 token/s | 254.91 token/s |
| 工具轮次聚合 decode | 29.96 token/s | 32.71 token/s |
| 最后一轮 TTFT | 36.79 秒 | 33.98 秒 |
| 最后一轮 decode，512 token | 29.33 token/s | 35.30 token/s |
| 运行期 RSS 峰值 | **13444.70 MiB（13.13 GiB）** | **11249.84 MiB（10.99 GiB）** |
| 整卡显存峰值 | **15873.10 MiB** | **15931.10 MiB** |
| 最低可用显存 | **178.90 MiB** | **120.90 MiB** |

两项 prefill 都取全部 33 个请求的实际新增输入总量 261545 token，除以对应总耗时。首遍时间 IQ3/IQ4 为 598.878878/561.729397 秒；有效时间为 1075.61780/1026.03272 秒，包含历史重算、检索和检查点管理。工具轮次 decode 按总生成 token / 总 decode 时间聚合，包含思考。

RAM 是运行阶段 VmRSS 采样峰值，不含加载；显存为整卡采样。每种配置测一轮，不把小幅速度差异解释为稳定优势，也未覆盖 256K 历史、多图和生成满预留的组合负载。

IQ3/IQ4 分别有 26/27 轮走历史重算，约占工具轮次 prefill 的 42.0%/42.8%。最后一轮均复用 253618 token 的历史，仅重算新增的 7927 行；并非每轮重新计算全部上下文。

## 用当前模型复测

下面命令会使用当前默认文件生成一轮新测量，不能据此复现已删除的旧 IQ4 文件。脚本的固定 GPU 设置与端口要求见[任务一运行说明](recommended-config-performance.md#复测任务一)。两组顺序运行，并使用新的输出目录。

```bash
python3 scripts/multimodal_canary.py \
  --query-policy user --kv q8_0 --draft-kv f16 \
  --image-max-tokens 512 --budget 36864 --reserve 16384 --ctx 262144 --batch 512 \
  --thinking-budget 128 --long-context-benchmark --long-chunk-tokens 8192 \
  --folder logs/task2-iq3

python3 scripts/multimodal_canary.py \
  --model models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf \
  --mmproj models/unsloth/Qwen3.8-27B-GGUF/mmproj-BF16.gguf --device cpu \
  --query-policy user --kv q5_0 --draft-kv f16 \
  --image-max-tokens 512 --budget 32768 --reserve 12288 --ctx 262144 --batch 512 \
  --thinking-budget 128 --long-context-benchmark --long-chunk-tokens 8192 \
  --folder logs/task2-iq4

python3 scripts/summarize_canary.py \
  logs/task2-iq3 logs/task2-iq4
```

每个目录保存请求/响应、逐轮 trace、`long-context.json`、NVML/RSS 采样，以及 `metrics.csv`、`metrics.json` 汇总。

测试进程 VmSwap 达到 512 MiB，或系统 swap 比起点增加 1024 MiB 时，脚本保存 `SWAP_STOP.json` 和内存状态，再关闭自己启动的服务。可通过 `--swap-stop-mib` 和 `--system-swap-growth-stop-mib` 设置阈值；触发后应暂停实验，不继续下一组配置。

本地历史原始记录位于 `logs/long_context_256k_20260914/{iq3,iq4}/`，不随仓库分发。历史两组服务进程 VmSwap 峰值均为 0，未触发暂停。
