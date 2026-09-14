# 推荐配置与测试结果

测试机器：RTX 5060 Ti 16 GiB、Intel Core Ultra 7 255H、物理 RAM 32 GiB；WSL2 Ubuntu 22.04.5，Linux 可见 16 个逻辑 CPU、约 19.53 GiB RAM。测试日期为 2026-09-14。

## 推荐配置

| 配置 | IQ3 | IQ4 |
|---|---|---|
| 启动脚本 | [start-iq3.sh](../scripts/start-iq3.sh) | [start-iq4.sh](../scripts/start-iq4.sh) |
| 主权重 | IQ3_S，GSQ-RCO | IQ4_XS，MTP 权重 Q4_0 |
| 主模型设备 | RTX 5060 Ti | RTX 5060 Ti |
| 视觉头 / 设备 | Q8_0 / GPU | BF16 / CPU |
| 主 KV / MTP KV | Q8_0 / F16 | Q5_0 / F16 |
| MTP 草稿长度 | 2 | 2 |
| 检索预算 | 36864（36 × 1024） | 32768（32 × 1024） |
| 生成预留 / 默认最大输出 | 16384（16 × 1024） | 12288（12 × 1024） |
| context / batch | 262144 / 512 | 262144 / 512 |
| 图片 token 上限 | 512 | 512 |
| query replay / policy | auto / user | auto / user |
| 正式脚本默认思考预算 | 4096 | 4096 |

IQ4 当前使用 [llama-quantize 配方](../scripts/quantization/quantize-iq4-mtp.py)生成的 MTP Q4_0 权重；需要原始 IQ4_XS 模型、imatrix 和配套张量类型表，下载与量化命令见 [README](../README.md)。IQ3 使用发布的 ISTA MTP 模型。

## 任务一：当前配置的图文测试

约 12K 文本背景 → 一张 896×896 三色图形 PNG → 根据图片生成 HTML/SVG。全程 thinking，思考预算 128 token，每请求最多生成 512 token；temperature=1、top_p=.95、top_k=20、min_p=0、presence/frequency penalty=0、repetition penalty=1、seed=42。

| 指标 | IQ3 推荐配置 | IQ4 推荐配置 |
|---|---:|---:|
| 全任务聚合 prefill，首遍 | **549.50 token/s** | **582.34 token/s** |
| 全任务聚合 prefill，有效 | **517.99 token/s** | **378.67 token/s** |
| 图片编码 | **0.30 秒（GPU）** | **10.70 秒（CPU）** |
| 图片回答 decode，含思考 | 37.35 token/s | 41.25 token/s |
| 图片回答输出量，含思考 | 78 token | 66 token |
| 写代码 decode，含思考 | **38.28 token/s** | **41.04 token/s** |
| 整卡采样峰值显存 | **15639.10 MiB** | **15539.10 MiB** |
| 峰值时可用显存 | **412.90 MiB** | **512.90 MiB** |
| 运行阶段进程 RAM 峰值（RSS，不含加载） | **3012.52 MiB** | **3674.52 MiB** |

prefill 按全部 3 个请求聚合：新增输入总量除以对应总耗时。共 12617 个新增输入位置，其中 12133 个文本 token、484 行视觉输入，不计缓存历史或前轮生成量。首遍时间 IQ3/IQ4 为 22.961039/21.666163 秒；有效时间为 24.35754/33.31895 秒，包含图片编码和缓存管理。任务一没有历史重算。

RAM 取请求阶段的进程 VmRSS 峰值，排除模型加载；显存取 NVML 整卡采样峰值。decode 包含思考 token。两组均正确识别三种颜色和形状，代码续接仅追加 46 行文本，没有重新编码图片；代码达到 512 token 上限，未检查完整页面功能。各配置测一轮，表格不代表满预算、多图和长输出的组合压力。

IQ4 两种 MTP 量化方式在相同任务一输入下，正文和思考内容均一致。原 Python/原生工具版本的代码 decode 为 40.99/41.04 token/s，草稿接受数为 319/386 和 320/384，峰值显存相同。当前表格采用原生工具版本，不把单次略高的接受率视为稳定优势。

## 任务二：256K 工具测试的历史结果

[长程测试说明](long-context-benchmark-2026-09-14.md)保留两种配置接近 256K 的结果和运行方法。**其中 IQ4 使用切换前的 Python 量化 MTP 权重，当前原生量化版本尚未复测任务二。** 该历史结果不作为当前 IQ4 文件的长程验收。

## 复测任务一

脚本使用固定的 RTX 5060 Ti UUID，换机器前需调整 `scripts/multimodal_canary.py` 中的 `GPU`。运行前应确保该 GPU 和测试端口 18201 可用；每组完成后脚本会关闭自己启动的服务。

```bash
python3 scripts/multimodal_canary.py \
  --query-policy user --kv q8_0 --draft-kv f16 \
  --image-max-tokens 512 --budget 36864 --reserve 16384 --ctx 262144 --batch 512 \
  --long-words 12000 --quick --performance --thinking-budget 128 \
  --folder logs/task1-iq3

python3 scripts/multimodal_canary.py \
  --model models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf \
  --mmproj models/unsloth/Qwen3.8-27B-GGUF/mmproj-BF16.gguf --device cpu \
  --query-policy user --kv q5_0 --draft-kv f16 \
  --image-max-tokens 512 --budget 32768 --reserve 12288 --ctx 262144 --batch 512 \
  --long-words 12000 --quick --performance --thinking-budget 128 \
  --folder logs/task1-iq4

python3 scripts/summarize_canary.py logs/task1-iq3 logs/task1-iq4
```

输入、响应、逐请求 trace、NVML/RSS 采样和汇总保存在指定目录。汇总脚本只读取已有记录，不启动模型。

本地原始记录（不随仓库分发）：IQ3 为 `logs/iq3_budget_36k16k_image_thinking128_20260914/`；IQ4 为 `logs/iq4_mtp_quant_task1_20260914/native_quantize/`。测量使用的服务二进制 SHA-256 为 `b0f0a9ad776395f4133bdf2e850bbd238e18eb8877de6c3a5758c62fc6717b5c`；当前 IQ4 模型 SHA-256 为 `5ba8b178ab74794bcc5d901695706bc520b501543bdf397e8700cedd37891348`。
