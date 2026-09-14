# IQ3 多模态实现与验收

日期：2026-09-14。功能已完成并启动。代码位于当前工作树，没有创建提交或修改 submodule pin。
依据：[设计](multimodal-integration-design-2026-09-13.md)、[计划](multimodal-implementation-plan-2026-09-13.md)。

## 已实现的功能

- IQ3 加载指定 `mmproj-Q8_0.gguf`；视觉编码可在 GPU 或 CPU，主模型设备独立保持不变。
- OpenAI `messages[].content[]` 支持 `text`、`image_url`，图片可用 base64 data URI 或 HTTP(S) URL。
- 单图、多图、图后追问、SSE、工具调用和工具结果续接。
- 同一驻留会话复用有效文本及图片前缀，处理新增 suffix；图片后的文本实际使用 MTP 2。
- 逻辑 KV 行、原生 M-RoPE 位置、GPU slot 分开；视觉主 KV 与草稿 KV 可换出、恢复及重排。
- 请求失败或取消回到可恢复的前缀；缺少有效 recurrent 检查点时自动按缓存未命中处理当前完整请求，不要求客户端额外传重置参数。

复用的是 llama.cpp 的 `mtmd`、原生图片读取/解码与内容 ID、`server_tokens` 前缀比较、chat template、采样器、MTP 模型图和 KV 搬运路径。没有另写视觉 encoder、RoPE 或 attention kernel。

## 默认启动配置

```bash
scripts/start-iq3.sh
# 切换视觉设备需要重启：
MMPROJ_DEVICE=cpu scripts/start-iq3.sh --restart
MMPROJ_DEVICE=gpu scripts/start-iq3.sh --restart
```

| 项目 | 默认值 |
|---|---|
| 主模型 | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` |
| 视觉头 | `models/unsloth/Qwen3.8-27B-GGUF/mmproj-Q8_0.gguf` |
| 视觉设备 | GPU；`MMPROJ_DEVICE=cpu` 改为 CPU |
| 主 KV | Q8_0 |
| MTP | 2；草稿 KV F16 |
| 上下文 / batch | 262144 / 512 |
| 检索 / 生成预留 | 40000 / 16000；实际按 128 行块向下对齐 |
| 图片上限 | `IMAGE_MAX_TOKENS=512`，使用原生动态分辨率处理 |
| Thinking / reasoning budget | 开启 / 4096 |
| 采样 | temperature 1、top_p 0.95、top_k 20、min_p 0、presence/frequency 0、repeat 1 |
| 服务 | `127.0.0.1:18200` |

主模型、主 KV、batch、检索预算、生成预留及采样值保留原 IQ3 配置。草稿 KV 显式设为此前选定的 F16：旧 IQ3 脚本会继承 Q8，在完整预算加 GPU 视觉头时，实测首次大批次 prefill 触发 CUDA 分配失败。F16 路径完成了长会话验收。可用 `SPEC_KV_DTYPE` 覆盖草稿类型；类型支持不代表任意预算组合都能装入 16 GB。

使用原生 `warmup=false`，让视觉库首次编码时按实际图片预热。原先固定 1472×1472、2116 个视觉 token 的 dummy graph 会提前保留约 248 MiB 计算空间，甚至超过指定图像上限。此调整不改变图像预处理或模型计算。

完整预算的 GPU 模式显存余量较小，详见实测表；需要更大图片或更多显存余量时可使用 CPU 视觉头。`MMPROJ` 可覆盖 projector 路径，`IMAGE_MAX_TOKENS` 可覆盖原生上限。

## 图片请求和续接

```python
import base64
import json
import urllib.request

image = base64.b64encode(open("photo.png", "rb").read()).decode()
messages = [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
    {"type": "text", "text": "描述这张图片。"},
]}]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def chat(messages):
    body = {"messages": messages, "max_tokens": 512}
    req = urllib.request.Request(
        "http://127.0.0.1:18200/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(req, timeout=600) as response:
        return json.load(response)["choices"][0]["message"]

answer = chat(messages)
print(answer)
messages += [answer, {"role": "user", "content": "再描述一下左边的物体。"}]
print(chat(messages))
```

把 `image_url.url` 换成 HTTP(S) 图片地址也可以。同内容不同 URL 可命中；同尺寸不同图片不会误命中。后续请求继续携带完整 messages 历史，包含原图和 assistant 返回内容。`stream: true` 返回 SSE，包括工具调用 delta、usage 和 `[DONE]`。

未配置 projector 时，图片请求返回 HTTP 400。图片组连同必需文本超出预算也返回 400，不静默丢弃图片。有有效 recurrent 检查点时继续复用前缀；切换对话、工具定义或编辑历史后找不到可恢复检查点时，服务自动清理失效状态并重新处理当前完整 prompt。`"cache_reset": true` 仍可主动重置，但客户端不需要用它规避缓存错误。

2026-09-14 修正了最初的 HTTP 409/SSE 错误策略：仅凭公共 token 数大于 4 不能区分历史编辑与新会话。dsh 的主会话和自动标题请求会共用同一个服务，缓存未命中属于正常情况。`multimodal_reset reason=no_recurrent_checkpoint` 日志记录公共前缀、旧缓存长度及最早检查点，usage 如实报告 0 个缓存命中；正常追加图片仍走原有增量路径。

同轮回归还修正了普通解码重复请求时的空回答：recurrent 检查点不保存 logits，恢复 prompt 末尾后必须至少重新计算一个 token 才能采样。普通解码因此选择末尾之前的有效检查点；MTP 保持原本由 `spec_generate` 处理待解码 prompt token 的逻辑。

## 增量 prefill 证据

最终长会话使用真实 27B、5060 Ti、默认 40K/16K 预算、主 Q8、MTP F16、图像上限 512。日志在 `logs/multimodal_20260914/gpu-acceptance/`。

主 context 始终为 `0x62009e4c0950`。先处理 60020 个 prompt token 并回答 OK；之后追加图片：

| 请求 | 命中旧行 | 新文本行 | 新图片行 | query replay 行 | encoder 调用 |
|---|---:|---:|---:|---:|---:|
| 追加第一张图 | 60021 | 30 | 484 | 24 | 1 |
| 旧图追问 | 60545 | 24 | 0 | 24 | 0 |
| 追加第二张图 | 60571 | 28 | 484 | 22 | 1 |
| 替换第一张图 | 60027 | 18 | 484 | 18 | 0 |

换图时 encoder 为 0，是因为该新图刚作为第二张图编码过，命中了 CPU 投影缓存；主模型仍重新计算受换图影响的图片行。没有使用旧图片的 KV。

`image.trace.log` 的实际 decode 区间从 60021 开始；query replay 也只覆盖新增问题的短段。`followup.trace.log` 没有旧图片行的 decode。`logical_cursor=60535`、`model_cursor=60073` 分开记录，`mtp_synced_rows=60535`，草稿包含完整视觉输入。

CPU 投影嵌入缓存上限 128 MiB；此例每图约 9.45 MiB。最多保留四份 recurrent/MTP/尾块均值检查点，另有当前请求的回滚快照；四份检查点在本模型上约 600.6 MiB。位置元数据约 20 字节/逻辑行，CPU 常驻，不随 attention 层数复制。

## 性能与显存

最终长会话的初始 60019 行 prefill 用时 121.66 秒，约 493 行/秒，包含检索、520 行 query replay 和检查点工作。第一次追加图片：encoder 230.92 ms，整个增量 prefill 4478.20 ms；第二张图 encoder 169.24 ms、增量 prefill 3231.20 ms。前缀恢复、检索搬运、检查点和 query replay 均计入增量 prefill 墙钟时间。

长会话完整矩阵整卡采样峰值 **16015.10 MiB（15.64 GiB）**；进程 RSS 采样峰值 **11951.45 MiB**，RSS 含映射模型页与 CPU 缓存。显存由 NVML 采样，属于采样到的峰值，不能视为所有瞬时分配的严格上界。

另一次 60K 历史的图后代码生成使用 Thinking 推荐采样，生成 512 token：decode **33.68 token/s**，MTP 接受率 **81.5%**。该测试显式把 reasoning budget 限为 128、输出限为 512，便于稳定计时。它改变了模板的系统前缀，发生了预期的缓存失效；不将这一请求宣称为增量命中。固定采样和短历史的 CPU/GPU、普通/MTP 对照结果在最终矩阵目录中。

短回答只生成 1–9 token，首轮开销对 tok/s 影响很大，不能用这种样本评价稳态 decode 速度。

同一张图、短历史、同样的 40K/16K 预算，固定 512 token 的代码生成对照：

| 模式 | 首张图 encoder | 首张图整个 prefill | 代码 decode | MTP 接受率 | 本轮显存峰值 |
|---|---:|---:|---:|---:|---:|
| GPU 视觉 + MTP 2 / F16 | 236 ms | 3.691 s | **42.04 tok/s** | 77.4% | 15567.10 MiB |
| GPU 视觉 + 普通解码 | 268 ms | 3.403 s | **26.82 tok/s** | — | 14759.10 MiB |
| CPU 视觉 + MTP 2 / F16 | 7784 ms | 10.905 s | 未运行此代码对照 | — | 14787.10 MiB |

前两项代码请求使用相同 Thinking 采样与输出长度，MTP decode 约为普通解码的 **1.57 倍**；随机采样不要求两条路径生成完全相同的代码。首张图的 greedy 回答逐字一致。CPU 行是完整图片功能回归的峰值，其余两行是图片加代码请求的峰值，不能把不同请求矩阵当作严格显存 A/B。

## 验证与复现

```bash
# 主机块存储、选择、CPU/NVMe、partial-block 截断测试
LD_LIBRARY_PATH=/home/leye/kvmem_qw3/.cu13-env/lib \
  .venv/bin/ctest --test-dir build --output-on-failure

# 实际 GPU 字节测试，需要本地 0.8B MTP 模型；普通 CTest 无模型时跳过此项
source scripts/gpu.sh small
build/bin/kvmem-mtp-kv-test \
  models/unsloth/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-Q8_0.gguf

# 27B 测试固定绑定本机 5060 Ti；运行前停止占用同卡的推理服务
python3 scripts/multimodal_canary.py \
  --budget 40000 --reserve 16000 --ctx 262144 --long-words 60000 \
  --image-max-tokens 512 --draft-kv f16 --trace \
  --folder logs/multimodal-check

# 官方采样代码生成；--spec none 为普通解码对照，--device cpu 为 CPU 视觉头
python3 scripts/multimodal_canary.py --quick --performance \
  --budget 40000 --reserve 16000 --ctx 262144 \
  --image-max-tokens 512 --draft-kv f16 --folder logs/multimodal-perf
```

测试脚本只停止自己启动的测试进程。逐请求保存 JSON、响应、decode 范围、encoder 时间、MTP 接受数、NVML 曲线和 RSS 曲线。

- 主机五项 CTest 通过；带模型手动执行主/MTP KV 字节测试通过七种目标/草稿组合，包含 Q8/Q5/Q4/F16、不同 slot、D2D、部分块、共享 t 坐标以及 batch 分割/重排。
- 原生 mtmd CLI 用同一个真实 IQ3 与 projector 在 CPU/GPU 均识别出红方块、蓝圆、绿三角；自定义普通解码与 MTP 的 greedy 图片回答一致。
- 首次 60K 完整矩阵通过：增量图片、HTTP 与 data URI 同图命中、多图、换图、流式、无效图片、prefill/生成取消、错误后的继续请求、工具调用、工具结果，以及当时的历史 409 与显式重置。后续 dsh 兼容修复将历史 409 改为自动缓存未命中处理，并补充会话切换回归。
- `chat_sampling_canary.py` 通过普通/MTP 共 12 个生成案例、32 个非法请求；`server_smoke.py`、`mtp_canary.py` 通过。
- 全量 patch 和增量 patch 均实际回放，检查修改文件逐字节一致；再次执行幂等。构建不再依赖旧的 `git am` 或本机未记录的 native 修改。
- 最终 5060 Ti 矩阵还通过主/草稿 Q5、Q4、F16 的真实图片测试，原生内存 `--no-kvmem` 图片 + MTP，以及容量错误后的正常请求；CPU 视觉模式完成同样的取消、换图、工具和历史恢复流程。

`logs/multimodal_20260914/` 保留构建日志、测试结果、修改前 diff、旧服务启动参数、默认配置实测和失败尝试。`gpu-long`、`gpu-long-256` 记录早期 Q8 草稿显存失败；`gpu-long-final` 的测试断言失败来自混入 Thinking 模板切换后仍要求缓存命中，最终功能验收使用独立的 `gpu-acceptance`。

## 实现边界

当前为单个驻留会话槽，不提供多会话持久缓存。模板/系统提示词改变、主动重置或切换无缓存对话会按真实前缀失效处理；历史编辑超出检查点时自动重建。标题请求或其他独立会话占用槽后，返回原会话可能需要完整 prefill；同一有效驻留前缀的追加请求仍增量处理。图片检索按完整媒体组选择，最近图片保留，但跨大量历史图片的语义召回质量尚未进行独立基准评估。

保存的是原生已旋转、已量化的 packed KV；恢复同时还原 cell 的原始位置。CPU/NVMe partial-block 测试和 GPU 字节测试验证了数据路径；没有把稀疏检索结果与原生完整窗口要求逐位等同，也没有宣称已完成真实图片的全词表逐 logit 对照。

本轮还修复了拒绝/EOS 后的草稿多余行、纯文本 GDN/MTP carry 回退、SSE `[DONE]` 多写一个 NUL，以及回退部分块时 packed 有效行和 mean-K 检查点的一致性。

## 最终服务状态

默认 IQ3 服务已在 5060 Ti 启动，地址 `http://127.0.0.1:18200`。dsh 缓存兼容修复后重启为 PID `42619`；脚本在 `scripts/start-iq3.sh`，当前 PID 以 `logs/iq3_18200.pid` 为准。修复后的启动记录在 `logs/dsh_cache_fix_20260914/restart-final.log`，默认配置保持不变。此前 IQ4 进程配置保留在 `logs/multimodal_20260914/service-before.json`。

## dsh 缓存兼容修复验收

最新 dsh 日志记录了 07:49:26、07:50:00 和 07:50:31 的缓存错误；后两次是新会话首轮。服务端依次拒绝了 181394、197、12107、162、12072 token 的请求。根据最后一个会话日志重建主请求与标题请求后，修复前均复现了 HTTP 200 流中携带 `history edit is outside retained recurrent checkpoints` 错误。主请求包含 3 条消息及 52 个工具，prompt 恰为 12072 token。

修复后在默认 IQ3 / GPU 视觉头 / Q8 主 KV / F16 MTP 2 配置上重放：

- 主请求正常回复“你好！有什么我可以帮你的吗？”，没有传 `cache_reset`。
- 相同主请求再次调用命中 12071 token，仅余 1 个待生成路径处理的 prompt token，没有重复旧 prefill。
- 标题和主请求交替时，实际公共前缀为 41 token，早于保留检查点；自动按缓存未命中处理，SSE 不再返回缓存错误。
- MTP 和普通解码各 20 个回归场景通过，包含同系统前缀的新会话、标题插入后返回原会话、工具定义编辑、显式重置、图片追问、换图、SSE 和取消恢复。普通解码重复请求恢复后重新计算 logits，正确返回 Blue。

日志、重建请求、响应与 trace 位于 `logs/dsh_cache_fix_20260914/`；最终矩阵为 `gpu-mtp-final` 和 `gpu-ordinary-final`，线上重放汇总为 `live-replay.json`。重放为控制测试时长将主请求输出限制为 64 token（原客户端为 20000）；标题本身上限为 64。部分 Thinking 回复在预算内只有 reasoning 内容，因此这些重放验证流协议、状态与缓存，不用于评价标题质量。首次主请求确实包含最终中文回复。

`gpu-mtp` 留有测试断言误将 SSE 分块字符串当完整单词的失败记录；拼接后断言已修正。`gpu-ordinary` 留有首次发现旧 logits 导致空回答的失败记录；修复后的 `gpu-ordinary-final` 全部通过。
