# Bonsai 小显存版本：无 MTP 的兼容分析

分析日期：2026-09-20。分支：`kvmem-bonsai-llama.cpp`。
基线：rc3 快照 + Prism `9a9394a895b96003ca842a6041cb28ac49a108f7`。

本方案替代 `prism-compatibility-analysis.md` 中包含 MTP/Record-Fold 的首版范围。本文是源码分析和实施方案，尚未修改推理代码、编译或加载 Bonsai 验证。

## 结论

难度从完整兼容方案的中高，降到中等。新增量化类型和 Hadamard 内核直接使用固定的 Prism 实现；首版主要移植 KVMem 的普通缓存接入、逻辑位置、检索和多轮状态处理。无需为了 KVMem 再改写 Prism 的 GDN 内核。

原始补丁 38 个文件、107 个片段，16 个拒绝片段分布于 10 个文件。CPU GDN、CUDA GDN、CUDA 分派、recurrent.cpp、delta-net-base.cpp 这 5 个文件的 11 个拒绝片段属于可省去的 Record/Fold 移植。剩余已知文本冲突是 context/model includes、KV header、KV cell、KV cache，共 5 个文件各 1 个片段。这是范围缩减统计，并非新补丁已通过编译。

## 运行时关闭与构建时解耦

server 已有 `spec_mtp=false` 默认值，并会在未启用 MTP 时设置 `mtp_state=0` 和 `load_mtp=false`。但旧代码仍无条件编译或引用：

- `llama-memory-kvmem.cpp` 的 GdnReplay、replay_capacity/replay_l、CUDA fold 接口。
- capture 图复用逻辑中的 `recr_->replay_recording`。
- hybrid 构造函数末尾新增的 replay 参数。
- `tools/kvmem-spec.cpp`、MTP adapter 和 MTP/GDN 测试目标。
- factory、capture、换入换出代码中的 MTP follower 方法。

因此只传 `--spec-type none` 无法让未移植的源码编译通过。推荐首版用明确的功能边界禁用 KVMem MTP/Record-Fold，实现及依赖从编译路径移除，必要的公共入口保留明确的不支持响应。不要给 Prism recurrent 类添加一套无作用的伪 replay 字段来凑编译。

可以用编译开关保留旧实现供将来使用，但实验版本应固定关闭，启用未验证组合时明确报错。编译开关必须同时覆盖 factory、capture、follower、spec 驱动和测试目标，不能只删 CMake 源文件后留下未解析引用。移除死代码本身主要减少移植负担，显存收益来自不加载/不分配相关运行时资源。

## 补丁边界

| 处理 | 内容 |
| --- | --- |
| 保留 | out-of-tree adapter 构建入口、内存工厂、Q/K/V 捕获、图复用约束、harvest |
| 保留 | batch/ubatch 逻辑位置、KV storage 接口、逻辑位置删除、空洞保留和 attention mask |
| 保留 | Qwen 普通 attention 层捕获、缓存量化/换入换出、主模型 hybrid adapter |
| 保留 | 多轮 query replay、递归状态 checkpoint/restore、必要的多模态位置基础设施 |
| 保留 | 与 MTP 无关的 server/template/reasoning 修复，逐项确认 API 兼容 |
| 省去 | common/speculative 的 KVMem MTP 修改、MTP 专用 Qwen 捕获和 ctx_other 注入 |
| 省去 | RECORD 算子、GDN K=0、CUDA fold、CPU scratch 修改和 Record/Fold 专用测试 |
| 省去 | llama-memory-recurrent 和 llama-memory-hybrid 的 replay 扩展、delta-net-base 的 recording 分支 |
| 调整 | 顶层 adapter 和工具中的 MTP/replay 引用，以及构建/测试/启动器配置 |

`ggml/include/ggml.h` 中的 RECORD 修改可以省去；`include/llama.h` 中普通 batch 的逻辑位置信息仍可能需要。不能按文件名含有 replay 或 MTP 就整文件删除，应按用途拆片段。

## 仍须保留的递归状态处理

Bonsai 是混合注意力模型。关闭 MTP 后，普通线性注意力仍需当前 GDN 状态和卷积历史。

Query replay 与 MTP Record/Fold 是两套用途：前者在检索、更换上下文工作集和多轮前缀复用时重新处理查询；后者用于草稿验证后提交被接受的状态前缀。关闭后者不能删除前者。

当前 server 使用 `std::vector<uint8_t>` 在主机保存 query-begin 和 generation-start checkpoint，通过 `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` 保存/恢复 recurrent 状态。Prism 原生 hybrid 已支持 PARTIAL_ONLY 跳过 attention cache 的读写。因此有现成接口可复用，不需要新增 GDN 算子。

这些主机 checkpoint 仍消耗 RAM，并可能增加 PCIe 传输延迟；不能宣传为关闭 MTP 后不再需要任何 checkpoint。`n_rs_seq=0` 应在实际上下文参数和分配日志中确认，不能只检查 CLI 字符串。

## 剩余正确性风险

1. KV logical_pos：Prism cell reset 使用 memset，必须显式恢复 -1 sentinel；文本和 embedding 写入均应更新逻辑位置，并保留 M-RoPE x/y。
2. 稀疏 KV：恢复旧块时不能按连续区间假设清理缓存；attention mask 必须覆盖恢复后的有效位置，不能使用旧 mask 快捷路径。
3. 多轮 GDN：删 query 对应 attention KV 时不要随意回退 GDN；通过 checkpoint 恢复和重放到正确游标，避免把查询累计两次。
4. 图复用：移除 Record 状态判定，但保留 Q/K 捕获开关和逻辑位置相关的复用条件。
5. Hadamard：保留 Prism 权重变换，捕获投影后的语义 Q/K/V；不要与 KVMem 自身 KV 量化旋转混用。Prism 可选 K-cache mean centering 首版不启用，因为直接 stage-in 必须遵守相同的中心化约定。
6. 模型契约：实际 GGUF architecture、head dimensions、RoPE、层过滤和 chat template 仍需加载验证；模型名称相似不代表所有尺寸及接口相同。

不接入 Record/Fold 后，上一版分析中的 RECORD/raw-gates 断言冲突不再属于首版问题。普通 Prism raw-gates 可以保留，不需要全局关闭。n_rs_seq=0 时其常规 ring/rows 优化不会因为 MTP 被启用。

## 小显存预算

官方权重大小约为 PTQ1_0 5.95 GB、PQ2_0 7.21 GB，差约 1.26 GB。小显存首选 PTQ1_0 做验证；它与 PQ2_0 的速度优劣取决于设备和 prefill/decode 阶段，不能用文件大小推出吞吐。

总显存需要分别测量：常驻权重 + attention KV 池 + 当前递归状态 + 计算图/临时 workspace + KVMem capture/stage-in 缓冲 + CUDA/显示占用。

- attention KV 池当前按 budget + gen_reserve 分配；必须同时调整二者。
- budget=0 在现有实现中可能退回按 n_ctx 分配，不能把它当作最省显存设置。
- KV 每 token 字节数应按实际 attention 层逐层累加 `ggml_row_size(type_k, dim_k) + ggml_row_size(type_v, dim_v)`，再计 padding/layout 差异。三值权重不等于三值 KV。
- capture 当前有设备 staging 和 pinned host 缓冲，随捕获张量/ubatch 增长。降低 prefill ubatch 可减峰值，但会影响吞吐。
- query checkpoint 主要在 RAM，历史 KV 的分层存储也会将一部分容量压力移到 RAM/SSD。
- 不加载视觉头能进一步减少开销；编译链接 mtmd 不等于加载视觉权重。

8 GB 是有价值但必须实测的目标，尤其是 PTQ1_0、单序列、有限 KV 工作集、较小 ubatch、纯文本组合。12 GB 的余量更大。6 GB 不应承诺全模型 GPU 常驻，可能需要部分权重 CPU offload 并承担性能代价。上述为容量判断，均不是已验证的硬件支持声明。

不要将关闭 MTP 的收益估算成再省掉一个完整 27B 模型：MTP 可能复用权重，rc3 Record/Fold 本身也是减少传统回滚快照开销的方案。实际差额取决于 draft 权重、context、KV 和缓冲配置，应通过分配日志和显存峰值验证。

## 启动器需要修改

`scripts/windows/start-server.ps1` 当前强制 `draft-mtp`，Mtp 参数范围为 1-5；强制要求 mmproj；IQ3 默认 budget=36864、reserve=16384，并设置 262144 context。这些是原 rc3 使用场景的配置，不适合直接作为小显存 Bonsai 默认值。

建议新增明确的 Bonsai 配方：spec none、无 draft、纯文本不要求 mmproj、单序列、可调整的 KV budget/reserve 和 batch/ubatch。区分总上下文容量与 GPU 常驻工作集；先测较短上下文，再扩大到长对话。小 KV budget 会增加检索丢失重要信息的风险，gen_reserve 太小则约束长推理输出，必须一起做质量测试。

## 实施及验收顺序

1. 固定无 MTP 的编译边界，提取最小 KVMem 补丁；确认不再依赖 RECORD/fold/replay 字段。
2. 使用现有 Windows/CUDA 工具链构建，验证新补丁在干净 Prism pin 上应用和重复执行；后端库统一构建，不混用原 runtime。
3. 同一 Bonsai 文件、相同参数、关闭检索，对照未修改 Prism 的 logits/短文本与 prefill/decode；不能只凭输出看起来正常。
4. 开启 KVMem，但工作集足以容纳输入，验证无淘汰情况与普通推理一致。
5. 刻意使用小预算触发淘汰/恢复，验证检索、mask、位置和多轮 checkpoint；覆盖前缀复用、工具轮、取消后继续和长输出。发生检索淘汰后不要求与完整上下文 logits 完全相同。
6. 在目标显存硬件测加载、prefill、检索换入和 decode 四个阶段峰值；记录权重、KV、递归状态、capture/staging 和主机内存。观察是否有非预期 CPU 回退。
7. 再扩大上下文，验证早/中/晚信息检索，评估小预算质量及吞吐；跑普通 Qwen 非 MTP 回归。

MTP/Record-Fold 测试不纳入这个版本的通过率，不能用 skip 冒充通过；保留的主机缓存、KV、server 和启动器测试应按本次范围运行。

## 难度与工作量估算

- 代码移植：中等；主工作是清理编译依赖和修复 KV 接入，不再是 GPU kernel 开发。
- 正确性验证：中等偏高；风险集中在稀疏缓存和多轮状态一致性。
- 达到具体小显存容量：需实测调参，8 GB 尚未确认。
- 维护：固定 Prism commit + 一组小补丁，升级时重跑回归；不重写其量化与 Hadamard 实现。

粗略预计 1-3 个工程日得到基础可编译/可生成版本，再用 2-4 个工程日完成检索、多轮与小显存验证，总计约 3-7 个工程日。不包含新写量化内核、跨所有后端或复杂多模态适配；实际 API/模型不匹配会延长时间。

官方模型依据：https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
