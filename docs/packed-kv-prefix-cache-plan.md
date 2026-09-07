# Packed GPU KV + prefix cache

**状态：** 阶段 A 已进 **v0.9.0**。阶段 B、C 已进 **v0.10.0**。下一步 D（可选落盘）。不改 FA，不窗口化 pos。  
**前提：** cell 上始终是**原始 pos**。因此冷块回来不必再 RoPE。

---

## 目标

1. **不再存未旋转 raw-K 整行。** K 和 V 在 stage-out 时按 GPU 量化格式（默认 Q8）整行拷出，stage-in 整行拷回。
2. **Prefix cache：** 下一轮只 prefill 新增 token。公共前缀的 GPU KV / GDN / MTP 接着用。
3. **尽量复用 llama.cpp：** 进程内用 server 的公共前缀 + `llama_memory_seq_rm`；落盘用 `llama_state_*`。不新发明缓存格式，不搬 qw3 `generate_mtp`。

## 非目标

- compact-window `[0..W)` / 改 FA / 改 `speculative.cpp`
- 接入上游 `llama-server` 核心（先独立 `llama-kvmem-server`）
- 多 slot / continuous batching
- 从已 RoPE 的 Q8 算 mean-K

---

## 数据模型

| 数据 | 何时产生 |
|---|---|
| GPU 工作集 K/V（Q8 成品） | 图写入 |
| 挤出块 packed K/V | **stage-out D2H** |
| mean-K（每块 × 注意力层，F32） | **prefill 写块时**；decode 在**块写满**时补 |
| query Q | 每次 retrieval 前（现有 capture） |
| 块表 | 现有 `KvMemStore` |

检索：`mean-K · Q`。还原：packed 拷贝。

---

## 阶段 A — 去掉 raw-K（已完成）

- Prefill harvest：pre-RoPE K 只 `write_layer_mean_k`，不 `write_layer_k_rows`。
- 主干 stage-out：与 MTP 对称，D2H packed **K 和 V**。
- 主干 `write_block_to_gpu`：`copy_k_gpu` / `copy_v_gpu` + slab H2D。产品路径不再解 Q8 再 RoPE。
- `has_block`：mean 或 packed K 即可，不要求 raw-K 行。
- dump：packed 对 GPU，不再 rebuild-from-raw。
- MTP prefill 不再 D2H 未旋转 K；冷块 `has_k_gpu` memcpy。
- Prefill 块写满后异步 D2H packed K+V（与后续 ubatch 重叠）；挤出 `has_k_gpu` 则 skip。

**0.8B / 5050 门（2026-09-07）：**

| 项 | 结果 |
|---|---|
| `raw_kv_store_test` / `kvmem_store_test` | PASS |
| identity Q8_0 | PASS |
| recency 256 不召回 | PASS |
| retrieval 256 `--no-think` | **BLUEBIRD-42**；`selected` 含 block 13；`L23 K/V packed_vs_gpu cos=1.000`；packed bytes mismatch=0 |
| MTP canary n_max=2 | PASS；`n_no_raw=0`；retrieval+MTP 打出 BLUEBIRD-42；accept 45.7% |

## 阶段 B — decode mean-K（已完成）

- retrieval pin 之后：pre-RoPE K 在 GPU 上 running sum；**块写满或本轮结束** D2H 一份 mean。
- 只计已接受 token；MTP 草稿不算（verify stash → accept 后 `commit(ids.size())`）。
- 块未满就被挤出：stage-out 前落 running mean。
- GPU add 失败则 `ggml_backend_tensor_get` host fallback。
- 短 prompt（`query_begin==0` 不 pin）不会走这条路径。

**0.8B / 5050 门（2026-09-07）：**

| 项 | 结果 |
|---|---|
| identity Q8_0 | PASS |
| recency 256 不召回 | PASS |
| retrieval 256 `--no-think` | **BLUEBIRD-42**；`decode_mean_flush` 出现 |
| 生成满 1 块 | `flush block=14 n=32` gpu=6 host=0 `rms≈1.20`；`n_tok=64 n_miss=0` |
| MTP n_max=2 | PASS；`n_no_raw=0`；retrieval+MTP BLUEBIRD-42；verify `n_pos=3` 后 flush；accept 68.8% |

## 阶段 C — 进程内 prefix reuse（已完成）

`llama-kvmem-server` 不再每请求 `llama_memory_clear`。客户端带完整 messages；单 slot。

```text
n_past = token LCP(cached, new_prompt)
seq_rm(n_past, -1)     // 主干 + MTP
truncate 块表 + raw 冷块
unpin retrieval，重新 harvest 后缀 mean-K
GDN：能对齐则不动；否则 restore 上一轮 query ckpt 再 catch-up
只 prefill 后缀
apply_retrieval + query replay + MTP follow
decode
```

助手文本 roundtrip 往往对不齐整段 cached，LCP 落在 generation prompt 附近。此时用 retrieval 的 GDN ckpt catch-up，不整段重算。对不上 ckpt 则 fallback clear。

**0.8B / 5050 门（2026-09-07）：**

| 项 | 结果 |
|---|---|
| greedy + streaming chat | PASS |
| turn-1 retrieval | **BLUEBIRD-42** |
| turn-2 完整 messages | `reused=1 n_past=846 n_prompt=891 n_new=45`；GDN catch-up 834→845；输出 **BLUEBIRD-42** |

## 阶段 D — 落盘（可选）

补齐 `llama_memory_kvmem::state_write/read`（块表 + mean-K + 挤出 packed KV）。之后 `llama_state_save_file` / llama-server prompt cache 自动带上。第一版可不把全部冷块写入 blob。

---

## 顺序

```text
A  packed K/V + mean-K only
B  decode 块满补 mean-K
C  kvmem-server 前缀复用
D  state_write 补齐
```

C 没有 B 则助手输出进不了下一轮 ranker。

## 测试

| 阶段 | 必过 |
|---|---|
| A | identity；needle；MTP n_max=2 接受率不回 0 |
| B | 生成 ≥1 块后 mean-K 非空 |
| C | 第二轮 cached ≈ 前缀；仍能召回旧针 |
| D | save/load 后续写 |

速度不当 GO/NO-GO。召回 miss、MTP 0% 算失败。
