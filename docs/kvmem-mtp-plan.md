# KVMem × llama.cpp MTP（方案 B）

**状态（2026-09-05）：** P7-0～P7-3 已过关，本地 milestone **`v0.5.0`**。27B/5090 固定 budget 256+256：T=8k–64k 主 KV 32 MiB、MTP KV **2 MiB** 不变。Writeup：[milestones/v0.5.0.md](milestones/v0.5.0.md)。

总计划入口：[modification-plan.md](modification-plan.md)。架构短文：[architecture.md](architecture.md)。

---

## 0. 决议

KVMem 的产品目标是 **逻辑长上下文、GPU 工作集有界**。MTP 只有一层注意力，但方案 A（draft 走 stock `llama_kv_cache`，`n_ctx` = 逻辑 ctx）会让这一层按全长涨：27B 在 256K 大约多 **1 GiB**，16K 大约多 67 MiB。长上下文 decode 还会让 MTP 的 FA 扫满 `n_ctx`，qw3 自己在 128K 上 MTP 甚至慢于不开 MTP。

因此 **只做方案 B**：MTP KV 与主干注意力 **同一套槽位池**（`budget + gen_reserve`），同一组 block id，同一个 slot 下标，cell 上仍是 **原始单调 pos**（不 pack `[0..W)`）。

对照：

| | 方案 A（不做） | 方案 B（本计划） |
|---|---|---|
| MTP `n_kv` | 逻辑 `-c` | `budget + gen_reserve` |
| 27B@256K MTP KV | ~1 GiB | ~1.5 MiB（池 384 cell） |
| 实现 | 工厂继续 `return nullptr` | follower memory + lockstep |

投机循环 **不搬 qw3 的 `generate_mtp`**。用 llama.cpp 已有的 `--spec-type draft-mtp`（`common/speculative.cpp`）。Adapter 不出现模型张量名。

27B Unsloth `UD-Q4_K_M` 主 GGUF 已焊 `blk.64.nextn.*`，开 MTP **不必**再下 `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`。独立 `-md` 头是可选的另一条 draft 模型，工厂按层 filter 处理，不写死 `blk.64`。

---

## 1. 目标形态

llama.cpp 保持两个 context：

| | target `ctx_tgt` | draft `ctx_dft`（`LLAMA_CONTEXT_TYPE_MTP`） |
|---|---|---|
| 图 | 主干（hybrid 则含 GDN） | `LLM_GRAPH_TYPE_DECODER_MTP`，只跑 nextn |
| Memory | 已有 `llama_memory_kvmem` / `_hybrid` | **`llama_memory_kvmem_mtp`**：plain 槽位池，仅 nextn 层 |
| 选块 | ranker 打主干 raw-K | 同一组 block id，不另打分 |
| 槽 / pos | block → slot；原始 pos | **同一 slot 下标、同一 pos** |
| FA | 扫池子 | 扫同一池子大小 |

两个 context 不能共享一个 `llama_memory_i` 对象。B 是 **follower**：

```text
ctx_tgt  llama_memory_kvmem         拥有 KvMemRuntime、主干 Raw-K、主干 GPU cache
ctx_dft  llama_memory_kvmem_mtp     不选块；问 target「这个 pos 在哪个 slot」
                                    自己的 llama_kv_cache 只有 nextn 层
                                    size = target->kv_size()   // 禁止按 draft n_ctx 重算
```

`cparams.ctx_other` 已指向 target（`speculative.cpp` 创建 MTP context 时设置）。

---

## 2. 非目标（本计划不做）

| 不做 | 原因 |
|---|---|
| 自研 draft/verify 状态机 | llama.cpp `draft-mtp` 已覆盖 Qwen 单头 / Step 多头 / Gemma4 共享 KV |
| 方案 A 当产品路径 | 显存随逻辑 ctx 涨，违背 KVMem |
| pack MTP 窗口为 `[0..W)` | 与主干「原始 pos + hole」冲突 |
| 改 FA kernel | 硬约束 |
| KVMem + CB + MTP | qw3 产品也拒；仍单 slot |
| DFlash2 / `--spec-type draft-dflash` | 另一种 drafter；MTP XOR DFlash2。需要时另开 PR |
| Gemma4 `is_mem_shared`（draft 共用 target KV） | 不是 nextn 条带；有需求再单独立项 |
| 把 MTP 条带塞进 NVMe | 一层很小；B 先 CPU 常驻 |
| 追上桌面 5090 + MTP/NVFP4 的 tok/s | 速度只记录 |

---

## 3. 硬约束（相对总计划的补充）

总计划第 3 节仍然有效。MTP 额外：

1. **Draft KV 与 target KV 分离。** nextn 不进主干 hybrid 的 attn filter（`il < n_layer()`）。两个 `llama_kv_cache`，同一 `kv_size`。
2. **Draft 不是 GDN。** recurrent 只在 target。verify reject 走 `n_rs_seq` GPU 平面（短后缀 `seq_rm`）；query replay 挖洞仍不回滚 GDN。host `PARTIAL_ONLY` 只在 `draft > n_rs_seq` 时用。
3. **Lockstep。** 同一 `block_id` → 同一 slot；同一 cell 偏移 → 同一 `pos`。`stage_in` / `stage_out` / `seq_rm` / retrieval writeback 成对发生。
4. **Retrieval 只打主干 K。** MTP K 跟着 selected 集合走。
5. **投机写入不是权威。** 未 accept 的 MTP cell 不得 harvest 进 raw-K。
6. **两套时钟。** 逻辑 pos vs 槽号。禁止 modulo 进 `n_ctx`。本地 RoPE 要求 `budget + gen_reserve ≤ n_ctx_train`（与 qw3 相同）。
7. **`gen_reserve ≥ n_max + query-replay 余量`。** 不够硬失败，不挤召回块。
8. **Query replay 是 prefill。** 先 GDN restore + 主干 replay，再重建 MTP prefix。
9. **单 drafter。** `--spec-type draft-mtp` 与 DFlash2 互斥（若以后加）。
10. **不改 `speculative.cpp` 的状态机。** 只在 factory / capture / tools 接线。

从 qw3 只借这些约束，不借 `generate_mtp` / paged MTP / 窗口 pack。

---

## 4. 阶段 DAG

```text
P4-2 ─┬─► P5（已完成，v0.4.0）
      │
      └─► P7-0 ─► P7-1 ─► P7-2 ─► P7-3
           接线      槽位条带    retrieval    长 ctx 显存
           + GDN                 + replay
           snapshot              同步
```

P7-0 允许 MTP KV 暂时仍是 stock（方案 A），只验证投机循环 + hybrid reject。P7-1 才是省显存的那一刀。不要把 B1 和「重写 speculative」绑在一起。

硬件：P7-0/P7-1/P7-2 日常 **Qwen3.5-0.8B Q8_0，RTX 5050**。日常 `unsloth/Qwen3.5-0.8B-GGUF` **没有** nextn；P7 用 `unsloth/Qwen3.5-0.8B-MTP-GGUF` 的同名 Q8_0（`scripts/download-test-models.sh mtp`）。P7-2 抽检 + P7-3：**Qwen3.8-27B UD-Q4_K_M，RTX 5090**（`source scripts/gpu.sh 27b`）。

速度不当失败。Retrieval 生成 miss 仍是 GO/NO-GO。

---

### PR P7-0 — 接线 draft-mtp + verify 时 GDN 可回滚

**标题：** `tools: wire llama.cpp draft-mtp; snapshot GDN on verify`

**依赖：** P4-2、P5

**改动：**

- `llama-kvmem-cli` / `llama-kvmem-server` 链 `llama-common`，认与 `llama-server` 相同的 flag：`--spec-type draft-mtp --spec-draft-n-max N`（可选 `--spec-draft-p-min`）
- Prefill +（若开启）retrieval + query replay 之后，decode 走 `common_speculative`，不再手写 1-token 循环。这样每个 target ubatch 之后会调 `process()`，draft KV 才能跟上
- retrieval pin 之后 **verify 的 n>1 不得再捕主干 Q/K**（今天 capture 用 `n_tokens>1` 开门，会在投机一步打乱 raw-K）
- hybrid：verify 前对 GDN 做 `PARTIAL_ONLY` snapshot；reject 时 restore。`n_rs_seq=0` 时不能指望 `seq_rm` 回滚 recurrent（P4-2 同构）
- 本阶段工厂对 MTP **仍可 `return nullptr`**（stock draft cache）。目的是先把投机 + GDN 做对

**不做：** follower memory；retrieval 后 MTP writeback；改 FA。

**退出：**

- 无 `--kvmem`：`--spec-type draft-mtp` 在 0.8B 上能出 token（对照 llama-server 行为）
- `--kvmem` + 短 ctx + **无 retrieval**（identity budget 或 recency 盖住全文）：0.8B、`n_max=2`、greedy；有 accept TRACE；identity 仍过（可关 MTP 比）
- reject 路径（低 `p_min` 或强制短链）：GDN restore 后后续 token 不崩、不乱序
- Device 必须是 RTX 5050

---

### PR P7-1 — MTP 槽位条带（方案 B 的显存刀）

**标题：** `adapter: llama_memory_kvmem_mtp lockstep slot-pool`

**依赖：** P7-0

**改动：**

工厂：

```text
MTP 且 ctx_other 是 KVMem
  → new llama_memory_kvmem_mtp(target)
  → llama_kv_cache：
        size = target->kv_size()     // 禁止按 cparams.n_ctx 重算池
        filter：
          同模型焊头：il >= n_layer()
          独立 mtp-only GGUF：全部层
MTP 且 target 不是 KVMem → nullptr（vanilla）
```

`llama_memory_kvmem_mtp`：

- `init_batch` 仍返回 `llama_kv_cache_context`（`graph_mtp` 的 `build_attn_inp_kv()` 不用改）
- `prepare_ubatches` **禁止自己 alloc**；按 pos 查 target 已分配的 slot
- `alloc_slot` / `free_slot` 只属于 target；follower 清同一下标
- 独立 `RawKvStore`，只存 MTP 层
- `0002`：`qwen35::graph_mtp` 在 `ggml_rope_multi` **之前**钉 MTP K（与主干 qwen35 那几行同构）。独立 `-md` 若走同一 `graph_mtp` 则不必再补

**不做：** retrieval writeback（P7-2）；NVMe；DFlash2。

**退出：**

- TRACE：MTP `n_kv` == 主池 `kv_size`（budget+reserve），不是 `-c`
- occupy：同一 `block_id` 在主干与 MTP 同一 slot，cell pos 一致
- 27B 短/16k 抽样（5090）：MTP KV 字节为池子量级（~数 MiB），不是 ~67 MiB @16k
- 0.8B identity（可关 MTP 或短链）仍过
- 无 FA 改动；`kvmem/` 仍零 llama 头

---

### PR P7-2 — retrieval / query replay 后同步 MTP

**标题：** `adapter: MTP stripe follows retrieval and query replay`

**依赖：** P7-1

**改动：**

- `apply_retrieval`：follower 用 **同一 selected 集合、同一 `layout_gpu_slots_by_orig_pos`**，从 MTP raw 写回（host RoPE，原始 pos）
- 主干 `stage_out` 某 block 时，MTP **同一 block** harvest V 并清同一槽（prefill 压力路径必须成对）
- query replay：follower 同一 `[p0,p1)` `seq_rm`；replay ubatch 之后靠已有 `process()` 用新 `h_nextn` 补 MTP 后缀
- pin：decode/verify 不得 recency 挤召回块（主干已有 `retrieval_pinned_`，follower 同 pin）
- 未 accept 的 MTP cell 不进 raw-K
- `gen_reserve` 不够则硬失败

**退出：**

- 0.8B（5050）retrieval 256 bt=32 + `n_max=2`：ranker 仍含 needle 块；关思考后生成含 `BLUEBIRD-42`（与现 needle 同构）
- recency 256 对照仍不召回
- TRACE：retrieval 后 MTP occupy 与主干 selected 一致；`query_replay` 后 MTP prefix 非空
- 4B / 27B 抽检：ranker 集合正确；生成 miss 记 GO/NO-GO，不自动停
- reject 不丢 pin、不把草稿写进 raw

**落地（2026-09-05，5050 / 0.8B-MTP Q8_0）：** `scripts/mtp_canary.py` PASS。identity 无 MTP 仍过；recency 256 不召回；retrieval 256 bt=32 `n_max=2` `--no-think`：needle block 13 在 selected，`mtp_selected` 同套，`mtp_follow n_gpu=8 n_writeback=8 n_no_raw=0`，`mtp_after_query_replay seq_pos=[0,847]`，生成含 BLUEBIRD-42。GDN restore 仍命中。

**27B 抽检（2026-09-05，5090 / UD-Q4_K_M，`--no-think`，`n_max=2`）：** recency 256 说「no secret code」；retrieval 256 含 needle block 13，`mtp_selected` 同套，`mtp_follow n_no_raw=0`，`mtp_pool cells=512 n_ctx=2048 bytes=2097152`（2 MiB），生成 **BLUEBIRD-42**。日志 `logs/p7_2_27b.log`。4B 未跑。

**27B 速度（5090，`n_predict=64`，decode=`KVMEM_GEN_WALL`，不当失败）：**

- 改 `n_rs_seq` 前：开 MTP ~19–22 tok/s（每步 host GDN ~150 MiB）。
- 方案 1（`n_rs_seq = n_max` GPU 平面，verify `ckpt_bytes=0`）：off ~40–45 tok/s；**off+MTP ~50–56 tok/s（~1.2×）**；retr+MTP ~49–52。表：`logs/p7_nrs_27b_speed.log`。

实现要点：

- `apply_retrieval` 后 `harvest_resident_v`（只拷 MTP 上仍占着的 cell）再 `follow_retrieval`（`seq_rm` 全条带，按主干 slot 写回 raw-K）
- 主干 `stage_out` 成对 `on_stage_out`
- query replay：draft 同一 `[p0,p1)` `seq_rm`，靠已有 `process()` 补后缀
- **`ctx_other` 只给工厂找 target，不等于 Gemma4 共享 KV。** `is_mem_shared` 还要求 `llama_get_memory(dft)==llama_get_memory(tgt)`，否则 `process()` 会跳过 prefill catch-up，MTP raw 为空
- 建图 / harvest 按 `LLM_GRAPH_TYPE_DECODER_MTP` 分边，互不清 pending
- pin 后 `want_prefill_capture=false`：verify n>1 不进 raw

---

### PR P7-3 — 长上下文显存封顶（MTP 也封）

**标题：** `canary: MTP KV bytes follow budget not n_ctx`

**依赖：** P7-2

**改动：**

- `scripts/` 显存金丝雀：固定 budget，扫 8K / 32K / 64K（27B 在 5090 能跑到哪算哪）
- 断言：主干 KV 字节 + MTP KV 字节随 **budget** 变，不随 T 线性变
- 记录 27B decode tok/s：无 MTP vs `n_max=2`（只记录）

**退出：**

- 27B（5090）在至少一档长 ctx（建议 32K 或 64K，显存够再 128K）：主池 + MTP 池均远小于「一层 × T」
- 相对 P7-1 的 16k 对照，更长 T 时 MTP 字节几乎不变
- 速度表写入 `logs/`，不当失败

**落地（2026-09-05，5090 / 27B UD-Q4_K_M，budget=256 gen_reserve=256 pool=512）：** `scripts/mtp_vram_canary.py` PASS。T=8192/16384/32768/65536：`kv_bytes=33554432`（32 MiB）、`mtp_bytes=2097152`（2 MiB）全程不变；MTP cells=512 ≠ n_ctx。64k 方案 A 约 258 MiB。Decode（记录）：8k off 37.8 vs retr+MTP 61.6；64k off 32.2 vs retr+MTP 49.3。日志 `logs/p7_3_vram.log`。128k 未跑。

---

## 5. 补丁与文件

```text
patches/0001  工厂：MTP → llama_memory_kvmem_mtp；CMake 加 mtp.cpp
patches/0002  graph_mtp pre-RoPE K capture（数行）
patches/0003  一般不动

src/adapter/llama-memory-kvmem-mtp.h
src/adapter/llama-memory-kvmem-mtp.cpp
src/adapter/llama-memory-kvmem.cpp     pin 后关 capture；stage_out 通知 follower
tools/llama-kvmem-cli.cpp              common_speculative
tools/llama-kvmem-server.cpp           同上
```

对外接口：`--kvmem` + `--spec-type draft-mtp --spec-draft-n-max N`。不要发明 qw3 风格的 `--mtp-chain`。

---

## 6. 测试矩阵

| 用例 | 阶段 | 模型 / 卡 | 期望 |
|---|---|---|---|
| 无 KVMem + draft-mtp | P7-0 | 0.8B / 5050 | 能生成 |
| KVMem 短 ctx + MTP，无 retrieval | P7-0 | 0.8B / 5050 | greedy 稳定；GDN reject 可 restore |
| MTP `n_kv` == 主池 | P7-1 | 0.8B / 5050 | TRACE |
| 27B MTP KV ≪ 一层×T | P7-1 | 27B / 5090 | 16k 为池子量级 |
| retrieval + MTP needle | P7-2 | 0.8B / 5050 | 关思考后 BLUEBIRD-42 |
| 长 ctx 两池都不随 T 涨 | P7-3 | 27B / 5090 | 字节随 budget |

`--no-think`：27B 默认会进 `<think>`，needle 的 `-n` 不够会假 miss（已在 27B 复测中证实）。

**ISTA IQ3_S 不在此矩阵里。** 2026-09-05 测过：`n_max=3` 接受率 ~28–41%，16k decode 无加速（off+mtp 40.6 vs off 47.3 tok/s，`logs/ista_iq3_16k_speed.log`）。产品默认 `--spec-type none`。以后不要对 IQ3_S 开 `--spec-type draft-mtp`，也不要把它加进 canary。27B MTP 车仍是 Unsloth `UD-Q4_K_M`。政策正文：[architecture.md](architecture.md#ista-iq3_s-mtp-off-do-not-test)。

---

## 7. 风险

| 风险 | 对策 |
|---|---|
| CLI 不走 `process()`，MTP 槽空 | P7-0 强制 `common_speculative` |
| verify n>1 打开主干 capture | pin 后关门 |
| GDN 无法 seq_rm 回滚 | P7-0 snapshot/restore |
| 独立算 draft 池，退回方案 A | 池大小只从 target 抄 |
| stage_out 只清主干 | P7-2 成对 |
| 补丁膨胀 | 不改投机状态机 / FA；0002 只加 graph_mtp 数行。`is_mem_shared` 一行：`ctx_other` 且同一 `llama_memory` 才算 Gemma4 共享 KV |
| 长 ctx MTP FA 仍慢 | B 把 `n_kv` 封在 budget；若仍慢再查 mask/holes |

---

## 8. 粗估

| PR | 周期 | 里程碑 |
|---|---|---|
| P7-0 | 3–7 天 | 投机在 KVMem 短 ctx 上成立 |
| P7-1 | 1–2 周 | MTP 显存封顶（方案 B） |
| P7-2 | 1–2 周 | retrieval + MTP 召回 |
| P7-3 | 2–4 天 | 27B 长 ctx 字节金丝雀 |

已打本地 tag **`v0.5.0`**，writeup 在 [milestones/v0.5.0.md](milestones/v0.5.0.md)。未 push。
