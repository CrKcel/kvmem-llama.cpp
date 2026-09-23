# KVMem + Bonsai 2 27B：把 160K 上下文塞进 10G 显存

> 基于 [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp)，
> 使用 [PrismML 的 llama.cpp](https://github.com/CrKcel/llama.cpp) 分支（低比特格式 + 三元量化运行时）。
> 本仓库在上面加了一个东西：**思维链撞到输出上限时不再截断，自动接着想下一轮，直到模型自己收尾。**

| | |
|---|---|
| 模型 | [Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)（PQ2_0 约 6.7 GB） |
| 显存 | 最低 8 GB（3080 10G 实测 8.8 GB 稳定跑 160K） |
| 内存 | Linux 16 GB / Windows 24 GB 起（历史存主机内存） |
| 上下文 | 逻辑工作区 160K，GPU 只留 `budget + reserve` 的滑动窗口 |
| 速度 | 3080 10G：prefill 420+ t/s，decode 59 t/s（q8_0 池 36864） |

---

## 一、三条命令跑起来（人类版）

```bash
git clone --recurse-submodules https://github.com/xzk-frist/kvmem-llama.cpp.git
cd kvmem-llama.cpp && ./scripts/apply-patches.sh
CMAKE_CUDA_ARCHITECTURES=86-real ./scripts/build-cuda.sh     # 改成本机架构
```

产物在 `build/bin/llama-kvmem-server`。启动参数见第五节。

> 不想编译就用[预编译包](https://github.com/kvmem/kvmem-llama.cpp/releases)（Windows rc3 有 CUDA 13.2 / 12.9 两个版本）。

---

## 二、用 AI Agent 一键编译部署

把下面整段丢给编码 agent（Claude Code / Codex / Cursor / CodeBuddy 都行），它就能自己走完全流程。

### 复制这段给它

```text
请在这台机器上编译并部署 KVMem + Bonsai 2 27B，目标 160K 上下文。

按 README 第四到八节执行，并遵守以下约束：

1. 先探测环境再动手：OS/架构、GPU 型号与显存、nvcc 版本、cmake/ninja、编译器、可用内存、磁盘。
   把探测结果列成表给我确认后再开始编译。
2. CUDA 架构参数必须按实际 GPU 设定，不要照抄默认值：
   RTX 30xx = 86-real / 40xx = 89-real / 50xx = 120a-real。
   CUDA Toolkit 需要 13.2 Update 2（nvcc 13.2.86）或更新。
3. 克隆必须带 --recurse-submodules，然后跑 scripts/apply-patches.sh（可重复执行）。
4. 编译前把 nvcc --version 的真实输出贴给我；编译用后台任务，超过 10 分钟要报告进度。
5. 产物是 build/bin/llama-kvmem-server。启动后必须真发一次请求验证，不要只看健康检查。
6. 显存预算按第六节的换算表算。记住：空载显存会骗人，判断余量必须真发请求。
7. 出现 KVMEM stagein q scratch: out of memory 时，先降 --kvmem-budget，再把 -ub 降到 64。
8. 每一步做了什么、观察到什么，用一句话汇报；失败时把原始报错整段贴出来。
```

### Agent 应该执行的 8 步

| 步骤 | 做什么 | 通过判据 |
|---|---|---|
| 1 | 探测环境：GPU / CUDA / 工具链 / 内存 | `nvidia-smi` 列出 GPU；`nvcc --version` ≥ 13.2.86 |
| 2 | `git clone --recurse-submodules` + `apply-patches.sh` | 输出 `applied current KVMem patch` 或 `already applied` |
| 3 | 按 GPU 定 `CMAKE_CUDA_ARCHITECTURES` | 与 `nvidia-smi` 型号对应 |
| 4 | `scripts/build-cuda.sh` | 打印 `binaries under <build>/bin`，且该 exe 存在 |
| 5 | 下载模型到 `models/`（已 gitignore） | 文件约 6.7 GB（PQ2_0） |
| 6 | 算显存预算：`budget + reserve` ≤ 安全上限 | 见第六节 |
| 7 | 启动服务 `--port 18200` | 日志出现 `listening on http://127.0.0.1:18200` |
| 8 | **发一次真实请求** | `/v1/chat/completions` 返回 200 且 `usage.completion_tokens > 0` |

### 三条铁律（最容易踩）

1. **别用默认 CUDA 架构** —— `build-cuda.sh` 默认 `120a-real`（作者的 5060 Ti）。架构不对要么编译失败，要么运行时内核报错。
2. **空载显存不可信** —— 暂存缓冲只在负载时分配。实测 `q8_0` 池 49152 空载 9620 MiB 看着还行，**一发请求就 CUDA OOM**。
3. **`reserve` 必须大于 `act-round-tokens`** —— 单次生成不能超过 `--kvmem-gen-reserve`，超了整轮作废。

### 故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `KVMEM stagein q scratch: out of memory` → `CUDA error` | 显存超了 | 先降 `--kvmem-budget`，再 `-ub 64` |
| `'token_embd.weight' is read without the inverse transform` | 缺 0005 补丁 | 重跑 `apply-patches.sh` |
| `non-consecutive token position` | 续思缺 `--act-prefill-mode auto` | 补上该参数 |
| `GGML_ASSERT(gdn->op == GGML_OP_GATED_DELTA_NET)` | MTP 用了默认 `replay` | 必须显式 `--kvmem-mtp-state snapshots` |
| 思考到固定 token 数就「自己结束」 | 请求里 `enable_thinking` 为假 | 见 `docs/auto-continue-thinking.md` |
| 输出乱码 | nvcc 版本问题（13.2.51 已知会让 IQ3 出错） | 换 13.2.86 并**新建** build 目录重编 |

---

## 三、模型与下载

| 文件 | 显存 | 说明 |
|---|---|---|
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 8G 可用 | 最低门槛 |
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 12G 推荐 | 更快，本仓库实测用它 |
| `Bonsai-2-27B-PQ2_0-MTP.gguf` | 多占 0.35 GB | 带 MTP 草稿头，仅用 `--spec-type draft-mtp` 时需要 |

从 <https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf> 下载，放进 `models/`（已在 `.gitignore`）。

---

## 四、编译（各种配置）

```bash
CMAKE_CUDA_ARCHITECTURES=86-real ./scripts/build-cuda.sh
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `120a-real` | **必改**：86 / 89 / 120a |
| `CMAKE_CUDA_COMPILER` | `nvcc` | CUDA 不在 PATH 时给绝对路径 |
| `CMAKE_BUILD_TYPE` | `Release` | |
| `BUILD_DIR` | `<root>/build` | 换 Toolkit 后**必须换新目录** |
| `NPROC` | `nproc` | 并行度 |

**纯 CPU / 其他后端**（不需要 CUDA）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
      -DKVMEM_BUILD_LLAMA=ON -DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT="$PWD"
cmake --build build -j
```

Metal / ROCm / Vulkan / SYCL：关掉 `GGML_CUDA`，打开对应 `GGML_<BACKEND>`，详见 `llama.cpp/docs/build.md`。

`build-cuda.sh` 会开 `GGML_CUDA_FA_ALL_QUANTS=ON`（`--kv-dtype q5_0` 需要）；只用 q4_0/q8_0 可以不开。

Windows：`powershell -File scripts/windows/build.ps1`，见 `scripts/windows/README.md`。

---

## 五、启动与服务

```bash
build/bin/llama-kvmem-server \
  -m models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --host 127.0.0.1 --port 18200 --alias bonsai-2-27b --no-ui \
  -ngl 99 -fa on -np 1 -t 6 -tb 6 -b 512 -ub 128 \
  -c 163840 \
  -ctk q8_0 -ctv q8_0 \
  --kvmem-budget 24576 --kvmem-gen-reserve 12288 --kvmem-block-tokens 64 \
  --kvmem-query-policy user \
  --auto-continue-thinking --act-round-tokens 6000 --act-max-rounds 8 \
  --act-total-tokens 36000 --act-prefill-mode auto \
  --enable-thinking --reasoning-budget -1 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

接口与 OpenAI 兼容：`POST http://127.0.0.1:18200/v1/chat/completions`。

### 关键参数

| 参数 | 作用 | 注意 |
|---|---|---|
| `-c` | 逻辑工作区，可远大于显存 | 160K 只用约 2 GB 显存 |
| `--kvmem-budget` | GPU 上常驻的检索窗口 | 与 `reserve` 之和 = 池 |
| `--kvmem-gen-reserve` | **单次生成硬上限** | 必须 > `--act-round-tokens` |
| `--kvmem-block-tokens` | 检索块粒度 | 64 与 128 实测无差别 |
| `--kvmem-query-policy user` | 按当前提问挑历史块 | 续思必须配 `auto` |
| `--act-round-tokens` | 每轮输出预算 | 撞上限就触发续思 |
| `--reasoning-budget -1` | 思考不限 | **不能给正数**，否则续思断链 |
| `-ctk` / `-ctv` | KV 精度 | `q4_0` / `q5_0` / `q8_0` / `f16` |

---

## 六、显存预算（3080 10G 实测）

每 token 的显存成本：

| KV 类型 | 成本（含草稿 KV） |
|---|---|
| `q4_0` | ≈ 22 KiB/token |
| `q8_0` | ≈ 32 KiB/token |

```
显存 ≈ 模型权重 + 池 × 每 token 成本
池   = --kvmem-budget + --kvmem-gen-reserve
```

实测锚点：

| 池 | KV | MTP | 空载 | 负载 |
|---|---|---|---|---|
| 32768 | q4_0 | 关 | 8428 MiB | 正常 |
| 36864 | q8_0 | 关 | 8754 MiB | 正常 |
| 36864 | q8_0 | 开 | 8960 MiB | 9118 MiB ✅ |
| 49152 | q8_0 | 开 | 9620 MiB | ❌ **CUDA OOM** |

**`q8_0` 大池与 MTP 在 10G 卡上不可兼得。** 要 q8_0 大窗口就 `--spec-type none`（省约 866 MiB）。

---

## 七、续思机制（auto-continue-thinking）

一轮撞上限时：把该轮已生成内容 `commit_cached` 进上下文 → 注入 cue → 推进 prompt 视图 → 重新 prefill（LCP 命中，只多解 1 个 token）→ 开下一轮，直到模型自己输出思考结束标记或 EOS。

实测对照（同一请求）：

| | 续思关 | 续思开 |
|---|---|---|
| `finish_reason` | `length` | `stop` |
| `completion_tokens` | 1200（被截断） | 31038（6 轮） |
| 思维链 | 半句断掉 | 完整 64507 字符 |
| 正文 | 空 | 完整 54402 字符 |

跨轮连续性判据（服务端日志）：`prefix_hit_rows` 每轮精确递进
`143 → 6143 → 12144 → 18145 → 24146 → 30147`。

详细说明见 **[docs/auto-continue-thinking.md](docs/auto-continue-thinking.md)**。

---

## 八、验证部署是否成功

```bash
curl -s http://127.0.0.1:18200/health
```

然后**必须发一次真实请求**（健康检查不能证明显存够用）：

```bash
curl -s -X POST http://127.0.0.1:18200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bonsai-2-27b","messages":[{"role":"user","content":"用一句话说明什么是 KV 缓存"}],
       "max_tokens":64,"stream":false}'
```

返回 200 且 `usage.completion_tokens > 0` 才算真跑起来。

验证续思：同一请求分别带 `"auto_continue_thinking": false` 和 `true`，后者应返回远大于
`max_tokens` 的 token 数且 `finish_reason` 为 `stop`。

---

## 九、仓库结构与补丁体系

```
kvmem/            主机侧 KV 存储与检索
src/adapter/      通过 llama.cpp memory 接口接入
tools/            llama-kvmem-server / cli
scripts/          apply-patches / build-cuda / 启动脚本
patches/          针对 llama.cpp pin 的补丁
docs/             架构、设计、基准
llama.cpp/        子模块（只存 pin；克隆后打补丁）
models/           本地 GGUF（已 gitignore）
```

**不要提交脏的 `llama.cpp` 工作树。** 子模块指针就是 pin，所有本地改动走 `patches/`，
由 `scripts/apply-patches.sh` 重放。补丁清单见 [patches/README.md](patches/README.md)。

---

## 十、许可与引用

llama.cpp 版权归上游；本 KVMem 移植沿用其许可。模型权重单独分发，条款可能不同。

论文：[KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU](https://arxiv.org/abs/2609.04852)

```bibtex
@misc{chai2026kvmem,
  title         = {{KVMem}: Virtualizing Million-Token Agent Workspaces on a Consumer {GPU}},
  author        = {Di Chai and Leye Wang and Zeshen Su and Zhiguo Xia and Zhihang Yu},
  year          = {2026},
  eprint        = {2609.04852},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.04852}
}
```

## 致谢

感谢 **kvmem** 团队与 **PrismML**，以及原项目所有贡献者。
Bilibili 的 **melis** 与 **redsnow23** 的测试反馈同样致谢。


## 十一、配套文档

- **[docs/agent-build-and-deploy.md](docs/agent-build-and-deploy.md)** —— 给 agent 用的编译部署 runbook（环境探测、架构选择、显存预算、验证判据、故障索引）。第二节的 agent 提示词就是让它去读这份文件。
- **[docs/auto-continue-thinking.md](docs/auto-continue-thinking.md)** —— 续思机制的参数、约束与诊断行。
- **[patches/README.md](patches/README.md)** —— 补丁清单与重放方式。
- 上游的完整 CLI 兼容表、环境变量表与混合 KV 精度验证矩阵见
  [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) 的 README。
