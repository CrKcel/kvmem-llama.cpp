# Bonsai 2 experimental Windows runtime — rc3-prism.2

Based on KVMem rc3 with Prism llama.cpp. This package is for **Windows x64,
NVIDIA RTX 30/40/50 series** (SM86/89/120a), using CUDA 12.9.86.
Kernel and MTP inference validation uses RTX 5050 Laptop 8 GB and RTX 5060 Ti
16 GB. RTX 30/40 are compiled targets, not hardware-tested. Requires an AVX2/FMA/F16C/BMI2 CPU, a compatible
NVIDIA driver and the Microsoft Visual C++ x64 Redistributable.
CUDA Toolkit, Visual Studio, Python and Node.js are not needed to run it.

## Quick start / 快速使用

1. Extract the entire ZIP, keeping `bin`, `scripts` and `share` together.
2. Download `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5.95 GB) from
   [ModelScope](https://modelscope.cn/models/prism-ml/Ternary-Bonsai-2-27B-gguf/files)
   or [Hugging Face](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/main).
3. The launcher defaults to **MTP off** and uses the original PTQ1 model:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-bonsai.ps1 -Model 'D:\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf' -Gpu 0
```

Open **http://127.0.0.1:18202/** after loading. Keep the terminal open;
Ctrl+C stops the server. The full chat UI is bundled and enabled automatically.
Change `-Gpu` if the NVIDIA device selected is not the desired card.
The model defaults to `%LOCALAPPDATA%\KVMem\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf`
when `-Model` is omitted. Weights are not included in this ZIP.
To enable MTP, pass `-Mtp` with a merged r3 model prepared using
`docs/bonsai-mtp-validation.md`. With `-Mtp` and no `-Model`, the launcher selects
`Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q8_0.gguf` in the same model directory.
`-NoMtp` and `-Mtp:$false` remain supported. No automatic fallback occurs when
the selected model is missing.

解压后按上面命令启动，然后打开浏览器即可聊天。无需编译或安装 CUDA Toolkit。
默认关闭 MTP，使用原版 PTQ1 模型、Q8 KV、128K 上下文、24K 检索预算和 10K 生成预留，开启思考，
思考预算为 4096 token（可用 `-ReasoningBudget` 覆盖）。思考 token 包含在总生成上限内。
128K 配置已在 5050 上完成 130,103 token 实际输入测试，无 OOM，但检索仅命中 1/3；不能视为长文质量验证通过。

手动加 `-Mtp` 开启时，默认 draft=1、草稿 K/V F16；代码场景可加 `-DraftTokens 2`。

Defaults: MTP off, target K/V Q8_0, context 131072, retrieval budget 24576, generation reserve 10240,
thinking enabled with a 4096-token reasoning budget. The 128K MTP1 run on RTX 5050
processed 130,103 input tokens without OOM, but recalled only 1/3 planted codes.
Prefill was 60.95 token/s (35m35s), sustained decode 4.63 token/s, peak VRAM 7845 MiB.
Windows shared GPU memory peaked at 706 MiB; that counter alone does not prove paging.
Those historical 128K measurements used Q8 draft KV. The F16 default was selected
after a separate RTX 5060 Ti MTP1 test at 4K/16K input: sampled total VRAM fell
from 7834 to 7766 MiB, with no consistent decode speed improvement.
Use `-DraftKvType q8_0` to override the launcher default; when running the binary
directly, use `--spec-kv-dtype q8_0`. `-KvType` controls only the target KV.
This is a failed recall test, not a 128K quality guarantee. See
`docs/bonsai-128k-mtp1-validation.md`. Thinking was disabled for benchmark requests;
only 256 output tokens were tested, not 10K.

## Tested 64K configuration / 已验证的 64K 配置

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-bonsai.ps1 -NoMtp -Model 'D:\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf' -Gpu 0 -Context 65536 -Budget 24576 -Reserve 10240
```

Earlier single-SM120a build on RTX 5050 Laptop: actual prompt 64,653 tokens,
129.36 tokens/s prefill, 9.40 tokens/s decode (256 output tokens), three of
three planted identifiers recalled. Whole-device VRAM peak 7,454 MiB out of
8,151 MiB. The 10K generation reserve was allocated, but 10K output was not
tested. Other GPU/display workloads can change available VRAM.
See the release validation for checks repeated on the packaged multiarch binary.

## Experimental scope

This version includes optimized PTQ1 CUDA decode and optional community r3 MTP.
MTP is off by default; `-Mtp` enables draft=1 with F16 draft KV. On the measured short decode workload, the new kernels
improved no-MTP speed by 19-54%; draft=1 added another 15-18% on two RTX 50 GPUs.
Those measurements used an 8K context and 2K+1K pool; they are not the throughput
of the default larger pool at near 128K input.
Prefill was roughly unchanged. See the bundled `docs/milestones/v0.16.0-rc3-prism.2.md`
and `docs/bonsai-kernel-comparison.md` for conditions and validation boundaries.

### MTP model and smaller pool / MTP 模型与小池配置

Prepare a merged r3 GGUF following the bundled `docs/bonsai-mtp-validation.md`.
The original GGUF has no MTP head; the launcher does not download or convert weights.
Use `-NoMtp` to run the original file.
For an 8GB GPU, start with the tested smaller pool:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-bonsai.ps1 -Model 'D:\models\Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q8_0.gguf' -Gpu 0 -Mtp -DraftTokens 1 -Context 8192 -Budget 2048 -Reserve 1024 -ReasoningBudget 256
```

The merged model adds about 430 MiB on disk; draft=1 added about 0.6 GiB VRAM
in the small-pool tests. Enabling MTP does not shrink the default 24K + 10K pool.
The default pool has now been exercised at near 128K input, with the recall failure
and low memory margin described above. Continuous 10K output remains untested.
draft=2 was slower than draft=1 in both short-workload GPU comparisons.

Server logs, including prefill/decode timing, are forwarded to the launch terminal.

- Text-only Bonsai PTQ1 is validated; MTP uses a community-trained head, not an
  official Prism checkpoint. DSpark is not integrated. Do not use rc3 IQ3/IQ4
  MTP launch recipes with this build.
- No NVMe KV storage. CPU-memory KV spill and retrieval are enabled.
- Full UI does not imply backend tool execution or stream resumption support.
- This is a separate experiment, not a general replacement for the rc3 runtime.
- `SHA256SUMS`, `BUILD-INFO.json`, `VALIDATION.json` and a matching source ZIP
  accompany the release. Source includes the applied Prism integration patch.

## Rebuild

From the `kvmem-bonsai-llama.cpp` branch with its submodule initialized:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -BuildDir build-win-bonsai-release -CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9' -ExperimentalCuda129 -CudaArchitectures '86-real;89-real;120a-real' -Jobs 4
python scripts/build-webui.py --full-ui --output build-win-bonsai-release/share/kvmem/ui
```
