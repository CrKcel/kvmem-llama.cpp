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
3. Open PowerShell in the extracted directory and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-bonsai.ps1 -Model 'D:\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf' -Gpu 0
```

Open **http://127.0.0.1:18202/** after loading. Keep the terminal open;
Ctrl+C stops the server. The full chat UI is bundled and enabled automatically.
Change `-Gpu` if the NVIDIA device selected is not the desired card.
The model defaults to `%LOCALAPPDATA%\KVMem\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf`
when `-Model` is omitted. Weights are not included in this ZIP.

解压后按上面命令启动，然后打开浏览器即可聊天。无需编译或安装 CUDA Toolkit。
默认使用 Q8 KV、128K 上下文、24K 检索预算和 10K 生成预留，开启思考，
思考预算为 4096 token（可用 `-ReasoningBudget` 覆盖）。思考 token 包含在总生成上限内。
128K 是启动配置，尚未完成 128K 实际输入验证；下文结果来自此前的 64K 测试。

Defaults: context 131072, retrieval budget 24576, generation reserve 10240,
thinking enabled with a 4096-token reasoning budget. The 128K default has not
been validated with an actual 128K input.

## Tested 64K configuration / 已验证的 64K 配置

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-bonsai.ps1 -Model 'D:\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf' -Gpu 0 -Context 65536 -Budget 24576 -Reserve 10240
```

Earlier single-SM120a build on RTX 5050 Laptop: actual prompt 64,653 tokens,
129.36 tokens/s prefill, 9.40 tokens/s decode (256 output tokens), three of
three planted identifiers recalled. Whole-device VRAM peak 7,454 MiB out of
8,151 MiB. The 10K generation reserve was allocated, but 10K output was not
tested. Other GPU/display workloads can change available VRAM.
See the release validation for checks repeated on the packaged multiarch binary.

## Experimental scope

This version includes optimized PTQ1 CUDA decode and optional community r3 MTP.
MTP is off by default. On the measured short decode workload, the new kernels
improved no-MTP speed by 19-54%; draft=1 added another 15-18% on two RTX 50 GPUs.
Prefill was roughly unchanged. See the bundled `docs/milestones/v0.16.0-rc3-prism.2.md`
and `docs/bonsai-kernel-comparison.md` for conditions and validation boundaries.

### Optional MTP / 可选 MTP

Prepare a merged r3 GGUF following the bundled `docs/bonsai-mtp-validation.md`.
The original GGUF has no MTP head; `-Mtp` does not download or convert weights.
For an 8GB GPU, start with the tested smaller pool:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-bonsai.ps1 -Model 'D:\models\Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q8_0.gguf' -Gpu 0 -Mtp -DraftTokens 1 -Context 8192 -Budget 2048 -Reserve 1024 -ReasoningBudget 256
```

The merged model adds about 430 MiB on disk; draft=1 added about 0.6 GiB VRAM
in the small-pool tests. Enabling MTP does not shrink the default 24K + 10K pool.
That default pool, long actual inputs (32K+), and 10K output have not been tested
with MTP. draft=2 was slower than draft=1 in both tested GPUs.

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
