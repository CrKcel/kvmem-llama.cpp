# Bonsai 2 experimental Windows runtime

Based on KVMem rc3 with Prism llama.cpp. This package is for **Windows x64,
NVIDIA RTX 30/40/50 series** (SM86/89/120a), using CUDA 12.9.86.
Actual inference validation uses RTX 5050 Laptop 8 GB; RTX 30/40 are compiled
targets, not hardware-tested. Requires an AVX2/FMA/F16C/BMI2 CPU, a compatible
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
默认使用 Q8 KV、32K 上下文、2K 检索预算和 1K 生成预留；这个低显存默认配置
适合先验证启动，不能保证 32K 历史完整召回（已有三项召回测试只通过一项）。

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

- Text-only Bonsai PTQ1 is validated. MTP and DSpark are disabled; do not use
  rc3 IQ3/IQ4 MTP launch recipes with this build.
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
