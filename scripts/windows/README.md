# Windows x64 CUDA preview — v0.16.0-rc2

## Download and run

Download the **windows-x86_64-cuda13.2.86-sm120a.zip** runtime from
[v0.16.0-rc2](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.16.0-rc2).
It contains the server, CLI, matching CUDA DLLs and browser UI. It does **not**
contain model weights or the quantization tool. The separate **quantizer** ZIP
is optional; it is unnecessary when your GGUF files are already prepared.

Requirements: Windows x64, a compatible NVIDIA driver, Microsoft Visual C++ x64
runtime, and an AVX2/FMA/F16C/BMI2 CPU. The CUDA target is `sm_120a`; the tested
GPU is RTX 5060 Ti 16 GiB. CUDA Toolkit,
Visual Studio and Node.js are not needed to run the package. Tested driver: 610.62.

### Text-only quick start (no model conversion)

Download the ready-made
[Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/blob/main/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf)
from ISTA-DASLab. Use the `-mtp` file. Open PowerShell in the extracted runtime
directory, select your GPU index, and use your actual model path:

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
.\bin\llama-kvmem-server.exe -m 'D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf' `
  --host 127.0.0.1 --port 18200 -c 262144 -n 16384 `
  --kvmem-budget 36864 --kvmem-gen-reserve 16384 `
  -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3 `
  --enable-thinking --reasoning-budget 4096
```

Open http://127.0.0.1:18200/ after loading. Keep the terminal open; Ctrl+C stops
the server. The command uses default block size 128.

### Ready-made vision projector (no local quantization)

Download **`mmproj-Qwen3.8-27B-Q5_K-MIX.gguf`** from
[HermiHg/Qwen3.8-27B-mmproj-Q5_K-MIX-GGUF](https://huggingface.co/HermiHg/Qwen3.8-27B-mmproj-Q5_K-MIX-GGUF).
Use this exact path with `-Mmproj` in the recipe below. No quantization tool is
needed for the IQ3 main model or this projector. It is a ready-made mixed-precision vision projector.

The historical performance tables used Q8_0 (IQ3) and BF16 (IQ4) projectors;
those numbers are not measurements of this Q5_K-MIX projector. IQ4 still uses
the separately prepared MTP-Q4_0 main model; this release does not provide a
download link for that derived main-model file. It is retained only as an optional test configuration, not a primary download
recommendation.

Target: Windows x64 with an NVIDIA CUDA GPU. NVMe KV storage is disabled;
CPU memory KV, retrieval, MTP/ReplaySSM and vision remain in the build.
The initial GPU target is `120a-real`, matching the current Linux package.
Other architectures require an explicit build target and separate testing.

See the validation record below for the completed checks.

## Build

Install Visual Studio 2022 C++ Build Tools (MSVC x64, Windows SDK and the CMake
tools including Ninja), Git for Windows and **CUDA Toolkit 13.2 Update 2
(nvcc 13.2.86) or newer**. Version 13.2.86 is the tested Windows baseline;
newer toolchains need their own correctness validation. Use an independent
checkout on a Windows local drive, with its pinned llama.cpp submodule initialized:

```powershell
git clone --recurse-submodules https://github.com/kvmem/kvmem-llama.cpp.git C:\src\kvmem
cd C:\src\kvmem
git checkout v0.16.0-rc2
git submodule update --init
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\build.ps1 `
  -CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
```

CUDA 13.2 Update 2 still installs under `v13.2`. Verify the actual compiler:

```powershell
& 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe' --version
# Expected for the validated baseline: V13.2.86
```

Do not use the original CUDA 13.2.51 build for the 27B IQ3 recipe: it produced
garbage output on RTX 5060 Ti, including with KVMem and MTP disabled. Upgrading
the installed Toolkit or swapping DLLs does not repair the old executable.
Rebuild into a new directory (for example, pass `-BuildDir build-win-cuda13286`),
and pass the same `-BuildDir` to the launchers. Use the matching CUDA libraries
when packaging. The `nvidia-smi` CUDA version is not the nvcc version.

The script initializes the installed x64 MSVC environment, applies the maintained
patch to the pinned llama.cpp source (or verifies it is already applied), builds
the server/CLI/quantizer and runs model-free tests. Use a clean submodule: the
script does not reset or discard local changes. Four build jobs are used by
default; `-Jobs` changes this. `-CudaArchitectures` selects another GPU target.

`-HostOnly -BuildDir build-win-host` builds/tests the memory KV library without
CUDA. The full KVMem server requires CUDA; disabling the ggml CUDA backend does
not produce a CPU-only KVMem server. `-BuildOnly` explicitly omits runtime tests
and must not be reported as a tested build.

Linux retains NVMe support by default. To test the memory-only configuration
there, pass `-DKVMEM_ENABLE_NVME=OFF`. Windows defaults to OFF and rejects ON.
An NVMe-disabled server rejects `--kvmem-raw-k-nvme` and a nonzero
`--kvmem-nvme-gb` before loading the model.

## Run

A packaged build needs a compatible NVIDIA driver and the Microsoft Visual C++
x64 runtime. CUDA DLLs are bundled; the Toolkit is only needed to compile.
Microsoft runtime download: https://aka.ms/vs/17/release/vc_redist.x64.exe

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-iq3.ps1 `
  -Model 'D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf' `
  -Mmproj 'D:\models\mmproj-Qwen3.8-27B-Q5_K-MIX.gguf' -Gpu 0 `
  -BlockTokens 32 -ReasoningEffort medium

```

These are foreground launchers: keep the terminal open and use Ctrl+C to stop.
They do not terminate existing listeners or offer background restart/stop.
`-DryRun` prints resolved arguments without starting a model. `MODEL`, `MMPROJ`,
`BUILD_DIR` and `CUDA_VISIBLE_DEVICES` can also provide paths/GPU selection.
Without an explicit GPU, a sole GPU or a unique RTX 5060 Ti is selected;
other multi-GPU configurations require `-Gpu`.

IQ3 is the primary recommendation. IQ4 is an optional experimental comparison
for testers with prepared files.

| Default | IQ3 (recommended) | IQ4 (experimental) |
|---|---|---|
| Context | 262144 | 262144 |
| Retrieval / generation reserve | 36864 / 16384 | 32768 / 12288 |
| Main / MTP KV | Q8 / F16 | Q5 / F16 |
| MTP | 3, ReplaySSM | 3, ReplaySSM |
| Vision placement | GPU | CPU |
| Block size | 128 | 128 |
| Thinking budget | 4096 | 4096 |

Other switches include `-Port`, `-Mtp`, `-VisionDevice cpu|gpu`,
`-ReasoningBudget`, `-ChatTemplateFile`, `-ChatTemplateKwargs`, `-UiDir`, `-NoUi`.
Reasoning effort follows the model template unless explicitly set. Sampling uses
the same server defaults as Linux and remains configurable per API request.


When invoking `bin/llama-kvmem-server.exe` or `bin/llama-kvmem-cli.exe` directly,
llama.cpp-style `-ctk TYPE -ctv TYPE` (or `--cache-type-k` / `--cache-type-v`)
is supported. **Quantized K and V must use the same cache type**: use
`-ctk q8_0 -ctv q8_0`, for example. Mixed pairs such as `q8_0/q4_0` or
`q8_0/f16` are rejected before model loading. `--kv-dtype TYPE` sets both
together; the recipe launchers already select matching K/V types.

## UI

Copy the existing built page into `build-win/share/kvmem/ui/`, or pass `-UiDir`.
The WSL/Linux-built static page is reusable unchanged. Open
http://127.0.0.1:18200/ after model loading. No Node.js is needed at runtime.
Windows locates bundled assets relative to the executable, not the working directory.

## Package

Run `scripts/windows/package.ps1` in x64 Developer PowerShell. Like the Linux
packager, it requires a matching source manifest (`files`: relative path to
SHA-256 mapping) and source archive, supplied by the release preparation step:

```powershell
.\scripts\windows\package.ps1 -SourceDir C:\src\kvmem -BuildDir C:\src\kvmem\build-win `
  -SourceManifest C:\release\source-manifest.json -SourceArchive C:\release\source.zip `
  -OutputDir C:\release\kvmem-windows-x64-cuda13 `
  -CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2' `
  -UiDir C:\src\kvmem\build-win\share\kvmem\ui
```

The packager defaults to `-Component Runtime` (server/CLI, no quantizer).
Run it separately with `-Component Quantizer` and a different `-OutputDir`
for the optional self-contained quantizer ZIP.

The packager verifies the manifest, scans PE dependencies recursively, bundles
application/CUDA DLLs and their license texts, and writes build metadata and
SHA-256 checksums before producing a ZIP. The NVIDIA driver and Microsoft runtime
are external prerequisites. Distribute the exact matching source archive beside
the binary ZIP. Packaging does not run model tests or mark the package GPU-verified.

## Validation

The rc2 Windows server/CLI passed eight model-free tests, 52 KV argument cases,
and four IQ3 short-text checks with `-ctk/-ctv`, KVMem Q8 and MTP3/ReplaySSM
at 8K context on RTX 5060 Ti. Launcher and split-package checks also passed.
See `VALIDATION.json` for the tested binary hashes.
