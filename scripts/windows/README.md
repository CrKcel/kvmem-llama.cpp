# KVMem Windows rc3 runtime / Windows rc3 运行包

## Quick start / 快速启动

Choose either the CUDA 13.2.86 or 12.9.86 runtime ZIP and extract the entire package.
Keep `bin`, `scripts` and `share` in their relative locations. Models are separate.
选择一个 CUDA 13.2.86 或 12.9.86 运行包并完整解压，保留目录结构；模型单独下载。

- [IQ3 main model / 主模型](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/blob/main/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf)
- [Vision projector / 视觉头](https://huggingface.co/HermiHg/Qwen3.8-27B-mmproj-Q5_K-MIX-GGUF/blob/main/mmproj-Qwen3.8-27B-Q5_K-MIX.gguf)

Open PowerShell in the extracted package directory. Check GPU indices with `nvidia-smi -L`.
Replace the example paths and GPU index in this command:
在解压目录打开 PowerShell，用 `nvidia-smi -L` 查看显卡编号，替换下面的模型路径和显卡编号：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-iq3.ps1 `
  -Model 'D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf' `
  -Mmproj 'D:\models\mmproj-Qwen3.8-27B-Q5_K-MIX.gguf' `
  -Gpu 0
```

Open **http://127.0.0.1:18200/** after loading. Keep the terminal open; **Ctrl+C** stops the service.
加载后打开上述地址，保持终端开启；按 **Ctrl+C** 停止服务。普通日志包含请求结束时的 prefill/decode 速度。

## Script settings / 脚本设置

The scripts are independent and directly call `bin/llama-kvmem-server.exe`.
They accept only `-Model` and `-Mmproj` (required), plus `-Gpu` (default `0`, index or UUID).
Executable and UI paths are relative to the script, not a fixed drive or username.
脚本独立运行，无需公共启动脚本。仅支持必填的 `-Model`、`-Mmproj`，以及默认 `0` 的 `-Gpu`。
程序和 UI 使用相对脚本位置的路径，不绑定盘符或用户名。

To change other settings, edit the script directly / 其他设置直接编辑脚本：

| Setting / 设置 | Edit / 修改位置 |
|---|---|
| Port / 端口 | `$Port = 18200` |
| Full UI (default) / 完整 UI（默认） | `$UiDir` ends in `share\kvmem\ui` |
| Lightweight UI / 轻量 UI | Change `$UiDir` to end in `share\kvmem\ui-lightweight` |
| Disable UI / 关闭 UI | Replace `--webui` with `--no-ui` |
| GPU vision / GPU 视觉头 | Replace `--no-mmproj-offload` with `--mmproj-offload` |
| Fixed API model name / 固定 API 模型名称 | Add `--alias your-model-name` to the server arguments |
| KV, context, MTP, thinking / 其他推理设置 | Edit the corresponding server argument in the script |

When adding argument lines, retain PowerShell's trailing backtick on every continued line.
添加参数行时，注意保留 PowerShell 续行反引号。不要把服务端参数当作脚本参数传入。

| Default / 默认值 | IQ3 | IQ4 |
|---|---:|---:|
| Context / 上下文 | 262144 | 262144 |
| KVMem budget / 工作集 | 36864 | 32768 |
| Generation reserve / 生成预留 | 16384 | 12288 |
| Main K/V | Q8/Q8 | Q5/Q5 |
| MTP draft / 草稿 | F16, MTP3/ReplaySSM | F16, MTP3/ReplaySSM |
| Vision / 视觉头 | CPU | CPU |
| Block tokens | 128 | 128 |
| Thinking budget | 4096 | 4096 |

IQ4 is an optional experiment requiring the separately prepared MTP-Q4_0 main model.
Use `start-iq4.ps1` with the same three parameters and your IQ4 model path.
IQ4 为备选测试，需要自行准备 MTP-Q4_0 主模型；将脚本名改为 `start-iq4.ps1` 并传入相应模型路径。

Both UIs are included. Full UI does not add backend tool execution or stream resumption.
两套 UI 均随包提供；完整 UI 不代表服务端新增工具执行或断线续传功能。

## Text-only startup / 无视觉头的纯文本启动

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
.\bin\llama-kvmem-server.exe -m 'D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf' `
  --host 127.0.0.1 --port 18200 -c 262144 -n 16384 `
  --kvmem-budget 36864 --kvmem-gen-reserve 16384 `
  -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3 `
  --enable-thinking --reasoning-budget 4096 --verbosity 3
```

## Requirements and validation / 环境与验证

Windows x64, an NVIDIA driver compatible with the selected package, Microsoft Visual C++ x64 runtime,
and AVX2/FMA/F16C/BMI2 CPU support are required. CUDA Toolkit, Visual Studio and Node.js are not needed.
需要兼容的 NVIDIA 驱动、VC++ x64 运行库和支持 AVX2/FMA/F16C/BMI2 的 CPU；运行不需要 CUDA Toolkit、Visual Studio 或 Node.js。

CUDA 13.2.86 targets: 75/80/86/89/90/120a. CUDA 12.9.86 additionally includes 70.
RTX 5060 Ti was physically tested; other targets need community validation.
This script update does not rebuild or change native executables or DLLs.
本次脚本更新没有重编译或更换 EXE/DLL。两套 CUDA 包的独立 IQ3 脚本已通过短流式推理和实时日志检查；
IQ3 的 Ctrl+C 退出已验证。IQ4 无本地模型，仅完成语法检查。

See `BUILD-INFO.json`, `VALIDATION.json`, `UI-VALIDATION.json`, `TEST-REPORT.md` and `RELEASE-NOTES.md`
for binary provenance and the scope of earlier tests. `SHA256SUMS` verifies the packaged files.
