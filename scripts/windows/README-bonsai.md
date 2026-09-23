# Bonsai 2 Windows 实验版 — v0.16.0-rc3-prism.3

Windows x64，CUDA 12.9.86，编译包含 **SM75 / SM86 / SM89 / SM120a**，
对应 RTX **20 / 30 / 40 / 50** 系。RTX 5050 Laptop、5060 Ti 用于实机验证；
20/30/40 系仅完成架构编译，未做实机验证。

本包包含 server、CLI、完整聊天 UI 和 CUDA 运行库，不含模型。
需要兼容的 NVIDIA 驱动、Microsoft Visual C++ x64 Redistributable，
以及支持 AVX2/FMA/F16C/BMI2 的 CPU。运行不需要安装 CUDA Toolkit、Python、Node.js 或编译工具。

## 下载和启动

1. 解压整个 `kvmem-v0.16.0-rc3-prism.3-windows-x86_64-cuda12.9-rtx20-30-40-50.zip`，保留目录结构。
2. 从 [ModelScope](https://modelscope.cn/models/prism-ml/Ternary-Bonsai-2-27B-gguf/files)
   或 [Hugging Face](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/main)
   下载 `Ternary-Bonsai-2-27B-PTQ1_0.gguf`（约 5.95 GB）。
3. 在解压目录打开 PowerShell，执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-bonsai.ps1 -Model 'D:\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf' -Gpu 0
```

加载完成后打开 **http://127.0.0.1:18202/**。终端会显示 prefill/decode 时间和速度；
保持终端开启，Ctrl+C 停止服务。多显卡机器可修改 `-Gpu`，端口可用 `-Port` 修改。
省略 `-Model` 时，模型路径为 `%LOCALAPPDATA%\KVMem\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf`。
启动器不会自动下载或转换模型。

## 默认配置

| 设置 | 默认值 |
|---|---|
| MTP | **关闭**，无需 MTP 头 |
| 上下文上限 | 131072 tokens（128K） |
| 检索 KV 预算 | 24576 tokens（24K） |
| 生成预留 | 10240 tokens（10K） |
| 主模型 K/V | Q8_0 / Q8_0 |
| 思考 | 开启，预算 4096 tokens |
| batch / ubatch | 128 / 128 |

使用 `-Context`、`-Budget`、`-Reserve`、`-ReasoningBudget` 或 `-KvType` 覆盖。
思考 token 计入生成上限。128K 是可配置上限，**不代表长文召回质量通过验证**。

## 手动启用 MTP

需要额外准备带社区 ProCreations r3 MTP 头的合并 GGUF，见
`docs/bonsai-mtp-validation.md`；合并脚本在匹配的源码 ZIP 中。
MTP 头不是 Prism 官方 checkpoint。本版不集成 DSpark。

下面使用已准备好的 Q4_0 MTP 头模型，开启 MTP1：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-bonsai.ps1 -Model 'D:\models\Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf' -Gpu 0 -Mtp
```

代码生成场景可加 `-DraftTokens 2`。手动开启时默认 draft=1、草稿 K/V **F16**；
`-DraftKvType q8_0` 可覆盖草稿 KV，`-KvType` 只改变主 KV。
直接使用 server/CLI 时，手动启用参数为 `--spec-type draft-mtp`，
草稿类型可用 `--spec-kv-dtype` 覆盖。

如果只传 `-Mtp` 而不传 `-Model`，启动器仍按既有约定选择模型目录中的
`Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q8_0.gguf`；使用 Q4 头请显式指定路径。
`-NoMtp` 仍可显式关闭。开启 MTP 不会自动缩小 KV 池。

## 性能和验证范围

此前 5060 Ti、24K+10K、主 KV Q8、草稿 F16、关闭思考的两个 Python 代码任务：

| 输入长度 | 关闭 MTP | MTP1 | MTP2 |
|---|---:|---:|---:|
| 约 4K | 45.84 tok/s | 61.43 tok/s | 68.93 tok/s |
| 约 16K | 40.82 tok/s | 50.28 tok/s | 58.63 tok/s |
| 显存峰值 | 7208 MiB | 7766 MiB | 7916 MiB |

这些是 **decode** 速度；prefill 约 398–411 tok/s，整次请求提升较小。
测试使用 Q4 MTP 头、每项生成 156–352 tokens，未覆盖所有编程任务。
拓扑排序通过功能检查，区间合并三种模式均有同一边界错误。
详见 `docs/bonsai-coding-mtp012-5060.md`。历史速度数据来自加入 SM75 前的相同内核实现，
本次发布包验证以 `VALIDATION.json` 为准。

历史 F16/Q8 草稿 KV 对照中，F16 整卡峰值少约 68 MiB，速度没有一致改善。
长文测试曾出现答案块已选入预算但仍漏答：80K/128K 的部分测试仅召回 1/3；
本版不宣称修复长文召回，也未验证连续生成 10K。8GB 显存余量会受到显示和其他程序影响，
可先用 `-Context 8192 -Budget 2048 -Reserve 1024` 检查运行环境。
只支持主存 KV 外溢，不支持 NVMe 存储。

随包提供 `BUILD-INFO.json`、`VALIDATION.json`、`SHA256SUMS` 和许可证；
发布附件包含匹配源码 ZIP。升级请解压到新目录，以避免混用旧版 DLL 和脚本。

## 从源码重新编译

在 `kvmem-bonsai-llama.cpp` 分支初始化子模块后：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -BuildDir build-win-bonsai-release -CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9' -ExperimentalCuda129 -CudaArchitectures '75-real;86-real;89-real;120a-real' -Jobs 4
python scripts/build-webui.py --full-ui --output build-win-bonsai-release/share/kvmem/ui
```
