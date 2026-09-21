# Launcher fixes and rc3 sync — 2026-09-20

## English

This change set contains launcher-level fixes contributed by the melis2023
workspace and a synchronization with the upstream `v0.16.0-rc3` release
(`ccd8494`).

### Changes

1. **Allow LAN API access via configurable host** (`fix(launcher)`)
   - `scripts/start-server.py` and `scripts/windows/start-server.ps1` now
     respect `--host` / `-ListenHost` and the `HOST` / `LLAMA_ARG_HOST`
     environment variables as the bind address.
   - Previously the launchers hard-coded `127.0.0.1`, so clients on the LAN
     could not reach the API even when the server supported `--host 0.0.0.0`.
   - Windows launcher now also falls back to `$env:LLAMA_ARG_HOST` before the
     final `127.0.0.1` default, mirroring the `LLAMA_ARG_HOST` convention used
     by llama-server.

2. **Pass `--api-key` / `--api-key-file` through to the server**
   (`feat(launcher)`)
   - `scripts/start-server.py` gains `--api-key` and `--api-key-file`
     arguments that are forwarded verbatim to the server binary.
   - `scripts/windows/start-server.ps1` gains `-ApiKey` (and `-ApiKeyFile`,
     which defaults to the `LLAMA_ARG_API_KEY_FILE` environment variable).
   - This mirrors llama-server's own authentication options, so launcher users
     can enable API-key authentication without invoking the binary directly.
   - Uploaded as two commits: `feat(launcher): pass --api-key and
     --api-key-file through to the server` and `docs: document launcher
     --api-key / -ApiKey options` (README updates in both English and Chinese).

3. **Merge upstream `v0.16.0-rc3` (`ccd8494`)**
   - Rebases this workspace onto the upstream rc3 release (CUDA 13.2 and
     CUDA 12.9 Windows builds, CPU-vision defaults).
   - The Windows launcher now defaults vision offload to CPU
     (`--no-mmproj-offload`) for all recipes; pass `-VisionDevice gpu` to
     restore GPU vision offload. The previous IQ3/GPU - IQ4/CPU split was
     simplified to a single CPU default.
   - The API documentation now states that native TLS is not supported.

### Validation

- `python -m py_compile scripts/start-server.py` passes.
- `System.Management.Automation.Language.Parser` reports no errors for
  `scripts/windows/start-server.ps1`.
- Linux dry-run: `--api-key sk-abc` and `--api-key-file` appear in the
  resolved server argv.
- Windows dry-run (`-DryRun -ApiKey sk-abc`, `-DryRun` with
  `LLAMA_ARG_HOST=10.0.0.2`): host and api-key flags resolve correctly; CPU
  vision (`--no-mmproj-offload`) is applied by default.

---

## 中文

本变更集包含 melis2023 工作区对启动脚本（launcher）的修复，以及与上游
`v0.16.0-rc3`（`ccd8494`）的同步。

### 变更内容

1. **允许通过可配置 host 在局域网访问 API**（`fix(launcher)`）
   - `scripts/start-server.py` 与 `scripts/windows/start-server.ps1` 现在会
     遵循 `--host` / `-ListenHost`，以及 `HOST` / `LLAMA_ARG_HOST` 环境变量
     作为监听地址。
   - 此前启动脚本把地址硬编码为 `127.0.0.1`，即使服务端支持
     `--host 0.0.0.0`，局域网客户端也无法访问 API。
   - Windows 启动脚本在最终回退到 `127.0.0.1` 之前，会先读取
     `$env:LLAMA_ARG_HOST`，与 llama-server 的 `LLAMA_ARG_HOST` 约定保持一致。

2. **向服务器透传 `--api-key` / `--api-key-file`**（`feat(launcher)`）
   - `scripts/start-server.py` 新增 `--api-key` 与 `--api-key-file` 参数，
     原样转发给服务器二进制。
   - `scripts/windows/start-server.ps1` 新增 `-ApiKey`（以及 `-ApiKeyFile`，
     默认读取 `LLAMA_ARG_API_KEY_FILE` 环境变量）。
   - 与 llama-server 自身的鉴权选项对齐，使用启动脚本的用户无需直接调用
     二进制即可启用 API-Key 鉴权。
   - 拆分为两个提交：`feat(launcher): pass --api-key and --api-key-file
     through to the server` 与 `docs: document launcher --api-key / -ApiKey
     options`（README 中英文均同步更新）。

3. **合并上游 `v0.16.0-rc3`（`ccd8494`）**
   - 将本工作区对齐到上游 rc3 发布（Windows CUDA 13.2 与 CUDA 12.9 两个构建，
     CPU 视觉默认值）。
   - Windows 启动脚本所有配方默认将视觉模块卸载到 CPU
     （`--no-mmproj-offload`）；如需恢复 GPU 视觉卸载，请传
     `-VisionDevice gpu`。原先 IQ3/GPU、IQ4/CPU 的双默认值被简化为统一的
     CPU 默认值。
   - API 文档补充说明：暂不支持原生 TLS。

### 验证结果

- `python -m py_compile scripts/start-server.py` 通过。
- `System.Management.Automation.Language.Parser` 对
  `scripts/windows/start-server.ps1` 报告无错误。
- Linux dry-run：解析后的服务器 argv 中包含 `--api-key sk-abc` 与
  `--api-key-file`。
- Windows dry-run（`-DryRun -ApiKey sk-abc`，以及设置
  `LLAMA_ARG_HOST=10.0.0.2` 后运行 `-DryRun`）：host 与 api-key 参数解析
  正确；默认使用 CPU 视觉（`--no-mmproj-offload`）。