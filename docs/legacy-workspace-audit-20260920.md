# Legacy WSL workspace audit — 2026-09-20

The maintained Windows integration checkout was clean at `83bf285` before this documentation/version update. Two older WSL checkouts contain historical uncommitted work; they are not release build inputs.

## Findings and disposition

| Legacy content | Disposition |
|---|---|
| Server compatibility, auth, device selection, environment configuration, associated tests | Implemented in the newer integration; preserve newer logging/API/mixed-KV changes instead of copying old server files. |
| Detailed CLI/environment/authentication README sections | Restored and updated to require trace for structured diagnostics, with explicit LAN binding instructions. |
| rc3 VERSION and Windows headings | Applied to current source; published download links remain unchanged. |
| NVMe-disabled Windows portability and core Windows helpers | Present in rc2/current source; do not revert newer packaging/compiler/architecture safeguards. |
| Old Windows build/package scripts | Superseded: old copies default to one GPU architecture, lack newer compiler validation and combine the optional quantizer with runtime. |
| RawKvStore copy_mean_sum/commit_mean_sum and GGML masked-sum CPU/CUDA operator | Additional experimental implementation absent from the release baseline; no integration call sites found in current adapter/host source. Archived for a separate optimization review, not merged into rc3. |
| memory-guard/sample-memory scripts and tests | Auxiliary benchmark work, archived for separate adoption with its monitoring contract and tests; not a missing server runtime feature. |
| Feedback triage | Reconciled into the current feedback page; old unresolved user reports remain unresolved unless evidence establishes otherwise. |
| Reddit draft and froggeric template with license/provenance | Preserved in local archive; not installed as a default template or published as current guidance. |
| Remaining patched llama.cpp files | The rc2 worktree backend matches the frozen rc2 source for archived changed files. The earlier main worktree additionally contains the experimental masked-sum changes above. No submodule pointer is changed. |

## Archive and recovery

The local project root contains `wsl-archive-20260920.zip` and its expanded directory. It records each old HEAD, binary-capable tracked diffs, status, untracked/modified file contents, per-file SHA256, and comparisons with the current and frozen rc2 sources. The archive was CRC-checked and every recorded file hash verified. Original WSL checkouts were left intact; no reset, clean, or destructive removal was performed.

This is a preservation archive of working changes, not a standalone repository backup; recovery also needs the recorded base commits. Differences classified as experiments are deliberately retained rather than claimed to have been merged or validated.
