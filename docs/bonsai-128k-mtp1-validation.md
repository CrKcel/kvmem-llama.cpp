# Bonsai MTP1 128K validation on RTX 5050 Laptop

Tested on September 23, 2026 with the rc3-prism.2 multiarch server (binary source commit `37037f3`).
The Bonsai launcher now enables MTP draft=1 by default and chooses the merged r3 model.
Use `-NoMtp` to select the original PTQ1 model without MTP.

**Result: runtime completed without OOM or crash, but long-context recall failed (1/3 codes).**
The missing codes were near the start and middle; only the late code was recovered.
This is not a passing 128K quality validation. The script exited nonzero because
the recall assertion failed, after recording all timings and the 256-token decode.

## Configuration and results

Context 131072; retrieval budget 24576; generation reserve 10240; Q8 target/draft KV;
snapshot MTP draft=1; batch/ubatch 128; all layers on GPU; FlashAttention on.
Server thinking budget 4096 is preserved. **Benchmark requests disable thinking**
to measure prefill/decode and retrieval consistently with earlier tests.
The model is the unchanged PTQ1 base plus the community r3 Q8 MTP head.

| Measurement | Result |
|---|---:|
| Actual archive input | 130,103 tokens |
| Initial prefill including replay | 2134.732 s / 60.95 token/s |
| Initial request end to end | 2135.670 s |
| Planted-code recall | 1/3 |
| Recall input/output | 130,144 / 66 tokens |
| Sustained decode | 256 tokens / 55.267 s / 4.63 token/s |
| Final conversation size | 130,509 tokens |
| Peak whole-device VRAM (sampled) | 7845 MiB |
| Peak Windows shared GPU memory counter | 706.0 MiB |
| Peak process working set | 12.20 GiB |
| Lowest available system RAM | 2623 MiB |
| Prefill pressure events | 12 |
| OOM or crash | None |

Recall response:

```text
Based on the archive provided, only one secret access code was found:

*   **Desert Telescope:** CEDAR-8172 (from Archive entry 06885)

No secret access codes for the **lunar observatory** or the **coral laboratory** were present in the archive.
```

Final decode MTP statistics:

```text
KVMEM_TRACE spec_stats n_gen=256 n_drafted=157 n_accept=99 n_restore=0 accept_pct=63.1
```

The input contains 7650 archive entries, with three codes near the start, middle,
and 90% position. Reply space is reserved within the 131072-token context; this is
not 131072 input tokens plus an additional 10K output. The 10K generation pool was
allocated, but only 256 sustained output tokens were tested.
This is a single synthetic case, not a general quality or speed guarantee.
No matched 128K no-MTP benchmark was run. Do not extrapolate the earlier small-pool
MTP speedup percentage to this long-context result.
Shared GPU counters can include pinned/mapped resources; they do not independently
establish WDDM paging. Device memory samples can miss brief peaks, and display or
other GPU workloads can change available memory.

## Reproduction

```powershell
python scripts/bonsai-smoke.py --mtp --long --records 7650 --kvmem-only --decode-tokens 256 --request-timeout 7200 --context 131072 --budget 24576 --reserve 10240 --build build-win-bonsai-release --out logs/bonsai-128k-mtp1-rerun --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q8_0.gguf" --gpu GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29
```

Replace the GPU UUID for another machine. Responses, commands, API timings,
server trace, NVIDIA samples and Windows memory samples are in
`logs/bonsai-128k-mtp1-24k-10k-5050-final/`.
The first trial was deliberately interrupted to raise the client timeout from
1800 to 7200 seconds; its partial timings are excluded and it was not an OOM.

Binary SHA256: `496c873761801a3f61e689574a1ae0a36c1c254fd1975bb5067fa5874129b812`.
Model SHA256: `d020556899ccb7879bab6c2356cd6e633e14ddf12a4fb42eb7820b24d21220d9`.
