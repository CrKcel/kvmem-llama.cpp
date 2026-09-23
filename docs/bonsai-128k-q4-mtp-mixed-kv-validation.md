# Bonsai Q4 MTP head and mixed KV: 128K on RTX 5050 Laptop

Tested September 23, 2026. This local experiment keeps the original PTQ1 base
unchanged (all 851 tensor payloads verified). The pinned community r3 BF16 head is
directly quantized to Q4_0 matrices, with F32 norms. Added head size: 227.9 MiB,
versus 430.4 MiB for Q8_0. Merged model: 6,185,633,120 bytes.

Both configurations use context 131072, retrieval budget 24576, reserve 10240,
snapshot MTP draft=1, batch/ubatch 128 and FlashAttention. Both target and draft
use K Q8_0 / V Q4_0 in the new run; startup logs confirm both. Benchmark requests
disable thinking, while the server default reasoning budget remains 4096.

| Measurement | Previous: Q8 head, K/V Q8 | New: Q4 head, K Q8 / V Q4 |
|---|---:|---:|
| Identical archive input | 130,103 tokens | 130,103 tokens |
| Prefill speed | 60.95 tok/s | 130.49 tok/s |
| Prefill time | 2134.73 s | 997.06 s |
| Decode, 256 tokens | 4.63 tok/s | 16.28 tok/s |
| Sampled whole-device VRAM peak | 7845 MiB | 7766 MiB |
| Windows shared GPU counter peak | 706 MiB | 290 MiB |
| Planted-code recall | 1/3 | 1/3 |
| CUDA OOM / crash | None | None |

The verdict is **recall_failed**. Exact recall response:

A subsequent [original Prism full-KV control on RTX 5060 Ti](bonsai-prism-128k-recall-validation.md)
recalled all three codes with K Q8_0 / V Q4_0 and MTP disabled. This narrows the
investigation but does not isolate KVMem state handling, MTP, or kernel differences.

```text
Based on the archive provided, only one secret access code was found:

*   **Desert Telescope:** CEDAR-8172 (from Archive entry 06885)

No secret access codes for the **lunar observatory** or the **coral laboratory** were present in the archive.
```

Final sustained-generation MTP statistics:

```text
KVMEM_TRACE spec_stats n_gen=256 n_drafted=153 n_accept=103 n_restore=0 accept_pct=67.3
```

The test uses the exact same archive as the previous run. This is a combined
weight/KV quantization comparison: it cannot isolate either change's individual
effect on speed or quality. Shared GPU memory counters may include pinned/mapped
resources and alone do not prove device-memory paging. Sampling can miss peaks.
No 10K continuous generation or general long-context quality claim is made.

## Reproduce

The graft script accepts `--head-type Q4_0`; its default remains Q8_0. Use the
original BF16 `model_mtp.safetensors`, not a requantization of the Q8 GGUF.

```powershell
python scripts/bonsai-smoke.py --mtp --cache-type-k q8_0 --cache-type-v q4_0 --long --records 7650 --kvmem-only --decode-tokens 256 --request-timeout 7200 --context 131072 --budget 24576 --reserve 10240 --build build-win-bonsai-release --out logs/bonsai-128k-mixed-kv-rerun --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf" --gpu GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29
```

Server MTP cache default was corrected from F16 to the inheritance sentinel,
matching the CLI and help. The previous benchmark explicitly selected Q8 draft KV
and is unaffected. Existing release archives are unchanged by this local experiment.
The 11 existing CTest cases passed. A separate short run passed the new runtime
assertions for both target and draft mixed KV, plus arithmetic, translation and
multi-turn recall. The long test exits with assertion failure because recall is
1/3, not because the server crashed.

Binary SHA256: `047f88ed12c9b84a4e59c6972ca00d2f01f7f59c1be1a560a3286a9807bf6a95`.
Model SHA256: `4b0e0f9b252d1b77ace4e197f913dbb8729a13c2a0174d4eb8b2b80c22918832`.
Raw logs, requests, results, memory samples, and source diff are in
`logs/bonsai-128k-mtp1-q4-kq8-vq4-5050/`.
