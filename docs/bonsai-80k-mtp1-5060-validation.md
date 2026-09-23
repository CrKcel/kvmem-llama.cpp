# Bonsai 80K MTP1 recall on RTX 5060 Ti

Tested September 23, 2026. Context 81920, retrieval budget 24576, reserve 10240,
K/V Q8_0 for both target and draft, snapshot MTP draft=1. Original PTQ1 base
with the community r3 Q4_0 head (F32 norms). FlashAttention, batch/ubatch 128.
Server reasoning budget 4096; benchmark requests disable thinking.

The archive uses 4750 entries and the same three facility/code pairs as earlier
tests. They occur at entries 7, 2375 and 4275. The question is unchanged, but
the archive is shorter than the 128K test and the middle/late needles move.

| Measurement | Result |
|---|---:|
| Actual input | 80803 tokens |
| Recall | 1/3 |
| Prefill | 361.75 tok/s |
| Initial request wall time | 223.83 s |
| Decode, 256 tokens | 37.37 tok/s |
| Sampled device VRAM peak | 7876 MiB |
| Final conversation size | 81208 tokens |

No CUDA OOM or server crash. Recall response:

```text
Based on the archive provided, only one secret access code was found:

*   **Desert Telescope:** CEDAR-8172 (from Archive entry 4275)

No secret access codes for the **lunar observatory** or the **coral laboratory** were present in the archive.
```

Final sustained-generation MTP statistics:

```text
KVMEM_TRACE spec_stats n_gen=256 n_drafted=152 n_accept=104 n_restore=0 accept_pct=68.4
```

Verdict: **recall_failed**. This is one synthetic case, not a general
80K quality claim. It is not a length-only ablation of the recent 128K no-MTP
experiments: MTP is enabled and the archive changes. The 10K reserve is allocated,
but only 256 sustained output tokens were tested.

## Reproduce

```powershell
python scripts/bonsai-smoke.py --mtp --cache-type-k q8_0 --cache-type-v q8_0 --long --records 4750 --kvmem-only --decode-tokens 256 --request-timeout 7200 --context 81920 --budget 24576 --reserve 10240 --build build-win-bonsai-release --out logs/bonsai-80k-mtp1-rerun --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf" --gpu GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c
```

Raw results: `logs/bonsai-80k-mtp1-q4-kq8-vq8-5060ti/`.
Binary SHA256: `047f88ed12c9b84a4e59c6972ca00d2f01f7f59c1be1a560a3286a9807bf6a95`.
Model SHA256: `4b0e0f9b252d1b77ace4e197f913dbb8729a13c2a0174d4eb8b2b80c22918832`.
