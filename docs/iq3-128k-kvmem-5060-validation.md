# IQ3 128K KVMem recall on RTX 5060 Ti

Tested September 23, 2026. Model `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`. Same server binary and GPU
as the Bonsai Q8/Q8 sparse-KV test; command comparison confirms only the model
path changed. Context 131072, retrieval budget 24576, reserve 10240, K/V Q8_0,
MTP off, thinking off, FlashAttention, batch/ubatch 128. The same short warmups,
multi-turn check, archive and follow-up question were used. Archive JSON is
byte-identical. The IQ3 file contains an MTP head, but MTP was not enabled.

| Measurement | Bonsai PTQ1 | IQ3 checkpoint |
|---|---:|---:|
| Actual input tokens | 130103 | 130103 |
| Recall | 1/3 | 2/3 |
| Prefill | 372.20 tok/s | 607.94 tok/s |
| Initial request wall time | 350.24 s | 214.78 s |
| Decode, 256 tokens | 37.50 tok/s | 23.76 tok/s |
| Sampled device VRAM peak | 7250 MiB | 12848 MiB |

No CUDA OOM or server crash. Recall answer:

```text
Based on the archive provided, here are the secret access codes:

*   **Coral Laboratory:** MAPLE-2964
*   **Desert Telescope:** CEDAR-8172
*   **Lunar Observatory:** No code was found in the archive.
```

The IQ3 checkpoint also misses information under this KVMem configuration. The issue is not confined to the tested Bonsai checkpoint. A full-KV IQ3 control would be needed to distinguish this checkpoint's baseline behavior from the sparse-KV path.

These are different checkpoints, not two quantizations of identical source weights.
The recall selected set contains 192 blocks, including blocks 1, 508 and 914;
block 1 was not omitted from the selection. Selection alone does not prove
correct state restoration or use of its contents by the model.
Decode histories include different preceding model answers, so the rates are
descriptive. This single synthetic archive does not establish general quality.

## Reproduce

```powershell
python scripts/bonsai-smoke.py --cache-type-k q8_0 --cache-type-v q8_0 --long --records 7650 --kvmem-only --decode-tokens 256 --request-timeout 7200 --context 131072 --budget 24576 --reserve 10240 --build build-win-bonsai-release --out logs/iq3-128k-5060-rerun --model "$env:LOCALAPPDATA/KVMem/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" --gpu GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c
```

Raw results: `logs/iq3-128k-nomtp-kq8-vq8-5060ti/`.
Binary SHA256: `047f88ed12c9b84a4e59c6972ca00d2f01f7f59c1be1a560a3286a9807bf6a95`.
Model SHA256: `58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f`.
