# KVMem 128K Q8/Q8 control on RTX 5060 Ti

Tested September 23, 2026. Same GPU, original PTQ1 model, server binary, archive,
context 131072, retrieval budget 24576, reserve 10240, MTP off, thinking off,
batch/ubatch 128, FlashAttention and sampling settings as the preceding mixed-KV
run. Command comparison confirms that only `--cache-type-v` changed to `q8_0`.
Model and binary hashes match. Archive file is byte-identical: 130103 input tokens.

| Measurement | K Q8 / V Q4 | K Q8 / V Q8 |
|---|---:|---:|
| Recall | 1/3 | 1/3 |
| Prefill | 374.54 tok/s | 372.20 tok/s |
| Initial request wall time | 348.00 s | 350.24 s |
| Decode, 256 tokens | 35.72 tok/s | 37.50 tok/s |
| Sampled device VRAM peak | 6978 MiB | 7250 MiB |

No CUDA OOM or server crash. Recall answer:

```text
Based on the archive provided, only one secret access code was found:

*   **Desert Telescope:** CEDAR-8172 (from Archive entry 06885)

No secret access codes for the **lunar observatory** or the **coral laboratory** were present in the archive.
```

Raising V precision to Q8 does not resolve the recall failure. Neither MTP nor V Q4 is required to reproduce it. Further isolate sparse-history prefill/retrieval and recurrent-state handling from remaining kernel/server differences.

Code-block selection (blocks 1, 508, 914): `{"ORCHID-5831": true, "MAPLE-2964": true, "CEDAR-8172": true}`.
Selection is not proof of correct attention, layer representations or state restoration.
The [original Prism full-KV control](bonsai-prism-128k-recall-validation.md)
recalled 3/3 on this GPU with K Q8 / V Q4. Results cover one synthetic archive;
10K continuous output was not tested.

## Reproduce

Follow-up: [the same binary with KVMem disabled](bonsai-same-binary-128k-fullkv-validation.md)
recalls 3/3. The command differs only by `--no-kvmem`; GPU, model, binary and
Q8/Q8 types match. This narrows the failure to behavior enabled by KVMem, while
the precise prefill/retrieval/state stage remains unresolved.

```powershell
python scripts/bonsai-smoke.py --cache-type-k q8_0 --cache-type-v q8_0 --long --records 7650 --kvmem-only --decode-tokens 256 --request-timeout 7200 --context 131072 --budget 24576 --reserve 10240 --build build-win-bonsai-release --out logs/bonsai-128k-q8kv-5060-rerun --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf" --gpu GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c
```

Raw results: `logs/bonsai-128k-nomtp-kq8-vq8-5060ti/`.
Binary SHA256: `047f88ed12c9b84a4e59c6972ca00d2f01f7f59c1be1a560a3286a9807bf6a95`.
Model SHA256: `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`.
