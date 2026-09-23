# KVMem 128K recall without MTP on RTX 5060 Ti

Tested September 23, 2026. Original PTQ1 model, K Q8_0 / V Q4_0, context 131072,
retrieval budget 24576 plus generation reserve 10240, MTP off, FlashAttention,
batch/ubatch 128, four target CPU threads. Benchmark requests disable thinking.
The archive is byte-identical to the original Prism control: 130103 tokens.

| Measurement | Original Prism, full KV | KVMem, 24K + 10K pool |
|---|---:|---:|
| GPU | RTX 5060 Ti | RTX 5060 Ti |
| MTP | Off | Off |
| Recall | 3/3 | 1/3 |
| Prefill | 321.22 tok/s | 374.54 tok/s |
| Initial request wall time | 405.27 s | 348.00 s |
| Sampled device VRAM peak | 9630 MiB | 6978 MiB |

KVMem sustained decode: 256 tokens at
35.72 token/s. The Prism control only
generated a short recall reply, so its decode timing is not a matched comparison.
No CUDA OOM or server crash. The runtime test's final recall assertion determines
its exit code separately from whether requests completed.

Recall response:

```text
Based on the archive provided, only one secret access code was found:

*   **Desert Telescope:** CEDAR-8172 (from Archive entry 06885)

No secret access codes for the **lunar observatory** or the **coral laboratory** were present in the archive.
```

The recall failure persists without MTP on the same GPU where original Prism full KV recalled 3/3. MTP is therefore not required to trigger the failure. Investigate KVMem sparse-history prefill/retrieval, attention and recurrent-state handling, while retaining kernel/server differences as unresolved alternatives.

The three code blocks (1, 508, 914) were selected as follows:
`{"ORCHID-5831": true, "MAPLE-2964": true, "CEDAR-8172": true}`. Selection alone does not validate every layer's restored
KV, masks, higher-layer representations or recurrent state. This is a single
synthetic case; it does not isolate a specific faulty component.

## Reproduce

```powershell
python scripts/bonsai-smoke.py --cache-type-k q8_0 --cache-type-v q4_0 --long --records 7650 --kvmem-only --decode-tokens 256 --request-timeout 7200 --context 131072 --budget 24576 --reserve 10240 --build build-win-bonsai-release --out logs/bonsai-128k-nomtp-5060-rerun --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf" --gpu GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c
```

The smoke script enables MTP only when `--mtp` is specified. This differs from
the Windows user launcher, which enables MTP by default.
Raw results and logs: `logs/bonsai-128k-nomtp-kq8-vq4-5060ti/`.
Binary SHA256: `047f88ed12c9b84a4e59c6972ca00d2f01f7f59c1be1a560a3286a9807bf6a95`.
Model SHA256: `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`.
