# Original Prism full 128K KV recall on RTX 5060 Ti

Tested September 23, 2026. Pristine Prism `9a9394a895b96003ca842a6041cb28ac49a108f7`; all 3526
archived source files matched the local source before building. Rebuilt GGML and
the server from that source with `LLAMA_USE_SYSTEM_GGML=OFF`, CUDA 12.9,
SM120a and `GGML_CUDA_FA_ALL_QUANTS=ON`. No KVMem or community kernel patches.

Original PTQ1 model without an MTP head; MTP disabled. Context 131072, one slot,
full K Q8_0 / V Q4_0 KV, FlashAttention, batch/ubatch 128, four CPU threads,
all model layers requested on GPU, fit off. Thinking disabled. Exact same
7650-entry archive and follow-up question as the preceding KVMem experiments.
The archive request file is byte-identical and tokenizes to 130103 tokens.

## Results

- Recall: **3/3**.
- Initial archive request: **405.27 seconds**.
- Initial prefill: **321.22 token/s**.
- Recall response: 38 reported output tokens, **16.85 token/s**; this short reply is not a sustained decode benchmark.
- Sampled whole-device VRAM peak: **9630 MiB**.
- Windows shared GPU counter peak: **116 MiB** (not independently evidence of paging).
- No CUDA OOM or server crash.

```text
- Lunar observatory: ORCHID-5831
- Coral laboratory: MAPLE-2964
- Desert telescope: CEDAR-8172
```

The original model with full KV recalls all three codes in this case. The earlier failure is specific to the tested configuration or implementation differences, not an unavoidable inability of this model to solve this archive. A same-GPU KVMem no-MTP control is still needed to separate sparse-KV/state handling from MTP integration. The original and optimized CUDA kernels also differ; this run alone does not identify the failing component.

## Needle position check

Follow-up: [KVMem without MTP on the same 5060 Ti](bonsai-kvmem-128k-nomtp-5060-validation.md)
still recalled only 1/3 with a 24K + 10K pool and the same K Q8 / V Q4 types.
MTP is therefore not required to reproduce the failure. Sparse-history/state
handling and remaining kernel/server differences need further isolation.

With the original model tokenizer and chat template, the code tokens occupy
blocks 1 (ORCHID-5831), 508 (MAPLE-2964), and 914 (CEDAR-8172), using 128-token
blocks. Both earlier KVMem recall selections include all three blocks among
their 192 selected blocks. Thus simply failing to select these code blocks is
not supported by the logs. This does not prove that all restored KV payloads,
attention masks, higher-layer representations or recurrent states are correct.

GPU and MTP settings differ from the earlier KVMem runs; do not treat timings
as a controlled KVMem performance comparison. This is one synthetic example.

## Reproduce

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build-prism-reference.ps1 -Jobs 4
python scripts/prism-benchmark.py --profile original-128k-recall --cache-type-k q8_0 --cache-type-v q4_0 --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf" --gpu GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c --out logs/prism-128k-full-kq8-vq4-5060ti-rerun
```

The profile reads the prior archive at
`logs/bonsai-128k-mtp1-q4-kq8-vq4-5050/archive-request.json`.
Raw requests, responses, token positions, timings, commands and memory samples
are in `logs/prism-128k-full-kq8-vq4-5060ti/`.

Binary SHA256: `bda91a7a183f9a0cd46bfa5f5bc98c85d1b362886b6a9c0d118ee99cca8772ff`.
Model SHA256: `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`.
