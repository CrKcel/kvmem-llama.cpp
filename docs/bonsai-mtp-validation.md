# Bonsai MTP snapshot experiment

Local branch: `kvmem-bonsai-llama.cpp`, based on `c7485c2`; Prism remains pinned to
`9a9394a895b96003ca842a6041cb28ac49a108f7`. Tested September 23, 2026.
These measurements began before kernel integration. The current branch includes the optimized kernels; see [rc3-prism.2 notes](milestones/v0.16.0-rc3-prism.2.md). The previous rc3-prism.1 draft package lacks MTP.

## Model preparation

Downloaded the community-trained [ProCreations r3 head](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP)
from hf-mirror, pinned to `efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51`.
This is not an official Prism MTP checkpoint.

- Head SHA256: `7a4a18b2d02116ef184d1b0ee4af46d829825ff2c042f79cf37ef8a03c399218`.
- Original PTQ1 SHA256: `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`.
- Output: `%LOCALAPPDATA%\KVMem\models\Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q8_0.gguf`.
- Output size: 6,397,969,760 bytes; approximately 430.4 MiB added.
- Output SHA256: `d020556899ccb7879bab6c2356cd6e633e14ddf12a4fb42eb7820b24d21220d9`.

`scripts/bonsai-mtp-graft.py` reads BF16 safetensors without Torch, uses the pinned
Prism gguf-py quantizer for Q8 matrices by default (or `--head-type Q4_0`), converts zero-centered norms to effective
F32 multipliers, and appends 15 tensors. All 851 original tensor payloads are
verified byte-for-byte. The original model and Hadamard metadata are preserved;
the head does not duplicate the embedding table. A JSON manifest is beside the output.
Python dependencies are NumPy, PyYAML and tqdm, plus this submodule's gguf-py.

To reproduce the graft, use the matching source archive (including `llama.cpp/gguf-py`)
or a checkout with initialized submodules. Python is required only for model conversion,
not for running the prebuilt service. In PowerShell, from that source directory:

```powershell
python -m pip install numpy PyYAML tqdm
curl.exe -L --fail --retry 3 -o model_mtp.safetensors 'https://hf-mirror.com/ProCreations/Ternary-Bonsai-2-27B-MTP/resolve/efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51/model_mtp.safetensors'
python scripts/bonsai-mtp-graft.py --base 'D:\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf' --head model_mtp.safetensors --output 'D:\models\Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q8_0.gguf'
```

The converter verifies the pinned base/head hashes above and refuses to overwrite
an existing output. Keep the original PTQ1 model for no-MTP use.

For the Q4 MTP experiment, add `--head-type Q4_0` and choose a separate output
such as `Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf`. This quantizes only the
MTP matrices directly from the original BF16 head; norms stay F32 and the base
PTQ1 payloads are unchanged. The head adds about 227.9 MiB instead of 430.4 MiB.
See the [128K mixed-KV comparison](bonsai-128k-q4-mtp-mixed-kv-validation.md)
for the experiment using K Q8_0 / V Q4_0 in both target and draft caches.

## Implementation

- Restore the embedding Hadamard inverse in Qwen35's MTP graph.
- Restore rc3 speculative sampling/verification, using GPU GDN snapshot planes.
- Restore the MTP follower KV pool, sized from the target pool, and preserve logical
  positions, hidden-state carry, accepted-prefix trimming and retrieval synchronization.
- Distinguish a target-context reference from genuinely shared KV memory.
- Preserve no-MTP operation via `-NoMtp`; the Bonsai launcher now enables MTP with one draft token by default. The original implementation was opt-in.
- Record/Fold and DSpark remain outside this implementation.

`patches/llama-kvmem-current.patch` was replayed against pristine files from the
pinned Prism commit at the initial MTP stage; all 24 resulting files matched that
working tree. The current cumulative patch also includes the later CUDA kernel changes. The independent Prism comparison uses only
`patches/prism-bonsai-mtp-embedding.patch`, built with `build-prism-reference.ps1 -Mtp`.

## Validation

Later default-pool test: the [128K MTP1 run on RTX5050](bonsai-128k-mtp1-validation.md)
completed without OOM, but recalled only1/3 codes. See that report for the full
35-minute prefill, 4.63 token/s decode and memory measurements. This is a recall
failure, not a passing128K quality test. The smaller-pool measurements below are historical.

Follow-up: after the user disabled Smart App Control, the optional optimized
kernel build passed the previously blocked MTP KV suite. It also made draft=1
faster than no-MTP in the measured workload. See
[the kernel comparison](bonsai-kernel-comparison.md) for the required PDL fix,
full paired measurements and validation; the original-build measurements below
remain unchanged.

Build: MSVC, CUDA 12.9.86, SM120a. Current local build passed all 11 CTest tests.
An initial server-progress test invocation failed; its direct rerun and two later
CTest runs passed. The separate `kvmem-mtp-kv-test.exe` compiles but Windows
application control rejects execution with error 4551. Its low-level KV transfer
suite was therefore **not runtime-validated at that stage**.

RTX 5050 Laptop 8GB integration configuration: context 8192, target KV budget
2048 + generation reserve 1024, Q8 target/draft KV, batch/ubatch 128, temperature 0.
All three modes passed arithmetic, Chinese translation, multi-turn identifier
recall, output limits, a 4108-token archive crossing the 3072-cell pool followed
by a 4138-token recall turn, a 16-token thinking budget, and a new request after
stream cancellation. The archived identifier was recovered in each mode.

| Mode | Short prose decode, tok/s | Archive prefill, tok/s | Peak device MiB | Prose draft acceptance |
|---|---:|---:|---:|---:|
| MTP off | 13.46 | 137.38 | 6076 | n/a |
| draft=1 | 13.74 | 134.71 | 6664 | 50/78, 64.1% |
| draft=2 | 11.50 | 134.60 | 6814 | 62/132, 47.0% |

Decode is a single 128-token short-prompt sample, not a repeated benchmark or a
long-context speed guarantee. Logs exercise both full acceptance and rejection:
draft=1 has 34 rejected verification batches; draft=2 has 51. No host fallback
restores were needed in those generation loops; rejection uses GPU snapshots.
These integration checks do not establish bitwise equivalence for arbitrary prompts.

The final-build smoke repeated draft=1 and draft=2, including disconnecting only
after three actual streamed text chunks. Logs confirm `stream_abort phase=spec_gen`
after 6 and 7 generated tokens respectively; the following arithmetic request
succeeded. The PowerShell `-Mtp` launcher also passed an actual startup/request test,
with prefill/decode timing messages forwarded to its console log.

Independent Prism comparison on RTX 5060 Ti 16GB, context 8192 and Q8 KV:

| Mode | Short prose decode, tok/s | Peak device MiB | Prose draft acceptance |
|---|---:|---:|---:|
| MTP off | 41.45 | 6064 | n/a |
| draft=1 | 36.54 | 6724 | 52/74, 70.3% |
| draft=2 | 31.53 | 6874 | 66/120, 55.0% |

Prism passed the short-answer/multi-turn checks. A later KVMem run on the 5060
encountered other services sharing that GPU and was stopped; its timing results
are excluded. No other service was stopped. Host activity was not isolated, so
the small 5050 draft=1 difference should be treated as roughly unchanged speed.

Full responses, commands, timings and logs are under:

- `logs/bonsai-mtp-prism-5060/`
- `logs/bonsai-mtp-kvmem-5050/`
- `logs/bonsai-mtp-final-5050/` (final build smoke)

The later24K+10K/128K run is reported above. Continuous10K generation,
vision/tool requests, and RTX30/40 hardware remain unvalidated.
Do not apply the old no-MTP 64K memory result to this snapshot configuration.

## Start the local experimental build

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-bonsai.ps1 -Mtp -Gpu 0 -Context 8192 -Budget 2048 -Reserve 1024 -ReasoningBudget 256
```

Open `http://127.0.0.1:18202/`. This chooses the new local GGUF automatically,
uses Q8 target/draft KV and one draft token, and forwards server logs to the console.
Use `-DraftTokens 2` only for comparison. `-Model` can override the model path.

With `-NoMtp`, the original model is selected; context remains 128K, budget 24K,
reserve 10K, thinking budget 4096. Enabling MTP does not silently shrink those
defaults, so specify the smaller tested pool explicitly on an 8GB card.
The command above sets a smaller thinking budget to fit the example's 1K output reserve.

Reproduce the integration checks:

```powershell
python scripts/bonsai-mtp-smoke.py --binary build-win-bonsai/bin/llama-kvmem-server.exe --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q8_0.gguf" --gpu GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29 --kvmem --long --out logs/bonsai-mtp-rerun
```

The later PTQ1 kernel investigation is complete for the measured workload: draft=1
adds about 15-18% over optimized no-MTP decode on the two tested GPUs. See the
kernel comparison above. The launcher now defaults to MTP draft=1 at the user's request; explicit `-NoMtp` remains available to reduce memory use.
