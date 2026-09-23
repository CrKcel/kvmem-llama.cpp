# Bonsai MTP off/1/2: Python code generation on RTX 5060 Ti

Tested September 23, 2026, after the F16 draft-KV default change (`7b91140`).
The same PTQ1 trunk + r3 Q4_0 MTP head is loaded in all three modes. Target
K/V Q8_0; draft K/V explicitly F16; context limit 131072; KVMem budget 24576
and generation reserve 10240; default 128-token pinned prefix; batch/ubatch
128; four CPU threads; FlashAttention; full GPU offload. MTP uses recurrent
GPU snapshots and p_min=0. KVMEM_TRACE is enabled, retrieval audit disabled.
Thinking is disabled per request, temperature 0, seed 42, presence/frequency
penalties 0. No runtime defaults were changed for this test.

## Workload and interpretation

Two tasks implement one Python function each: merging overlapping/touching
closed intervals without mutating input, and deterministic topological sort
with duplicate-edge handling, isolated nodes, and cycle rejection. Each task
is embedded after synthetic repository context consisting of unrelated
normalization helpers. This is a controlled coding workload, not a real
repository issue or a broad coding benchmark.

Actual input lengths are 3883/3900 tokens for the short tasks and 16274/16291
for medium tasks. All modes receive identical prompts. A fresh server runs
each mode, with the same 128-token warmup as the earlier prose benchmark.
Each task/length combination runs once per mode: two different tasks at
each length, **not repeated trials of the same task**. Prefix-cache hits are
2 tokens on the first short task and zero on the remaining requests.
No input reaches the retrieval budget and no eviction occurs.

The output cap remains 512 tokens, but complete functions may end naturally.
This differs from the prose benchmark's forced 512-token generation length;
throughputs use actual output tokens divided by measured generation time.
Aggregate rates sum tokens/time across the two tasks. Absolute throughput
should not be compared to prose without accounting for these differences.

## Results

| Context | MTP | Prefill tok/s | Decode tok/s | Decode gain vs off | Draft acceptance | Sampled peak MiB |
|---|---|---:|---:|---:|---:|---:|
| ~4K | Off | 405.92 | 45.84 | — | — | 7208 |
| ~4K | 1 | 399.70 | 61.43 | +34.0% | 97.16% | 7766 |
| ~4K | 2 | 397.81 | 68.93 | +50.4% | 88.67% | 7916 |
| ~16K | Off | 410.73 | 40.82 | — | — | 7208 |
| ~16K | 1 | 401.29 | 50.28 | +23.2% | 93.65% | 7766 |
| ~16K | 2 | 400.46 | 58.63 | +43.6% | 85.11% | 7916 |

MTP2 improves decode over MTP1 by 12.2%/16.6% at the cost of 150 MiB
additional device memory. Each normal MTP2 verification really proposes two
tokens, as confirmed by trace histograms. Average committed output tokens per
verification are 1.96/1.93 for MTP1 and 2.76/2.69 for MTP2 (short/medium).
Acceptance excludes bonus tokens. Host checkpoint restores are zero; rejected
suffixes still use GPU snapshot rollback.

Compared with the earlier prose workload, the much higher measured draft
acceptance makes longer speculation useful here. This supports trying MTP2
for code generation on the 5060 Ti; it does not establish that all coding
tasks behave this way. The global MTP1 default is unchanged.

Prefill falls approximately 1.5–2.5% with speculation. Mean whole-request
latencies (off/1/2) are 14.24/13.23/12.90 seconds at ~4K and
46.06/45.57/45.11 seconds at ~16K. Thus the MTP2 end-to-end reduction is only
9.5%/2.1% in these short-output, cold-prefill requests. Decode gains should
not be described as equivalent reductions in total request time.

All twelve requests end naturally with `finish_reason=stop`, without OOM or
server errors. Output tokens for merge/toposort are:

| Context | Off | MTP1 | MTP2 |
|---|---|---|---|
| ~4K | 156 / 258 | 156 / 258 | 156 / 258 |
| ~16K | 159 / 352 | 159 / 327 | 156 / 350 |

At ~4K both output texts are byte-identical across all modes. At ~16K, MTP1
matches the off-mode merge function but changes the topological-sort output;
MTP2 changes both texts. The merge difference is its docstring; topological
sort also varies implementation details. This is not evidence of universal
speculative output equivalence. Short generation times and one run per
task/length limit statistical precision.

## Functional checks

`scripts/bonsai-coding-check.py` strips an outer Markdown code fence, checks
the candidate AST, and runs each function in a separate five-second-limited
Python subprocess with restricted builtins. Interval tests include 107
targeted/random cases and compare covered sets at half-integer points;
topological-sort tests include 104 targeted/random graphs and use exhaustive
lexicographic permutations as an independent oracle. Tests cover input
mutation, empty cases, duplicate edges, isolated nodes, cycles and self-loops.
They are useful correctness checks, not a proof for arbitrary inputs.

Every topological-sort implementation passes 104/104 cases. Every interval
implementation passes 93/107 and fails the same 14 cases: it uses
`start <= last_end + 1` instead of `start <= last_end`, filling real gaps
between intervals. For example, `[(1,2),(3,4)]` incorrectly becomes `[(1,4)]`.
The prompt's own example also distinguishes adjacent integer coordinates
from touching closed intervals. This defect occurs with MTP off as well as
MTP1/2, so these checks show no MTP-specific correctness regression.
All responses include Markdown fences despite the requested plain function;
functional checks remove those fences, so passing does not imply perfect
output-format compliance.

## Reproduce

```powershell
python scripts/bonsai-mtp-context-bench.py --binary build-win-bonsai-release/bin/llama-kvmem-server.exe --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf" --gpu GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c --drafts 0 1 2 --draft-kv f16 --workload coding --out logs/bonsai-coding-rerun
python scripts/bonsai-coding-check.py logs/bonsai-coding-rerun
```

Raw commands, model/binary SHA256, outputs, timings, draft lengths, acceptance,
telemetry, and functional-check results are under
`logs/bonsai-coding-mtp012-5060ti/`. Prompt definitions are in
`scripts/bonsai_coding_cases.py`.

Binary SHA256: `ab281da239581c41ee9945fef52198849d2a53a0e5439de4bc3ebae5706d2e6b`.
Model SHA256: `4b0e0f9b252d1b77ace4e197f913dbb8729a13c2a0174d4eb8b2b80c22918832`.
