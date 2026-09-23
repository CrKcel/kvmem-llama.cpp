# Bonsai 80K recall: packed KV and position audit

Tested September 23, 2026 on RTX 5060 Ti. Base source commit `829ce02`.
Same PTQ1 model with r3 Q4_0 MTP head, MTP1 snapshots, target/draft K/V Q8_0,
context 81920, retrieval budget 24576, generation reserve 10240, batch/ubatch
128 and FlashAttention as the previous 80K test. Thinking is disabled in requests.
The archive contains 80803 input tokens and the same three needles.

## Diagnostic coverage

`KVMEM_AUDIT_RETRIEVAL=1` compares the packed host and GPU K/V bytes across
all 16 attention layers before eviction and after selection/writeback. It also
checks every selected cell's sequence membership and original model position.
The switch is off by default. This verifies transport/storage consistency, not
whether those tensors encode the same computation as a full-history prefill.

The old `dump_kv_compare` sampled layers 0, 32 and 63; only layer 63 has attention
KV in this model. The new audit closes that coverage gap.

## Default prefix, 128 tokens

Recall remains **1/3**, with exactly the previous answer: only CEDAR-8172, while
claiming the lunar and coral codes were absent.

- All nine pre-eviction audit batches: zero byte mismatches, missing entries,
  or bad positions.
- Each of three post-retrieval audits: 192 resident blocks, 6112 K/V block-layer
  comparisons, 851181568 bytes checked, zero byte mismatches or bad positions.
- Each post-retrieval audit has 32 comparisons without a reference. This first
  build did not log their block IDs, so it cannot independently certify those
  missing comparisons. A follow-up build logs the first missing block/layer.
- The three answer blocks 1, 315 and 568 are resident after question retrieval.
- The final question token at position 80842 sees 24523 valid KV cells, matching
  191 full blocks plus the 75-token tail. The aggregate mask trace shows no loss
  of these occupied cells; it is not an attention-weight measurement.

Raw evidence: `logs/bonsai-80k-audit-allkv-5060ti/`, including `audit-summary.json`.
Diagnostic binary SHA256:
`52085d3796eaba6cda94aa3d8371b54ef43c0e230d7ce2c262335090582cf474`.

## Fixed prefix, 512 tokens

Adding `--sink-tokens 512` to the smoke command (server option
`--kvmem-sink-tokens 512`) changes recall to **3/3**:

```text
- Lunar observatory: ORCHID-5831
- Coral laboratory: MAPLE-2964
- Desert telescope: CEDAR-8172
```

The total selection budget remains 24576 and the reserve remains 10240; the
additional 384 prefix tokens replace other selected tokens. Device VRAM peak
is 7876 MiB in both runs. Initial request wall time is 224.22 seconds versus
224.14 seconds in the 128-token prefix audit. These diagnostic timings are not
a new performance benchmark.

All pre-eviction and post-retrieval byte/position checks again report zero
mismatches. The missing-reference entries now explicitly identify the partial
tail: block 631 with 34 tokens after the archive, block 631 with 75 tokens after
the recall query, and block 632 for the subsequent essay request. GPU storage
exists; the packed host reference is absent. The three needle blocks are full
blocks and are not these missing-reference entries.

The sink512 build differs from the first audit build only by logging details
for missing references. SHA256:
`0256561fd7df564fd4a1cbfee50609d6c978ff9ce8fc6a80d6034acc0274a667`.
Raw evidence: `logs/bonsai-80k-audit-sink512-5060ti/`.

## Interpretation and limits

No packed-KV corruption or position error was found in the comparisons that
could be performed. Selection success alone does not establish successful use
of the recalled information. The model's recurrent state and deeper-layer KV
were computed while the prefill working set was rolling; restoring earlier KV
does not reconstruct the recurrent computation for the missing history.

The server checkpoints recurrent state at the query boundary, probes the query,
selects KV, restores that checkpoint and replays the query. This is a candidate
for further controlled investigation, not a proven checkpoint bug. The earlier
same-binary 128K full-KV control succeeded, so the sparse KVMem path remains the
focus.

Keeping a larger prefix available throughout prefill and generation is a
promising configuration mitigation for this case. It changes both the prefill
computation and the selected context; it does not isolate recurrent state from
attention normalization or establish which extra prefix tokens matter. The
middle needle also recovers although it is outside the pinned prefix. Do not
interpret this as simply pinning every answer or as proof of a recurrent-state
restore defect. This is one synthetic 80K case; 128K and other needle positions
have not been validated with sink512. Production defaults remain unchanged.

## Validation

The diagnostic server builds successfully and all 11 existing CTest tests pass.
The default-prefix long-run harness exits 1 because its recall assertion fails;
the sink512 harness exits 0. Both finish the workload and 256-token generation
without crashing.

Reproduce with the command in `bonsai-80k-mtp1-5060-validation.md`, set
`$env:KVMEM_AUDIT_RETRIEVAL='1'`, choose a fresh `--out` directory, and add
`--sink-tokens 512` for the second variant. Remove the environment variable after
the diagnostic session if it was set in an interactive shell.
