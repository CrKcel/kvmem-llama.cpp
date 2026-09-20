# Bonsai no-MTP validation — 2026-09-20

Local branch: `kvmem-bonsai-llama.cpp`. This establishes initial text inference and KVMem spill/retrieval on 8GB hardware, not release-level quality or performance certification.

## Inputs and build

- KVMem rc3 snapshot plus Prism `9a9394a895b96003ca842a6041cb28ac49a108f7` and the maintained adapter patch.
- Model: `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5,946,648,928 bytes.
- Model SHA256: `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`.
- Download: ModelScope `prism-ml/Ternary-Bonsai-2-27B-gguf`, revision `b9720eaa678d70ad822a722c3e5b2f2f1082590c`.
- Windows x64, MSVC 19.44, CUDA 12.9.86, Release, SM120a, four build jobs.
- Server SHA256: `a159f1c39dc50bb5dff468a7912b6f143b06717ea7fce0734e783d46f5f3d9a6`.
- GPU: RTX 5050 Laptop, 8151 MiB reported by nvidia-smi; isolated by GPU UUID. RTX 5060 Ti was not used for inference.

## Results

| Check | Result |
| --- | --- |
| CLI, server, quantizer build | Passed |
| Host/server CTest suite | 11/11 passed; no skipped tests in this selected suite |
| PowerShell launcher validation | Passed, including Bonsai parameters and budget alignment |
| Patch replay on fresh Prism source | Passed; repeat application recognized; patched files matched live source |
| MTP startup rejection | Passed before model load |
| KVMem off/on, greedy short answers | Both returned `42` and `竹子` |
| Short multi-turn memory | Recalled `BAMBOO-7429` |
| Input beyond GPU KV pool | 4115-token archive processed with a 3072-token pool |
| Follow-up retrieval | 4148-token full prompt; recalled `ORCHID-5831` |

Model runs used `-ngl 99`, Flash Attention, Q8 K/V and batch/ubatch 128. The plain run used context 8192. The KVMem long run used context 32768, retrieval budget 2048 and generation reserve 1024. Model requests disabled thinking and used greedy sampling. Maximum reply length was 96 tokens.

Long-run traces contain prefill pressure with stage-out, retrieval with stage-in, and recurrent checkpoint restoration. The follow-up retrieval reported `stage_in=4 stage_out=4 gpu_reused=12`, with 28 replayed rows; the first archive request replayed 521 rows. These demonstrate that the test exercised memory movement and state replay, beyond a short prompt that stays resident.

The no-MTP allocation diagnostic reported 150,994,944 bytes of recurrent state, 5,898,240 bytes of convolution state and **0 rollback bytes**. Recurrent checkpoints still consume host memory: this smoke run reported a checkpoint peak of 785,783,820 bytes, excluding the other host allocations.

Peak whole-device VRAM sampled once per second was **6272 MiB** for the KVMem long run and 6284 MiB for the plain short run. These include desktop/driver allocations, are not process-only measurements, may miss brief peaks, and compare different context settings. The short-only KVMem run at context 8192 peaked at 6152 MiB. The 8-token retrieval reply measured about 14.1 token/s; this is too short to be a throughput benchmark.

## 32K follow-up test

The same binary, GPU and KV configuration subsequently processed **32013 input tokens** in a 32768-token context. The archive had 1880 synthetic entries with codes near the beginning, middle and 90% point. The input leaves room for follow-up messages and generated output; this is not 32768 input tokens plus output.

| Measurement | Result |
| --- | --- |
| Initial archive prefill, including retrieval/replay | 213.889 s, 149.67 token/s |
| Archive request, end to end | 214.177 s |
| Follow-up recall prompt | 32054 total tokens, 40 newly processed |
| Sustained generation prompt | 32162 total tokens, 43 newly processed |
| Sustained generation | 256 tokens in 18.912 s, **13.54 token/s** |
| Final conversation size | 32418 tokens including output |
| Peak sampled whole-GPU memory | **6263 MiB (6.12 GiB)** |
| OOM / process crash | None |
| Three-code exact recall | **1/3; failed** |

The retrieval response returned the late code `CEDAR-8172`, missed early `ORCHID-5831` and middle `MAPLE-2964`, and incorrectly attributed the returned code to entry 0940 (the actual late entry is 1692). The synthetic retrieval quality check therefore exits nonzero **after** completing and saving the sustained-generation measurements. Successful allocation and generation must not be reported as a successful recall test.

Whole-device VRAM was sampled once per second and includes desktop allocations; it may miss short peaks. This run demonstrates that the current bounded KV configuration can execute this approximately 32K workload on the tested 8GB GPU. It does not establish reliable 32K recall. The source of the recall failure is not isolated: comparison with a larger retrieval budget and a full-KV/unmodified-Prism baseline remains necessary before attributing it to the model, retrieval policy, or adapter.

Reproduce from the worktree (replace model path/GPU UUID as appropriate):

```powershell
python scripts/bonsai-smoke.py --long --records 1880 --kvmem-only --decode-tokens 256 `
  --out logs/bonsai-32k-smoke `
  --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf" `
  --gpu GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29
```

`logs/bonsai-32k-smoke/` contains `archive-request.json`, `results.json`, `kvmem.log` and timestamped `kvmem-memory.json` samples. No runtime source or binary was changed for this test.

## 64K with 24K retrieval and 10K generation reserve

At the user's requested settings, the same RTX 5050 8GB and binary passed a larger synthetic workload with Q8 K/V, no MTP, batch/ubatch 128 and context 65536. Retrieval budget was **24576**, generation reserve **10240**, total GPU KV pool **34816 tokens**. The allocation log reported **1,212,153,856 bytes (1156 MiB)** for the KV pool.

| Measurement | Result |
| --- | --- |
| Actual initial input | **64653 tokens** |
| Prefill including retrieval/replay | 499.795 s, **129.36 token/s** |
| Initial request end to end | 500.229 s (about 8 min 20 s) |
| Three-code recall | **3/3, correct facility/code associations** |
| Recall prompt and reply | 64694 input tokens, 37 output tokens |
| Sustained generation | **256 tokens in 27.233 s, 9.40 token/s** |
| Sustained generation input | 64774 tokens |
| Final conversation size | **65030 tokens**, including output |
| Peak sampled whole-device VRAM | **7454 MiB (7.28 GiB)** out of 8151 MiB |
| OOM / crash / test exit | None / none / 0 |

The response correctly paired lunar observatory with `ORCHID-5831`, coral laboratory with `MAPLE-2964`, and desert telescope with `CEDAR-8172`. There were 3800 archive entries, with codes at indices 7, 1900 and 3420. Prefill pressure, host stage-out/stage-in and state replay were exercised. Follow-up retrieval reported `stage_in=38 stage_out=38 gpu_reused=154`.

The peak leaves **697 MiB** against the device's reported total, including the desktop usage present during this run. Sampling is once per second and may miss brief peaks. The 10K generation reserve was allocated; this test generated 256 tokens rather than 10K. The near-64K input intentionally leaves room for replies within the 65536-token context.

This is a passing synthetic recall example, not proof of general 64K quality. Both history length and KV budget changed relative to the failed 32K/2K-budget case, so the runs do not isolate the effect of budget alone. More retained KV also increases attention work: sustained generation was 9.40 token/s in this run versus 13.54 token/s in the earlier smaller-pool run, which is a workload comparison rather than a controlled benchmark.

Reproduce:

```powershell
python scripts/bonsai-smoke.py --long --records 3800 --kvmem-only --decode-tokens 256 `
  --context 65536 --budget 24576 --reserve 10240 --out logs/bonsai-64k-24k-10k `
  --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf" `
  --gpu GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29
```

Interactive launcher equivalent: `./scripts/windows/start-bonsai.ps1 -Context 65536 -Budget 24576 -Reserve 10240 -Gpu 0` (choose the correct GPU index/UUID). The launcher's defaults remain unchanged.

Evidence: `logs/bonsai-64k-24k-10k/results.json`, `archive-request.json`, `kvmem.log` and `kvmem-memory.json`. The test service was stopped after completion and GPU memory was released. The runtime binary and installed app were not changed.

## Evidence and limits

Local evidence is retained in `logs/bonsai-smoke/` and `logs/bonsai-long-smoke/` (`results.json`, `plain.log`, `kvmem.log`), with CTest output in `build-win-bonsai/Testing/Temporary/LastTest.log`. The reproducible runner is `scripts/bonsai-smoke.py --long`.

The plain/KVMem comparison uses the same patched binary with KVMem disabled/enabled. It is not a comparison against pristine Prism, nor a token-logit equivalence test. Coverage consists of simple text, a passing 4K retrieval case, a failed 32K/2K-budget recall case, and a passing 64K/24K-budget recall case. No claims are made for general long-context quality, tool use, vision, ordinary Qwen regressions, other GPUs, Linux, or long-duration stability. Installed rc3 runtime files and the existing app were not changed.
