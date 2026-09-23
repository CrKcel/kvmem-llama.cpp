# Build and deploy with an AI agent

This is a runbook for a coding agent (or a human following along) that goes from a
fresh clone to a **verified** running `llama-kvmem-server`. It is deliberately
explicit about pass criteria, because the two failure modes that waste the most
time here are silent ones: a build made with the wrong CUDA architecture, and a
VRAM budget that only looks sufficient while idle.

---

## 0. Pre-flight: probe before touching anything

Report the probe as a table and stop for confirmation if anything is missing.

| What | How | Required |
|---|---|---|
| OS / arch | `uname -a` (Linux/macOS), `[Environment]::Is64BitOperatingSystem` + `$PSVersionTable` (Windows) | — |
| GPU + VRAM | `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv` | a listed GPU |
| CUDA toolkit | `nvcc --version` — read the **release** line, not `nvidia-smi` | **13.2.86 or newer** |
| CMake | `cmake --version` | ≥ 3.20 |
| Ninja (optional) | `ninja --version` | faster builds |
| Compiler | `gcc --version` / `clang --version` / `cl.exe` | C++17 |
| RAM | `free -g` / `Get-CimInstance Win32_OperatingSystem` | see README memory table |
| Disk free | `df -h` / `Get-PSDrive` | ≥ 15 GB for build + model |

**Do not skip the nvcc check.** The validated baseline is nvcc **13.2.86**
(CUDA 13.2 Update 2). A build made with 13.2.51 was observed to produce garbage
output from an IQ3_S model even with KVMem and MTP disabled; rebuilding the same
source with 13.2.86 fixed it. A successful build plus a small Q8 smoke test does
**not** validate a low-bit model.

`nvidia-smi` reports the **driver**'s CUDA capability, which is a different thing
from the toolkit that compiles the kernels. Check the toolkit.

---

## 1. Clone and replay the patches

```bash
git clone --recurse-submodules <repo-url>
cd kvmem-llama.cpp
scripts/apply-patches.sh
```

**Pass criteria:** the script prints either `applied current KVMem patch ...` or
`KVMem patches already applied`. It is idempotent — running it twice is expected
and safe.

**Do not** apply the numbered `patches/0001`–`0004` on top of the cumulative
patch; they are historical and do not replay on the current base. See
`patches/README.md`.

**Do not** commit a dirty `llama.cpp` working tree. The submodule pointer *is* the
pin; local modifications belong in `patches/` and are replayed by the script.

---

## 2. Pick the CUDA architecture

`scripts/build-cuda.sh` defaults to `CMAKE_CUDA_ARCHITECTURES=120a-real`, which is
the author's test GPU. Map the card actually present:

| GPU generation | `CMAKE_CUDA_ARCHITECTURES` |
|---|---|
| RTX 20xx (Turing) | `75-real` |
| RTX 30xx (Ampere) | `86-real` |
| RTX 40xx (Ada) | `89-real` |
| H100 / RTX 50xx | `90-real` / `120a-real` |

`-real` compiles for that architecture only and is enough when building on the
target machine. Add `-virtual` variants only if the binary must run elsewhere.
A wrong value either fails the build or produces kernels that fault at runtime.

---

## 3. Build

```bash
CMAKE_CUDA_ARCHITECTURES=86-real scripts/build-cuda.sh   # substitute your arch
```

Overridable environment variables — nothing is hardcoded to one machine:

| Variable | Default | Notes |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `120a-real` | see step 2 |
| `CMAKE_CUDA_COMPILER` | `nvcc` | absolute path if CUDA is off PATH |
| `CMAKE_BUILD_TYPE` | `Release` | |
| `BUILD_DIR` | `<root>/build` | **use a fresh directory after a toolkit change** |
| `NPROC` | `nproc` | parallelism |

**Pass criteria:** the script prints `binaries under <build>/bin` and
`<build>/bin/llama-kvmem-server` exists.

Replacing CUDA DLLs or upgrading the driver does **not** fix kernels already
compiled into an old binary — reconfigure into a new build directory and rebuild.

A CUDA-free build is possible for CPU or other backends:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
      -DKVMEM_BUILD_LLAMA=ON -DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT="$PWD"
cmake --build build -j
```

`build-cuda.sh` enables `GGML_CUDA_FA_ALL_QUANTS=ON`, which is what makes
`--kv-dtype q5_0` work on hybrid models. `q8_0` and `q4_0` do not need it.

---

## 4. Get a model

No weights are bundled. Download the GGUF matching the VRAM budget, or use a
release archive (see the README's prebuilt table for CUDA 12.9 / 13.2 Windows and
Linux/ROCm builds).

**Pass criterion:** the file exists and its size matches the published size —
a truncated download otherwise surfaces much later as a confusing load error.

Place it under `models/` (gitignored) or anywhere outside the tree.

---

## 5. Size the GPU KV pool

```
GPU pool (tokens) = --kvmem-budget + --kvmem-gen-reserve
resident VRAM     ≈ model weights + pool × per-token cost
```

Per-token cost depends on the KV type — `q4_0` roughly half of `q8_0`. Measure
rather than trust a table, and remember the actual requirement:

> **A single generation may not exceed `--kvmem-gen-reserve`.** Anything larger
> aborts the request.

So `--kvmem-gen-reserve` sets the longest output you can produce in one go, while
`--kvmem-budget` sets how much history stays retrievable on the GPU. The full
context lives in host memory; raising `-c` therefore costs RAM, not VRAM.

### The idle-VRAM trap

Staging buffers are allocated on first use, not at startup. A configuration can
look comfortable while idle and die on the first request:

```
KVMEM stagein q scratch: out of memory
ggml-cuda.cu:108: CUDA error: out of memory
```

**Never treat a passed health check as evidence that the budget is fine.** Fire
one real request (step 7) before reporting success.

Recovery order when it does happen: lower `--kvmem-budget` first, then reduce
`-ub` (e.g. `128` → `64`). Reducing `-b` is not a fix for this.

---

## 6. Launch

```bash
build/bin/llama-kvmem-server \
  -m /path/to/model.gguf \
  --host 127.0.0.1 --port 18200 --alias kvmem \
  -ngl 99 -fa on -np 1 -t 8 -tb 8 -b 512 -ub 128 \
  -c 262144 \
  --kvmem-budget 20480 --kvmem-gen-reserve 8192 --kvmem-block-tokens 32
```

Points that bite:

* `-ngl` accepts a non-negative count or `all`; automatic GPU fitting
  (`auto` / `-1`) is not implemented and errors out.
* `-np` only accepts `1`.
* Multi-GPU is not supported: pick one device with `--device CUDA0`,
  `--split-mode none --main-gpu INDEX`, or `CUDA_VISIBLE_DEVICES`.
* `-n` / `--n-predict` defaults to `-1` (no extra cap); generation is still
  bounded by EOS, the context and the KVMem generation reserve.

**Pass criterion:** the log line `listening on http://127.0.0.1:<port>` appears.
A bind failure prints an error and does not print a listening message.

### Startup diagnostics worth reading

With `--kvmem-trace` (or `KVMEM_TRACE=1`):

```
KVMEM_STARTUP requested=...     what was asked for, before loading weights
KVMEM_STARTUP ready=...         actual ctx/batch/threads, output limits, MTP state
```

The server validates configuration, files and incompatible settings **before**
loading weights, which is why bad flags fail fast.

---

## 7. Verify with a real request

```bash
curl -s http://127.0.0.1:18200/health

curl -s -X POST http://127.0.0.1:18200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kvmem","messages":[{"role":"user","content":"Say hello in five words."}],
       "max_tokens":64,"stream":false}'
```

**Pass criteria:** HTTP 200 and `usage.completion_tokens > 0`. A health check
alone does not exercise the KV pool, the staging path or the kernels, so it does
not prove the deployment works.

Then run the project's own regression scripts:

```bash
python scripts/test_server_compat.py --server build/bin/llama-kvmem-server
python scripts/test_server_environment.py --server build/bin/llama-kvmem-server --output /tmp/kvmem-env
```

Add `--model PATH` for live inference and auth checks, `--mtp` for MTP, and
`--mmproj PATH --image PATH` for the vision fixture.

Ground truth for correctness: **run one instruction-following prompt and read the
output.** A low-bit model can build cleanly, pass the health check and still emit
garbage if the toolkit was wrong.

---

## 8. Troubleshooting index

| Symptom | Likely cause | Action |
|---|---|---|
| `KVMEM stagein q scratch: out of memory` → `CUDA error` | pool too large for the card | lower `--kvmem-budget`, then `-ub` to 64 |
| Garbage / repeated tokens | toolkit too old, or wrong CUDA architecture | nvcc ≥ 13.2.86, fresh build dir, correct arch |
| `'token_embd.weight' is read without the inverse transform` | a patch that touches the embedding lookup is missing | re-run `scripts/apply-patches.sh` |
| `GGML_ASSERT(gdn->op == GGML_OP_GATED_DELTA_NET)` | MTP state mode incompatible with the hybrid model | use `--kvmem-mtp-state snapshots` |
| `llama.cpp differs from the supported pin ...` | local edits in the submodule | restore the submodule, keep changes in `patches/` |
| Bind failure on the port | another instance holds it | stop it, or change `--port` |
| Long context works, one long answer fails | single generation exceeded `--kvmem-gen-reserve` | raise the reserve, or split the generation |

---

## 9. Invariants an agent must not violate

1. **The submodule stays clean.** Changes to `llama.cpp` go into `patches/` and
   are replayed by `scripts/apply-patches.sh`.
2. **A fresh build directory after any toolkit change.**
3. **`--kvmem-gen-reserve` bounds one generation**; keep it above whatever
   per-round output budget is in use.
4. **Idle VRAM proves nothing.** Only a real request does.
5. **Read the output, not just the exit code.**

---

## 10. Optional: server-side auto-continue-thinking

This tree also carries an opt-in loop that keeps a reasoning chain in context when
a round hits its output budget and starts the next round automatically, instead of
returning a truncated answer. Flags, sizing rules and the diagnostics to look for
are in [auto-continue-thinking.md](auto-continue-thinking.md). Agent-relevant
points: `--kvmem-gen-reserve` must stay above `--act-round-tokens`, and
`--act-prefill-mode auto` is required for the loop to work with the default
`user` query policy.
