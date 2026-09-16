# KVMem Linux / WSL2 diagnostic build

Experimental precompiled build for issue #1. Target: NVIDIA `sm_120a`; host: Ubuntu 22.04-compatible x86_64, glibc >= 2.35, AVX2/FMA/F16C/BMI2 CPU. Python 3.10+, Bash and `ss` (iproute2) are needed for recipe launchers. A compatible NVIDIA driver is required; CUDA Toolkit is not needed. This is not a native Windows binary.

`BUILD-INFO.json` and `provenance/source-manifest.json` identify the exact source and compiler. The matching source archive is supplied separately. Bundled CUDA runtime and cuBLAS libraries are under `lib/`, with their terms under `licenses/`; the NVIDIA driver and host C/C++ libraries are supplied by the OS. No model weights are included.

Verify from the extracted directory:

```bash
sha256sum -c SHA256SUMS
bin/llama-kvmem-server --help
```

Set GPU selection explicitly on multi-GPU machines. Use your actual GGUF paths:

```bash
CUDA_VISIBLE_DEVICES=0 MODEL=/path/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  MMPROJ=/path/mmproj-Q8_0.gguf scripts/start-iq3.sh

CUDA_VISIBLE_DEVICES=0 MODEL=/path/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf \
  MMPROJ=/path/mmproj-BF16.gguf scripts/start-iq4.sh
```

Choose one service on port 18200; add `--restart` to switch this package's service. An unrelated listener is never stopped. `--dry-run` previews arguments. Both recipes use MTP3/ReplaySSM; IQ3 uses GPU Q8 vision and Q8 main KV; IQ4 uses CPU BF16 vision and Q5 main KV. To match the current local experiment, add `--kvmem-block-tokens 32 --reasoning-effort medium`; recipe defaults remain block 128 and the model's effort default.

## Issue #1 short-prompt comparison

Run on a GPU with enough free VRAM. The script does not stop an existing server; choose a free range starting at `--port` (default 18250), or stop your model service first. Each case uses the next port to avoid reuse delays. It loads models sequentially and terminates only its own subprocesses.

```bash
python3 scripts/prebuilt/diagnose.py --model /path/model.gguf \
  --gpu 0 --output diagnostic-results
```

The exact question is `What is 2+3? Answer with the number only.`, temperature 0, max_tokens 32. Four cases compare plain F16/no KVMem/no MTP, retrieval/no MTP, MTP2 snapshots and MTP2 ReplaySSM. Each uses a fresh server and 32K context; retrieval cases use an 8K budget plus 512-token reserve for this short-input isolation test. These deliberately are not the full 256K performance recipes. No projector is loaded. No tools are executed.

`--cases plain retrieval` selects a subset. ReplaySSM requires the supported Qwen 27B CUDA layout; unsupported models should use the other cases. Add `--hash-model` to record the GGUF SHA-256 (reads the entire model).

Send `summary.json`, response files, command files and stderr logs when reporting results. `thinking_only_at_limit` means the 32-token output allowance was consumed by reasoning; it does not mean garbage output. Other unexpected text is classified `needs_review`, not automatically as a numerical failure. The diagnostic pauses on process swap >=512 MiB or system swap growth >=1 GiB.

## Rebuilding

Extract the matching source archive and run `CUDA_HOME=/path/to/cuda bash source/scripts/prebuilt/build.sh`. This uses four compilation jobs by default (`JOBS` overrides it). `CMAKE` and `BUILD_DIR` can also be overridden. See `scripts/prebuilt/package.py --help` in the source snapshot for packaging, license and provenance inputs.
