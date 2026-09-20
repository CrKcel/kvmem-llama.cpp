# Optional Windows quantizer — v0.16.0-rc3 — CUDA 12.9.86 experimental build

This package uses CUDA 12.9.86 with GPU targets 70/75/80/86/89/90/120a. Consult the bundled validation report for actual hardware and inference coverage. It is separate from the CUDA 13.2 package.

This package contains `bin/llama-quantize.exe` and any discovered application
dependencies. It converts GGUF model weights; it does not run the chat server
and is not the same as the server's runtime `-ctk/-ctv` KV-cache settings.

If you downloaded prepared model files, you do not need this package. Ordinary
users should download the separate Windows runtime ZIP instead.

Use PowerShell in the extracted directory:

```powershell
.\bin\llama-quantize.exe --help
.\bin\llama-quantize.exe --max-buffer-size 256 'D:\models\input-BF16.gguf' 'D:\models\output-Q8_0.gguf' Q8_0
```

The example above illustrates generic GGUF conversion. The recommended IQ3 main model and vision projector are already quantized; download them using the links in the runtime README.

Requires Windows x64 and Microsoft Visual C++ x64 runtime. CPU target:
AVX2/FMA/F16C/BMI2. Any CUDA dependencies are bundled; no model files are included.
See BUILD-INFO.json and licenses/ for build information and licenses.
