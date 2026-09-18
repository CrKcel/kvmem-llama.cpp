# Optional Windows quantizer — v0.16.0-rc2

This package contains `bin/llama-quantize.exe` and any discovered application
dependencies. It converts GGUF model weights; it does not run the chat server
and is not the same as the server's runtime `-ctk/-ctv` KV-cache settings.

If you downloaded prepared model files, you do not need this package. Ordinary
users should download the separate Windows runtime ZIP instead.

Use PowerShell in the extracted directory:

```powershell
.\bin\llama-quantize.exe --help
.\bin\llama-quantize.exe --max-buffer-size 256 'D:\models\mmproj-BF16.gguf' 'D:\models\mmproj-Q8_0.gguf' Q8_0
```

For the selective IQ4 MTP conversion, follow the repository README and use its
tensor map and imatrix file. Do not blanket-requantize the model as a substitute.

Requires Windows x64 and Microsoft Visual C++ x64 runtime. CPU target:
AVX2/FMA/F16C/BMI2. Any CUDA dependencies are bundled; no model files are included.
The executable has passed a launch/help check; this rc2 publication did not run
a fresh full-model quantization. See BUILD-INFO.json and licenses/ for provenance.
