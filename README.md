# diffusers-rocm

<a href="https://github.com/lemonade-sdk/diffusers-rocm/releases/latest" title="Download the latest release">
  <img src="https://img.shields.io/github/v/release/lemonade-sdk/diffusers-rocm?logo=github&logoColor=white" alt="GitHub release (latest by date)" />
</a>
<a href="https://github.com/lemonade-sdk/diffusers-rocm/releases/latest" title="View latest release date">
  <img src="https://img.shields.io/github/release-date/lemonade-sdk/diffusers-rocm?logo=github&logoColor=white" alt="Latest release date" />
</a>
<a href="LICENSE" title="View license">
  <img src="https://img.shields.io/github/license/lemonade-sdk/diffusers-rocm?logo=opensourceinitiative&logoColor=white" alt="License" />
</a>
<a href="https://github.com/ROCm/ROCm" title="Powered by ROCm 7.12">
  <img src="https://img.shields.io/badge/ROCm-7.12-blue?logo=amd&logoColor=white" alt="ROCm 7.12" />
</a>
<a href="https://github.com/huggingface/diffusers" title="Powered by diffusers">
  <img src="https://img.shields.io/badge/Powered%20by-diffusers-yellow" alt="Powered by diffusers" />
</a>
<a href="#supported-devices" title="Platform support">
  <img src="https://img.shields.io/badge/OS-Ubuntu-0078D6?logo=ubuntu&logoColor=white" alt="Platform: Ubuntu" />
</a>

Portable builds of HuggingFace **diffusers** with **AMD ROCm 7.12** acceleration. Each release is a self-contained archive that bundles a relocatable CPython interpreter, diffusers + transformers + accelerate, PyTorch ROCm, the required ROCm user-space libraries (as pip packages), and a small OpenAI-compatible HTTP server (`diffusers-server`) — no system Python, PyTorch, or ROCm install required. Automated pipeline targets integration with [**Lemonade**](https://github.com/lemonade-sdk/lemonade).

> [!IMPORTANT]
> **Early Development**: This project is in active development. ROCm support for consumer AMD GPUs (RDNA) is still maturing. We welcome issue reports and contributions.

## Supported Devices

| GPU Target | Architecture | Devices |
|------------|-------------|---------|
| **gfx1151** | STX Halo APU | Ryzen AI MAX+ Pro 395 |
| **gfx1150** | STX Point APU | Ryzen AI 300 |
| **gfx120X** | RDNA4 GPUs | RX 9070 XT, RX 9070, RX 9060 XT, RX 9060 |
| **gfx110X** | RDNA3 GPUs | RX 7900 XTX/XT/GRE, RX 7800 XT, RX 7700 XT, RX 7600 XT/7600 |

All builds include ROCm 7.12 user-space built-in — no separate ROCm installation required. You still need a Linux kernel with a working amdgpu driver for your GPU; for gfx1151 specifically this means kernel 6.18.4+ (see [Lemonade's gfx1151 notes](https://lemonade-server.ai/gfx1151_linux.html)).

## Quick Start

1. **Download** the build for your GPU from the [latest release](https://github.com/lemonade-sdk/diffusers-rocm/releases/latest). Larger releases are split into `.partNN-of-MM.tar.gz` parts because they exceed GitHub's 2 GB per-asset limit.
2. **Extract**:
   ```bash
   mkdir -p ~/diffusers-rocm
   # Single-archive releases:
   tar xzf diffusers0.35.0-rocm7.12.0-gfx1151-x64.tar.gz -C ~/diffusers-rocm
   # Multi-part releases:
   cat diffusers0.35.0-rocm7.12.0-gfx1151-x64.part01-of-02.tar.gz \
       diffusers0.35.0-rocm7.12.0-gfx1151-x64.part02-of-02.tar.gz \
     | tar xz -C ~/diffusers-rocm
   ```
3. **Run** the server:
   ```bash
   ~/diffusers-rocm/bin/diffusers-server \
     --model unsloth/FLUX.1-schnell-GGUF \
     --port 8000
   ```
4. **Generate an image**:
   ```bash
   curl http://localhost:8000/v1/images/generations \
     -H "Content-Type: application/json" \
     -d '{
       "model": "unsloth/FLUX.1-schnell-GGUF",
       "prompt": "a red apple on a wooden table",
       "n": 1,
       "size": "1024x1024"
     }'
   ```

> **Lemonade Integration**: These builds are designed to work as a backend for [**Lemonade**](https://github.com/lemonade-sdk/lemonade), which manages downloading, launching, and routing requests automatically.

## What's Included

Each release archive extracts to a relocatable CPython 3.12 distribution:

```
bin/
  diffusers-server            # Launcher shim (sets LD_LIBRARY_PATH, execs the HTTP service)
  python3.12                  # Bundled CPython (python-build-standalone)
lib/
  libpython3.12.so
  python3.12/
    site-packages/
      diffusers/              # pip from PyPI
      transformers/           # pip from PyPI
      accelerate/, peft/, safetensors/, gguf/, ...
      torch/                  # pip from repo.amd.com/rocm/whl/<arch>/
      _rocm_sdk_core/lib/     # ROCm core user-space (hip, hsa, comgr, clang, llvm)
      _rocm_sdk_libraries_gfx<arch>/lib/
                              # Per-arch ROCm math libs (rocblas, hipblas, MIOpen, ...)
      diffusers_server/       # The OpenAI-compatible HTTP server in this repo
```

## API

`diffusers-server` exposes OpenAI-compatible endpoints:

- `POST /v1/images/generations` — text-to-image (sync)
- `GET /health` — liveness probe
- *(planned phase 2)* `POST /v1/videos/generations` + `GET /v1/videos/generations/{id}` — async video gen
- *(planned phase 3)* `POST /v1/images/edits`, `POST /v1/images/variations`

## Automated Builds

The GitHub Actions workflow:
- Downloads relocatable **CPython 3.12** from [`astral-sh/python-build-standalone`](https://github.com/astral-sh/python-build-standalone)
- Installs **PyTorch ROCm** from AMD's per-arch pip index (`https://repo.amd.com/rocm/whl/<target>/`), which pulls `rocm-sdk-core` and `rocm-sdk-libraries-gfx<target>` as transitive deps
- Installs **diffusers** + **transformers** + **accelerate** + **peft** + **safetensors** + **gguf** + **huggingface_hub** + **Pillow** + **fastapi** + **uvicorn** from PyPI
- Installs this repo's `diffusers-server` package into site-packages
- Generates a `bin/diffusers-server` shim that wires up `LD_LIBRARY_PATH` at startup
- Tars the result, splits into `< 2 GB` parts, uploads to GitHub Releases

## Dependencies

### Runtime (bundled)
- **[diffusers](https://github.com/huggingface/diffusers)** — HuggingFace diffusion model library (pure Python)
- **[PyTorch](https://pytorch.org/)** — tensor compute (ROCm wheel)
- **[ROCm SDK wheels](https://github.com/ROCm/TheRock)** — AMD's pip-packaged ROCm user-space
- **[python-build-standalone](https://github.com/astral-sh/python-build-standalone)** — relocatable CPython 3.12

### Build (CI only)
- **Ubuntu 22.04** GitHub Actions runner
- `pip` (no compilation, no `cmake`, no `patchelf` — everything is pre-built wheels)

## License

MIT — see [LICENSE](LICENSE).
