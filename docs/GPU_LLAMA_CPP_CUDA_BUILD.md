# Building `llama-cpp-python` with CUDA (GGUF describer GPU acceleration)

> **Audience:** maintainers and AI agents working on this repo. This is a
> toolchain note, deliberately **not** in `README.md`. It documents a
> non-obvious environment workaround that is required to make the GGUF scene
> describer (`src/kuaa/models/describer/gguf.py`) run on the GPU instead
> of CPU. Read this end-to-end before touching the CUDA toolchain, the NVIDIA
> driver, the kernel, glibc, or `llama-cpp-python`.

Last verified: 2026-05-18, on Fedora 44 with a Blackwell (`sm_120`) GPU, CUDA
13.0.2, glibc 2.43.

> **Re-verify before relying on this.** This procedure was last verified on
> 2026-05-18 against one specific toolchain combination (Fedora 44, a
> Blackwell / `sm_120` GPU, CUDA 13.0.2, glibc 2.43, `gcc15`). GPU archs,
> distro package versions, and CUDA/glibc releases move fast; do not assume
> this still applies byte-for-byte on a different GPU, distro, or CUDA version.
> Re-check the toolchain versions in §2 against what is actually installed
> before following the patch/build steps on a new setup.

---

## 1. Why this is needed (the problem)

The scene describer runs Moondream 2 as a GGUF model via `llama-cpp-python`.
On CPU a full Jeca Tatu run (412 scenes × 6 prompts) takes **~5–6 hours**.
On a Blackwell (`sm_120`) GPU the same work is **~15–25 minutes** (~10–25×
faster).

To use the GPU, `llama-cpp-python` must be compiled **from source with CUDA**
(`-DGGML_CUDA=on`) and the model layers offloaded (`n_gpu_layers != 0`). The
prebuilt wheels do **not** work here:

- PyPI `llama-cpp-python` ships **CPU-only** wheels (`llama_supports_gpu_offload()`
  returns `False`).
- The community CUDA wheel index (`abetlen.github.io/.../whl/cuXXX`) only builds
  for older GPU archs. Blackwell GPUs are **compute capability 12.0
  (`sm_120`)**; those wheels have no `sm_120` cubin/PTX → `no kernel image
  available` at runtime.
- There is **no real `nvidia-cuda-nvcc-cu13` PyPI wheel** (the package is a
  `0.0.1` stub); the `cu12` line cannot target Blackwell.

So we must build from source. That build hits **two** toolchain walls on this
bleeding-edge distro:

### Wall 1 — host compiler too new (solved by installing `gcc15`)

CUDA 13.0's `nvcc` rejects host GCC newer than 15. Fedora 44 ships **GCC 16**,
whose `libstdc++` headers (`/usr/include/c++/16/...`) `nvcc` cannot parse, even
with `-allow-unsupported-compiler` (it then fails with ~100 `type_traits`
errors). Fix: install the parallel-installable `gcc15` / `gcc15-c++` packages
and point `nvcc` at `g++-15` as the CUDA host compiler. `gcc15` ships its own
`/usr/include/c++/15` headers and does **not** disturb the system GCC 16.

### Wall 2 — glibc 2.43 vs CUDA 13.0.2 `rsqrt` clash (solved by the header patch)

glibc 2.41+ added C23 IEC-60559 math, including `rsqrt`/`rsqrtf`, declared in
`/usr/include/bits/mathcalls.h` with `noexcept (true)`. The block is gated by
`#if __GLIBC_USE (IEC_60559_FUNCS_EXT_C23)`, which is force-enabled because
llama.cpp's host code is compiled with `_GNU_SOURCE` (→ `__USE_GNU`) and the
flag is recomputed by `<bits/libc-header-start.h>` on every header include, so
there is **no clean feature-test-macro override**.

CUDA 13.0.2's `crt/math_functions.h` independently declares device builtins
`rsqrt(double)` / `rsqrtf(float)` **without** an exception specification. C++
treats two declarations of the same function with different exception specs as
a hard error:

```
/usr/include/bits/mathcalls.h(206): error: exception specification is
incompatible with that of previous function "rsqrt" (declared at line 629 of
.../crt/math_functions.h)
```

NVIDIA fixed this in **CUDA 13.1** by adding the matching `noexcept (true)` to
those two declarations. CUDA 13.1 is **not obtainable on this machine** (the
configured CUDA repo `cuda-fedora42-13-0-local` is an offline local mirror that
only carries 13.0.2). The patch below applies NVIDIA's own 13.1 fix by hand to
the 13.0.2 header.

---

## 2. Prerequisites

| Requirement | Reference setup | Install if missing |
|---|---|---|
| NVIDIA GPU, Blackwell `sm_120` | e.g. RTX 5090 | — |
| NVIDIA driver supporting CUDA 13 | 595.71.05 | distro/NVIDIA |
| CUDA 13.0 toolkit (`nvcc`) | `/usr/local/cuda-13.0` (13.0.2, nvcc build 88) | `sudo dnf install cuda-toolkit-13-0` |
| GCC 15 host compiler | `/usr/bin/gcc-15`, `/usr/bin/g++-15` (15.2.1) | `sudo dnf install -y gcc15 gcc15-c++` |
| `uv` + project venv | `.venv` (Python 3.11) | `uv venv && uv sync --extra full --group dev` |

The describer already has the GPU knob wired: `config/default.yaml` →
`llm.gpu_layers: -1` (offload all layers). `-1` is a **no-op on a CPU-only
build**, so this config is safe to keep regardless of which `llama-cpp-python`
is installed. See `MoondreamGGUFDescriber.__init__` / `_load_model` in
`src/kuaa/models/describer/gguf.py`.

---

## 3. How to apply the CUDA header patch

The file is **root-owned and read-only**, so this needs `sudo`. The command is
**idempotent** (safe to re-run) and makes a **one-time backup** the first time.

```bash
F=/usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h
sudo test -e "$F.bak-kuaa" || sudo cp "$F" "$F.bak-kuaa"
sudo sed -i \
  -e 's/rsqrt(double x);$/rsqrt(double x) noexcept (true);/' \
  -e 's/rsqrtf(float x);$/rsqrtf(float x) noexcept (true);/' \
  "$F"
grep -nE "rsqrtf?\(.* x\) noexcept" "$F"
```

Expected output (lines 629 and 653, exact whitespace preserved):

```
629:extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x) noexcept (true);
653:extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x) noexcept (true);
```

Why idempotent: after patching, the line ends with `noexcept (true);`, so the
`...(double x);$` / `...(float x);$` patterns no longer match — a second run is
a no-op. The backup is only taken if `$F.bak-kuaa` does **not** already
exist, so re-running never clobbers a clean backup with a patched file.

Optional fast sanity check that the clash is gone (no full build needed):

```bash
printf '#include <cmath>\n__global__ void k(float*a){*a=rsqrtf(*a);}\nint main(){return 0;}\n' > /tmp/p.cu
/usr/local/cuda-13.0/bin/nvcc -ccbin /usr/bin/g++-15 -arch=sm_120 -c /tmp/p.cu -o /tmp/p.o && echo "PROBE PASS"
rm -f /tmp/p.cu /tmp/p.o
```

---

## 4. How to build `llama-cpp-python` with CUDA

Run from the repo root, in the project venv. This pins the whole CUDA build to
the `gcc-15` toolchain and targets `sm_120` only (smaller/faster build). It is
a from-source build (`--no-binary`, `--no-cache`, `--reinstall`) and takes
**~10–15 minutes**.

```bash
cd /mnt/a/projects/kuaa
CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 \
CUDACXX=/usr/local/cuda-13.0/bin/nvcc \
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15" \
FORCE_CMAKE=1 \
PATH="/usr/local/cuda-13.0/bin:$PATH" \
UV_LINK_MODE=copy \
uv pip install --no-cache --reinstall --no-binary llama-cpp-python "llama-cpp-python>=0.3,<0.4"
```

Notes:
- `--no-cache` is **required**: uv caches built wheels by version, so without
  it `CMAKE_ARGS` is ignored and the stale CPU wheel is reused (the build
  "succeeds" in <1 s and stays CPU-only).
- Targeting only `sm_120` is intentional. Add more archs to
  `-DCMAKE_CUDA_ARCHITECTURES` (e.g. `120;90`) only if other GPUs must be
  supported by the same wheel.

### Verify the build is GPU-capable

```bash
.venv/bin/python -c "import llama_cpp, llama_cpp.llama_cpp as L; \
print(llama_cpp.__version__, 'gpu_offload =', L.llama_supports_gpu_offload())"
# Expect: 0.3.x gpu_offload = True
```

When the describer loads the model it logs the offload setting:

```
✓ Moondream GGUF carregado em N.Ns (n_gpu_layers=-1)
```

and `nvidia-smi` should show the Python process holding VRAM during a run.

---

## 5. How to run the pipeline on GPU

Nothing extra to do — `config/default.yaml` already sets `llm.gpu_layers: -1`.
Once a CUDA `llama-cpp-python` is installed, the existing command uses the GPU
automatically:

```bash
.venv/bin/python -m kuaa process --video data/raw/<film>.mp4 --steps llm
```

**Run with `.venv/bin/python`, not `uv run`,** for long jobs — see the hazard
in §7. To force CPU even on a GPU build, set `llm.gpu_layers: 0` in
`config/local.yaml`.

The existing 412-scene Jeca Tatu artefacts were generated on **CPU with the
f16 GGUF model**. A GPU rerun uses the **same f16 weights**, so descriptions
are numerically equivalent (greedy decode); regenerating is **not required**
for consistency — GPU only matters for *future* processing speed.

---

## 6. How to revert

Restore the stock CUDA header and (optionally) go back to the CPU wheel:

```bash
F=/usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h
sudo cp "$F.bak-kuaa" "$F"          # undo the patch

# optional: reinstall the stock CPU wheel
UV_LINK_MODE=copy uv pip install --reinstall --only-binary llama-cpp-python \
  "llama-cpp-python>=0.3,<0.4"
```

The patch is purely additive (`noexcept (true)` matches glibc and is what
CUDA 13.1 ships), so reverting is only needed if you suspect it or want a
pristine toolchain. Leaving it in place is harmless for other CUDA builds.

---

## 7. Maintenance: what breaks this, and what to do

The patch lives **outside the repo**, on a root-owned system file, and the
CUDA `llama-cpp-python` is a from-source build in `.venv`. Several routine
system actions silently undo one or both. Symptoms of a lost GPU build:
`llama_supports_gpu_offload()` flips to `False`, or the describer logs
`n_gpu_layers=-1` but `nvidia-smi` shows no Python VRAM and runs are slow again.

| Event | Effect | Action |
|---|---|---|
| **NVIDIA driver update** | Usually fine — driver is forward-compatible; `sm_120` keeps working; no rebuild needed. If the driver is downgraded below CUDA 13 support, GPU init fails. | Verify `nvidia-smi` works; no rebuild normally required. |
| **Kernel update** | Irrelevant to the userspace CUDA build and the header patch. Only the NVIDIA kmod must match the driver (DKMS/akmod handles this); a broken kmod means no GPU at all, not a llama-cpp problem. | Ensure `nvidia-smi` works after reboot. Nothing to rebuild. |
| **`cuda-toolkit-13-0` reinstall/update (still 13.0.x)** | The system header is **overwritten → patch lost**. `$F.bak-kuaa` is a separate file and survives unless the dir is fully removed. | Re-run §3 (idempotent), then **rebuild** §4. |
| **CUDA upgraded to ≥ 13.1** | NVIDIA's own fix is present → **patch no longer needed**. A new toolkit dir (e.g. `/usr/local/cuda-13.1`) is unpatched and correct. | Drop the patch; point `CUDACXX`/`PATH` at the new toolkit dir and rebuild §4. Remove the stale `.bak-kuaa`. |
| **glibc update** | If glibc changes `mathcalls.h` again, **new** symbol clashes may appear (this patch only covers `rsqrt`/`rsqrtf`). If glibc later drops/relocates the C23 block, the clash disappears and the patch becomes a harmless no-op. | If the build fails with a new `exception specification is incompatible` on a different symbol, extend the §3 `sed` with the same `noexcept (true)` treatment for that symbol. |
| **`gcc15` removed / GCC bumped past 15 with no `gcc15`** | Build fails at the compiler-id stage (host compiler too new). | `sudo dnf install -y gcc15 gcc15-c++` and rebuild §4. |
| **`uv sync` / `uv run` / `uv pip install` (any)** | uv reconciles `.venv` to `pyproject.toml` and **replaces the CUDA build with the cached CPU wheel** — this is the most common silent regression. | For long GPU jobs invoke `.venv/bin/python -m kuaa ...` directly. After any `uv sync`/`uv run`, re-verify §4's check and rebuild if `gpu_offload` is `False`. |
| **`llama-cpp-python` version bump in `pyproject.toml`** | Next `uv sync` installs the CPU wheel of the new version. | Rebuild §4 against the new version (keep the `>=0.3,<0.4` range in sync, or widen as needed). |
| **numpy version on this GPU box** | The §4 re-resolve floats numpy to 2.x; `pyproject.toml` pins `numpy>=1.24,<2` (deliberate — protects the default torch-2.2 CLIP path, commit `e3ed9da`). A `uv sync` downgrades numpy back to 1.26.4. | **Harmless to the GPU stack** — torch 2.12+cu130 supports numpy 1.x *and* 2.x. The only side effect of that `uv sync` is the llama-cpp CPU-wheel clobber above; `ensure_gpu_llama.sh` repairs it and its rebuild re-floats numpy to 2.x. No action needed; do **not** "fix" the pyproject pin to `>=2`. |
| **OS upgrade (Fedora bump)** | New GCC/glibc/CUDA — re-evaluate from scratch. Prefer installing **CUDA ≥ 13.1** and dropping the patch entirely. | Re-derive prerequisites (§2); patch only if the toolkit is still < 13.1 and the glibc clash recurs. |

### Safety nets (so this can't silently regress)

Two mechanisms make the silent-CPU-fallback loud and one-command-fixable:

- **Runtime warning.** `MoondreamGGUFDescriber._warn_if_cpu_build()` logs a
  loud `WARNING` at model load when GPU offload is requested and an NVIDIA GPU
  is present but the installed `llama-cpp-python` is CPU-only. You will see it
  in pipeline logs immediately, not after a 6-hour run.
- **`scripts/ensure_gpu_llama.sh`.** Idempotent: instant no-op if already
  GPU-capable; otherwise checks the toolchain/patch preconditions (printing the
  exact sudo command if the header is unpatched) and runs the §4 build. Run it
  before GPU jobs and after any `uv sync`/`uv run`.

### Rule of thumb

Whenever you touch CUDA, glibc, GCC, or reinstall/upgrade `llama-cpp-python`,
just run `./scripts/ensure_gpu_llama.sh` — it encodes the steps below:

1. Re-apply §3 (idempotent — harmless if already patched or unneeded).
2. Rebuild §4.
3. Run the §4 verification (`gpu_offload = True`).
4. Confirm a real run shows Python VRAM in `nvidia-smi`.

If CUDA is ever ≥ 13.1, skip §3 entirely — the patch is obsolete.
