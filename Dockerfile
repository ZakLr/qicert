# qicert dev container — Decision 2026-08-16 (chair + user)
#
# Derived FROM the user's existing CUDA-Q image (nvcr.io/nvidia/nightly/
# cuda-quantum:cu13-latest) so ONE container carries BOTH:
#   * CUDA-Q (statevector/tensor-network sims, noise models)  — cu13, GPU-ready
#   * the ML stack for the robotics-first baselines (torch cu13 wheels,
#     transformers/peft/accelerate for MiniVLA-1B LoRA fine-tuning)
# Target: RTX 5060 Laptop (8 GB, Blackwell sm_120), Windows Docker Desktop WSL2.
#
# Build:
#   docker build -t qicert-dev:cu13-cudaq -f Dockerfile .
# Run (repo mounted, GPUs passed through):
#   docker run --gpus all -it --rm -v "$PWD":/workspace qicert-dev:cu13-cudaq bash
#   # or non-interactive, e.g.:
#   docker run --gpus all --rm -v "$PWD":/workspace qicert-dev:cu13-cudaq \
#       python -m qicert.bench.all --rows=smoke
#
# Notes:
#   - Deliberately NO bitsandbytes in v1 (Blackwell sm_120 pitfalls); the
#     INT8 reference uses torch.quantization (Decision 2026-08-16). MiniVLA-1B
#     fits fp16/bf16 LoRA at 8 GB without quantization.
#   - Freeze exact pins into environment.yml the moment this image works
#     (Q19 discipline applies to the container too; never drift).

FROM nvcr.io/nvidia/nightly/cuda-quantum:cu13-latest

# The CUDA-Q image prints a banner via ENTRYPOINT ["bash","-l"]; bench/CI runs
# must exec cleanly, so clear it. NOTE: the WORKDIR contains a `python/` dir, so
# bare `python` resolves to that directory at runtime — invoke /usr/bin/python3
# explicitly (Git Bash on Windows: prefix MSYS_NO_PATHCONV=1 for /-paths).
# Base image is Ubuntu 24.04 with python3.12 (verified: numpy 2.4.6, scipy 1.17.1).
ENTRYPOINT []
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# The CUDA-Q base image's default USER is non-root (sudo available); the build
# steps need write access to /opt and the venv, so build as root. Runtime
# behaviour for `docker run` is unaffected.
USER root

WORKDIR /workspace

# Reuse the base image's python3.12 and its numpy/scipy (present, verified);
# the base lacks python3.12-venv, so install straight into the system python
# as root (the container is ephemeral; runtime behaviour is unaffected).
# PyTorch with CUDA 13 wheels (sm_120 / Blackwell, matches the base CUDA 13).
# `--index-url` selects the cu130 wheel channel; torch >= 2.9 ships cu130
# wheels for python 3.12. numpy/scipy come from the base image (no rebuild).
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
        "torch>=2.9" \
    && python -c "import torch; print('torch', torch.__version__); print('arch', torch.cuda.get_arch_list())"

# qicert package + tests + ML stack (robotics-first: MiniVLA-1B deps).
COPY pyproject.toml README.md LICENSE ./
COPY python ./python
COPY bench ./bench
COPY tests ./tests
COPY cert ./cert
COPY docs ./docs
RUN pip install -e ".[dev]" \
    && pip install "transformers>=4.44" "peft>=0.13" "accelerate>=1.0" \
                   "safetensors>=0.4" "sentencepiece>=0.2" "huggingface_hub>=0.24" \
                   "pynvml>=11.0"   # recorder: GPU power/util capture (optional)

# Fast CI check: the smoke suite must pass inside the container, and both the
# ML runtime and CUDA-Q must be importable. NOTE: the CUDA-Q 'nvidia' target is
# NOT exercised here — libnvidia-ml is only mounted at `docker run --gpus all`
# time, not during build. Runtime verification is done on first run:
#   docker run --gpus all --rm qicert-dev:cu13-cudaq \
#       python -c "import cudaq; cudaq.set_target('nvidia'); print(cudaq.get_target())"
RUN python -m pytest tests/ -x -q \
    && python -m qicert.bench.all --rows=smoke \
    && python -c "import cudaq; print('CUDA-Q import OK')"

CMD ["bash"]
