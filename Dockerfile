# qicert dev container — Decision 2026-08-11 (chair)
#
# Purpose: the single reproducible dev runtime for the RTX 5060 (Blackwell,
# sm_120) machine. Robotics-first track (MiniVLA-1B), Python-first reference
# kernels. CUDA 13 base with PyTorch cu13 wheels.
#
# Build:
#   docker build -t qicert-dev -f Dockerfile .
# Run (repo mounted, GPUs passed through):
#   docker run --gpus all -it --rm -v "$PWD":/workspace qicert-dev bash
#
# Notes:
#   - Deliberately NO bitsandbytes in v1 (Blackwell sm_120 pitfalls, Q23-ish);
#     MiniVLA-1B fits fp16/bf16 LoRA at 8 GB without quantization.
#   - Freeze exact pins into environment.yml the moment the image works
#     (Q19 discipline applies to the container too; never drift).

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# System deps: python 3.11 (>=3.11 required by pyproject), build tools for the
# C++ kernel bindings when they land (Q19), git for huggingface_hub.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ENV VENV=/opt/venv
RUN python3.11 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# PyTorch with CUDA 13 wheels (sm_120 / Blackwell). The `--index-url` selects
# the cu13 wheel channel; torch >= 2.9 ships cu13 wheels.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu130 \
        "torch>=2.9" \
    && python -c "import torch; print('torch', torch.__version__); print('arch', torch.cuda.get_arch_list())"

# qicert package + tests + ML stack (robotics-first: MiniVLA-1B deps).
# transformers + peft + accelerate for the VLAM fine-tune path; numpy/scipy
# are the reference-kernel floor (CI/bench never import torch).
COPY pyproject.toml README.md LICENSE ./
COPY python ./python
COPY bench ./bench
COPY tests ./tests
RUN pip install -e ".[dev]" \
    && pip install "transformers>=4.44" "peft>=0.13" "accelerate>=1.0" \
                   "safetensors>=0.4" "sentencepiece>=0.2" "huggingface_hub>=0.24"

# Fast CI check: the smoke suite must pass inside the container.
RUN python -m pytest tests/ -x -q && python -m qicert.bench.all --rows=smoke

CMD ["bash"]
