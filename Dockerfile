FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta2

SHELL ["/bin/bash", "-lc"]

ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
ARG CONDA_DIR=/opt/miniconda3
ARG CUDA_ARCH_LIST=12.0
ARG INSTALL_FLASH_ATTN=1
ARG INSTALL_WORLDGEN_EXTRAS=1

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=${CONDA_DIR}
ENV HYWORLD_ROOT=/workspace/hyworld2
ENV HF_HOME=/models/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/models/.cache/huggingface/hub
ENV TORCH_HOME=/models/.cache/torch
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}
ENV MAX_JOBS=8
ENV INSTALL_WORLDGEN_EXTRAS=${INSTALL_WORLDGEN_EXTRAS}

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        ca-certificates \
        cmake \
        ffmpeg \
        git \
        git-lfs \
        libgl1 \
        libglm-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ninja-build \
        pkg-config \
        wget \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-toolkit-12-8 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q "${MINICONDA_URL}" -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" config --system --set auto_activate_base false \
    && "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && "${CONDA_DIR}/bin/conda" clean -afy \
    && ln -sf "${CONDA_DIR}/bin/conda" /usr/local/bin/conda

COPY . ${HYWORLD_ROOT}
WORKDIR ${HYWORLD_ROOT}

RUN set -euo pipefail \
    && "${CONDA_DIR}/bin/conda" create -y -n hyworld2 python=3.11.15 pip \
    && "${CONDA_DIR}/bin/conda" run -n hyworld2 python -m pip install --upgrade pip setuptools wheel \
    && "${CONDA_DIR}/bin/conda" run -n hyworld2 python -m pip install \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && grep -v '^cupy==' requirements.txt > /tmp/requirements-hyworld2.txt \
    && "${CONDA_DIR}/bin/conda" run -n hyworld2 python -m pip install cupy-cuda12x==13.6.0 \
    && "${CONDA_DIR}/bin/conda" run -n hyworld2 python -m pip install -r /tmp/requirements-hyworld2.txt \
    && "${CONDA_DIR}/bin/conda" run -n hyworld2 python -m pip uninstall -y onnxruntime-gpu \
    && "${CONDA_DIR}/bin/conda" run -n hyworld2 python -m pip install onnxruntime==1.22.1 \
    && cd hyworld2/worldgen/third_party/gsplat_maskgaussian \
    && "${CONDA_DIR}/bin/conda" run -n hyworld2 python -m pip install -e . --no-build-isolation \
    && cd "${HYWORLD_ROOT}" \
    && if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then \
        "${CONDA_DIR}/bin/conda" run -n hyworld2 python -m pip install flash-attn --no-build-isolation; \
    fi \
    && if [[ "${INSTALL_WORLDGEN_EXTRAS}" == "1" ]]; then \
        FORCE_CUDA=1 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 \
            "${CONDA_DIR}/bin/conda" run -n hyworld2 python -m pip install --no-build-isolation -r requirements_git.txt; \
        cd hyworld2/worldgen/third_party/navmesh; \
        RECAST_PATH="${HYWORLD_ROOT}/hyworld2/worldgen/third_party/recastnavigation" \
            "${CONDA_DIR}/bin/conda" run -n hyworld2 python -m pip install . --no-build-isolation; \
        cd "${HYWORLD_ROOT}"; \
    fi \
    && "${CONDA_DIR}/bin/conda" clean -afy

RUN "${CONDA_DIR}/bin/conda" create -y -n hyworld2-pano python=3.10 pip \
    && "${CONDA_DIR}/bin/conda" run -n hyworld2-pano python -m pip install --upgrade pip setuptools wheel \
    && "${CONDA_DIR}/bin/conda" run -n hyworld2-pano python -m pip install \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && "${CONDA_DIR}/bin/conda" run -n hyworld2-pano python -m pip install -r hyworld2/panogen/requirements.txt \
    && "${CONDA_DIR}/bin/conda" run -n hyworld2-pano python -m pip install peft==0.18.1 \
    && "${CONDA_DIR}/bin/conda" clean -afy

RUN cat >/etc/profile.d/hyworld-conda.sh <<'EOF'
source /opt/miniconda3/etc/profile.d/conda.sh
alias hyworld='conda activate hyworld2'
alias hypano='conda activate hyworld2-pano'
EOF

ENV PYTHONPATH=/workspace/hyworld2:/workspace/hyworld2/hyworld2/worldgen:/workspace/hyworld2/hyworld2/panogen

RUN "${CONDA_DIR}/bin/conda" run -n hyworld2 python -c \
        "import os, torch, diffusers, transformers; __import__('recast') if os.environ.get('INSTALL_WORLDGEN_EXTRAS', '1') == '1' else None; import hyworld2.worldrecon.pipeline; print('hyworld2 build check ok', torch.__version__, diffusers.__version__, transformers.__version__)" \
    && cd hyworld2/panogen \
    && "${CONDA_DIR}/bin/conda" run -n hyworld2-pano python -c \
        "import torch, pipeline, pipeline_with_qwen_image; print('hyworld2-pano build check ok', torch.__version__)"

# HF cache MUST point at the pre-downloaded weights under /models/.cache (set above).
# Do NOT override HF_HOME to /cache/huggingface — that dir is empty and breaks
# resolution of Wan/MoGe/SAM3/WorldStereo weights. /cache stays for writable
# runtime caches (torch hub, matplotlib) which are bind-mounted to the host.
ENV TORCH_HOME=/cache/torch
ENV MPLCONFIGDIR=/cache/matplotlib
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── World-generation pipeline fixes (landed from runtime debugging) ──────────
# huggingface.co is unreachable here; resolve public repos (ZIM/GD/MoGe) via the
# Tsinghua mirror. Defaults below point gated / large model repos at the mounted
# /models tree so the pipeline does not need ad hoc env exports at runtime.
ENV HF_ENDPOINT=https://hf-mirror.com
ENV SAM3_REPO_ID=/models/sam3
ENV WORLDSTEREO_REPO=/models/WorldStereo
ENV WORLDMIRROR_MODEL=/models/HY-World-2.0
ENV CAMERA_SELECTOR_MODEL=facebook/dinov2-base
ENV WS_TEXT_DTYPE=bf16
ENV WS_AUX_OFFLOAD=1
ENV WORLDMIRROR_NPROC_PER_NODE=1
ENV WORLDMIRROR_TARGET_SIZE=512
ENV WORLDMIRROR_CUDA_VISIBLE_DEVICES=0

# The base image runs uid 1000 with HOME=/root (non-writable), which breaks cupy /
# triton / vLLM-config cache writes. Make /root writable so the worldgen pipeline
# can create its $HOME caches (isaac-sim still gets HOME=/root).
RUN chmod 777 /root

# rtree: required by NavMesh reconstruction (--apply_recon_iteration, Stage 1).
RUN "${CONDA_DIR}/bin/conda" run -n hyworld2 python -m pip install \
        -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple rtree

# VLM server for WorldNav (stages 1-2). vLLM 0.23 cannot run on Blackwell (its
# bundled FlashInfer misdetects sm_120), so scripts/launch_vlm.sh serves Qwen3-VL
# through the lightweight transformers OpenAI-compatible shim in hyworld2-pano.

WORKDIR ${HYWORLD_ROOT}
USER 1000:1000
CMD ["/bin/bash"]
