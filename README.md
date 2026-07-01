## Docker 容器用法与 Prompt 生成 3D 场景教程

这份 fork 提供了两套 Docker 镜像：

| 变体 | 镜像 | 容器名 | 适用场景 |
|------|------|--------|----------|
| `isaac` | `hyworld2-isaaclab:3.0.0-beta2` | `hyworld2-isaaclab` | 需要 Isaac Lab / Isaac Sim 运行时，或希望使用完整本地环境。 |
| `base` | `hyworld2-base:3.0.0-beta2` | `hyworld2-base` | 不需要 Isaac 的普通 HY-World 推理、全景图和 3DGS 生成。 |

论文把 HY-World 2.0 表述为支持 `text prompts`、`single-view images`、`multi-view images` 和 `videos` 等输入，并说明 HY-Pano 2.0 用于从文本或单视图图像生成全景图。本仓库当前开源的 `hyworld2/panogen` CLI/API 示例仍然是图像条件入口：`pipeline.py` 和 `pipeline_with_qwen_image.py` 都要求传入 `--image`，再用 `--prompt` 控制风格和内容。因此，本地从“纯文本 prompt”开始时，推荐先用 `flux2` 环境生成一张条件图，再把这张图交给 HY-Pano 生成 360 度全景图。

## 模型清单与下载链接

完整 prompt -> 3DGS 流程会用到下面这些模型。`/models/...` 是容器内路径，对应宿主机默认 `~/ws/hyworld2-models/...`。本地运行默认只使用这些显式目录，不依赖 Hugging Face cache 结构。

模型下载方式见下方 Docker 运行步骤中的“下载模型”。

| 模型 | 用途 | 容器内默认位置 / repo id | 大小 | ModelScope | Hugging Face |
|------|------|--------------------------|------|------------|--------------|
| FLUX.2 Klein 9B | 文本生成条件图；也是 `--panorama-backend flux-lora` 的 base model | `/models/FLUX.2-klein-9B` | 50G | [black-forest-labs/FLUX.2-klein-9B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-9B) | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |
| FLUX.2 Klein 9B 360 Panorama LoRA | `--panorama-backend flux-lora` 直接从 prompt 生成 2:1 全景图 | `/models/flux-2-klein-9b-360-panorama-lora` | 约 0.4G | 无；`download` 会 fallback 到 Hugging Face | [crafiq/flux-2-klein-9b-360-panorama-lora](https://huggingface.co/crafiq/flux-2-klein-9b-360-panorama-lora) |
| Qwen-Image-Edit-2509 | HY-Pano Qwen backend base model | `/models/Qwen/Qwen-Image-Edit-2509` | 54G | [Qwen/Qwen-Image-Edit-2509](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2509) | [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) |
| HY-World 2.0 | HY-Pano-Qwen LoRA 和 WorldMirror 权重；`download` 默认跳过 80B full HY-Pano | `/models/HY-World-2.0`，只需 `HY-Pano-2.0/pytorch_lora_weights.safetensors` 和 `HY-WorldMirror-2.0/` | 约 5.6G | [Tencent-Hunyuan/HY-World-2.0](https://www.modelscope.cn/models/Tencent-Hunyuan/HY-World-2.0/files) | [tencent/HY-World-2.0](https://huggingface.co/tencent/HY-World-2.0) |
| Qwen3-VL-8B-Instruct | WorldNav trajectory planning / VLM captions 的 OpenAI-compatible VLM shim | `/models/Qwen/Qwen3-VL-8B-Instruct` | 17G | [Qwen/Qwen3-VL-8B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct) | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| SAM3 | WorldNav 目标分割、WorldMirror sky/semantic mask | `/models/sam3`，由 `SAM3_REPO_ID` 指定 | 6.5G | [facebook/sam3](https://www.modelscope.cn/models/facebook/sam3) | [facebook/sam3](https://huggingface.co/facebook/sam3) |
| WorldStereo | Stage3 WorldStereo adapter 权重；当前 `docker.py run` 默认只使用 `worldstereo-memory-dmd/` | `/models/WorldStereo/worldstereo-memory-dmd` | 33G | [hanshanxue/WorldStereo](https://www.modelscope.cn/models/hanshanxue/WorldStereo) | [hanshanxue/WorldStereo](https://huggingface.co/hanshanxue/WorldStereo) |
| Wan2.1 I2V 14B Diffusers | WorldStereo base video model；由 `WORLDSTEREO_BASE_MODEL` 指定 | `/models/Wan2.1-I2V-14B-480P-Diffusers` | 约 32G | [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) | [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) |
| MoGe v2 ViT-L normal | WorldNav 深度/法线、WorldMirror/3DGS 几何和法线估计 | `/models/moge-2-vitl-normal`，由 `MOGE_MODEL` 指定 | 约 1.3G | [Ruicheng/moge-2-vitl-normal](https://www.modelscope.cn/models/Ruicheng/moge-2-vitl-normal) | [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal) |
| ZIM Anything ViT-L | WorldNav mask refinement | `/models/zim-anything-vitl`，使用 `zim_vit_l_2092/` 子目录，由 `ZIM_MODEL` 指定 | 约 1.2G | [naver-iv/zim-anything-vitl](https://www.modelscope.cn/models/naver-iv/zim-anything-vitl) | [naver-iv/zim-anything-vitl](https://huggingface.co/naver-iv/zim-anything-vitl) |
| GroundingDINO tiny | WorldNav grounding / object mask 辅助 | `/models/grounding-dino-tiny`，由 `GROUNDING_DINO_MODEL` 指定 | 约 1.3G | [IDEA-Research/grounding-dino-tiny](https://www.modelscope.cn/models/IDEA-Research/grounding-dino-tiny) | [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) |
| DINOv2 base | WorldMirror memory-bank camera selector | `/models/dinov2-base`，由 `CAMERA_SELECTOR_MODEL` 指定 | 约 0.7G | [facebook/dinov2-base](https://www.modelscope.cn/models/facebook/dinov2-base) | [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) |

## Docker 运行步骤

### 1. 构建、启动和验证容器

```bash
cd /home/zxh/ws/hyworld2

# Isaac 版本
python docker.py isaac build
python docker.py isaac start
python docker.py isaac verify
python docker.py isaac enter

# 非 Isaac 版本，命令形态相同
python docker.py base build
python docker.py base start
python docker.py base verify
python docker.py base enter
```

常用宿主机命令：

```bash
python docker.py isaac status
python docker.py isaac exec nvidia-smi
python docker.py isaac stop
```

如果命令里省略 `base` / `isaac`，`docker.py` 默认使用 `base` 变体，例如 `python docker.py run ...` 等价于 `python docker.py base run ...`。

### 2. 下载模型

`docker.py download` 会通过 ModelScope CLI 下载当前 prompt -> 3DGS 流程需要的模型，目录结构会按上面的模型表和本地 `~/ws/hyworld2-models` 对齐：

```bash
conda activate torch
python docker.py download --path ~/ws/hyworld2-models
```

先检查命令列表可用 `--dry-run`：

```bash
python docker.py download --path ~/ws/hyworld2-models --dry-run
```

该命令会先调用 `modelscope download --model <model-id> --local_dir <target-dir>`；如果 ModelScope 没有对应 repo，会自动 fallback 到 `hf download <repo-id> --local-dir <target-dir>`，并设置 `HF_ENDPOINT=https://hf-mirror.com`。如果不激活 conda 环境，也可以直接用装了 ModelScope/Hugging Face Hub 的解释器运行，例如 `/home/zxh/miniconda3/envs/torch/bin/python docker.py download --path ~/ws/hyworld2-models`。

### 3. 使用 `docker.py run` 一键执行

可以直接从宿主机启动完整 prompt -> 条件图 -> 全景图 -> 3DGS 流程。命令会自动进入对应容器执行各阶段，阶段开始时在终端打印 `[RUN] ...`，并透传容器内命令输出：

```bash
python docker.py base run \
  --prompt "a realistic sunny mountain village with stone paths, trees, and distant snow peaks" \
  --runname my_prompt_scene \
  --device 0,1
```

默认 `--panorama-backend hypano` 会沿用 FLUX 条件图 + HY-Pano 的路径。若要用 FLUX.2 Klein 9B panorama LoRA 直接生成 `$SCENE/panorama.png`，跳过条件图和 HY-Pano：

```bash
python docker.py run \
  --panorama-backend flux-lora \
  --prompt "a realistic sunny mountain village with stone paths, trees, and distant snow peaks" \
  --runname my_prompt_scene \
  --device 0,1
```

`run` 不会自动 build 镜像；如果镜像不存在会直接报错。若对应容器已经存在，命令会启动/复用该容器并在退出时保留；若同名容器完全不存在，命令会临时启动一个容器并在流程结束或失败后自动关闭，且会把 `/models` 挂载为可写以便使用本地模型目录。`--device 0` 会用单卡跑完整流程，Stage3 默认使用 `group-stream` offload；`--device 0,1` 会用第 0 张卡跑 FLUX/HY-Pano、WorldNav 主进程和 3DGS training，用第 1 张卡跑 VLM shim，Stage3 和 `gen_gs_data.py` 使用两张卡。3DGS 训练的每 GPU 图像 batch 可用 `--batchsize N` 调整，默认 `4`。恢复已有场景时加 `--skip-existing`。

两个 panorama backend 默认都用 `1952x960` 做 panorama 推理；HY-Pano 会默认融合并裁掉 32px 接缝，FLUX LoRA 路径也会用 `--flux-pano-blend-width 32` 做同样处理，因此最终 `$SCENE/panorama.png` 默认保存为 `1920x960`。可分别用 `--pano-height/--pano-width`、`--flux-pano-height/--flux-pano-width` 和 `--flux-pano-blend-width` 覆盖。

### 4. 挂载路径和容器环境

`docker.py` 默认把仓库挂载到容器内 `/workspace/hyworld2`，把模型目录 `~/ws/hyworld2-models` 挂载到 `/models`，并挂载 Hugging Face、Torch 和 Matplotlib 缓存目录。

容器内常用 conda 环境如下：

| 环境 | 用途 |
|------|------|
| `hyworld2` | WorldNav、WorldStereo、WorldMirror、3DGS 训练和查看。 |
| `hyworld2-pano` | HY-Pano 2.0 全景图生成，以及 WorldNav 使用的 VLM shim。 |
| `flux2` | FLUX.2 Klein 文本生成条件图，不污染 `hyworld2-pano` 的 diffusers 版本。 |

下面的长推理命令都使用 `conda run --no-capture-output` 和 `python -u`，避免 `conda run` 缓冲日志导致终端长时间看不到进度。

## 手动运行

下面的命令适合需要逐步调试、复用中间产物或绕开 `docker.py run` wrapper 的情况。

### 1. 进入容器后设置场景变量

```bash
cd /workspace/hyworld2

SCENE=/workspace/hyworld2/examples/worldgen/my_prompt_scene
RESULT_DIR=$SCENE/gs_results
PROMPT="a realistic sunny mountain village with stone paths, trees, and distant snow peaks"

mkdir -p "$SCENE"
```

后续命令默认模型已经在 `/models` 下，例如 `/models/HY-World-2.0`、`/models/Qwen/Qwen-Image-Edit-2509` 和 `/models/FLUX.2-klein-9B`。

### 2. 从纯文本 prompt 生成条件图

如果你已经有一张输入图，可以跳过这一步，直接设置：

```bash
INPUT_IMAGE=/path/to/your/input.png
```

如果只有文字 prompt，先生成一张条件图：

```bash
cd /workspace/hyworld2

CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/bin/conda run --no-capture-output -n flux2 \
  python -u scripts/flux2_klein_text2img.py \
    --model 9b \
    --model-path /models/FLUX.2-klein-9B \
    --prompt "$PROMPT" \
    --output "$SCENE/condition.png" \
    --height 1024 \
    --width 1024 \
    --steps 4 \
    --guidance-scale 1.0 \
    --seed 42 \
    --placement offload

INPUT_IMAGE=$SCENE/condition.png
```

关键参数：

| 参数 | 说明 |
|------|------|
| `--model` | 使用 `9b`，与 `docker.py run` 默认路径一致。 |
| `--model-path` | 本地 FLUX.2 Klein 9B 权重路径，默认使用 `/models/FLUX.2-klein-9B`。 |
| `--height` / `--width` | 条件图分辨率；1024x1024 是常用起点。 |
| `--steps` | FLUX.2 采样步数；烟测可用 1，正式生成建议从 4 开始。 |
| `--placement` | `offload` 降低峰值显存；`cuda` 更快但更吃显存。 |

### 3. 生成 360 度全景图

如果 `$SCENE/panorama.png` 已存在，可以跳过本节。否则用条件图和 prompt 生成全景图：

```bash
cd /workspace/hyworld2/hyworld2/panogen

CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2-pano \
  python -u pipeline_with_qwen_image.py \
    --image "$INPUT_IMAGE" \
    --prompt "$PROMPT" \
    --save "$SCENE/panorama.png" \
    --pretrained-model-name-or-path /models/Qwen/Qwen-Image-Edit-2509 \
    --lora-path /models/HY-World-2.0 \
    --lora-subfolder HY-Pano-2.0 \
    --height 960 \
    --width 1952 \
    --num-inference-steps 4 \
    --load-strategy sequential-offload \
    --seed 42 \
    --reproduce
```

关键参数：

| 参数 | 说明 |
|------|------|
| `--image` | 必填；当前开源 CLI 用它作为全景图生成的条件图。 |
| `--prompt` | 控制场景语义、风格和补全方向。 |
| `--height` / `--width` | 全景图推理分辨率，默认示例为 960x1952；HY-Pano 默认再融合并裁掉 32px 接缝，最终保存为 960x1920。 |
| `--num-inference-steps` | 采样步数；先用 `4` 验证输出非空，再逐步加到 `8/16/24`。不要一开始直接 40。 |
| `--load-strategy` | 推荐 `sequential-offload`。`balanced` 已禁用；`cpu-offload` 曾在 32GB 卡上 OOM。 |

### 4. 运行 WorldNav、WorldStereo、WorldMirror 和 3DGS

`docker.py run` 把完整流程组织成 4 个可跳过的 wrapper stage：

| Stage | 内容 |
|------|------|
| `1` | panorama generation：`hypano` 路径会先用 FLUX.2 9B 生成条件图，再用 HY-Pano 生成全景图；`flux-lora` 路径直接生成全景图。 |
| `2` | trajectory planning、trajectory rendering 和 VLM captions；wrapper 会自动启动并在 Stage3 前停止 VLM shim。 |
| `3` | WorldStereo expansion 和 WorldMirror generation bank。 |
| `4` | `gen_gs_data.py` 数据准备，以及 `world_gs_trainer` 训练/导出 3DGS。 |

手动运行时需要自己管理 OpenAI-compatible VLM 服务。先在 GPU1 启动本地 shim：

```bash
cd /workspace/hyworld2
CUDA_VISIBLE_DEVICES=1 PORT=8000 scripts/launch_vlm.sh > /tmp/hyworld_vlm.log 2>&1 &
```

然后执行后续命令：

```bash
cd /workspace/hyworld2/hyworld2/worldgen

# 对应 docker.py run Stage 2: 轨迹规划。
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
  python -u traj_generate.py \
    --target_path "$SCENE" \
    --llm_addr localhost --llm_port 8000 --llm_name Qwen/Qwen3-VL-8B-Instruct \
    --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm

# 对应 docker.py run Stage 2: 轨迹渲染和 VLM caption。
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
  torchrun --nproc_per_node=1 traj_render.py \
    --target_path "$SCENE" \
    --llm_addr localhost --llm_port 8000 --llm_name Qwen/Qwen3-VL-8B-Instruct

# Stage 3 之前释放 VLM 显存。
pkill -f '[v]lm_server.py|[l]aunch_vlm.sh' || true

# 对应 docker.py run Stage 3: WorldStereo 扩展视角 + WorldMirror generation bank。
# 多卡手动运行：
CUDA_VISIBLE_DEVICES=0,1 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
  torchrun --nproc_per_node=2 video_gen.py \
    --target_path "$SCENE" --fsdp --local_files_only

# 单卡手动运行：不要加 --fsdp；使用和 docker.py run --device 0 默认一致的 group-stream。
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
  torchrun --nproc_per_node=1 video_gen.py \
    --target_path "$SCENE" --local_files_only --offload-mode group-stream

# 对应 docker.py run Stage 4: 准备 3DGS 训练数据。
CUDA_VISIBLE_DEVICES=0,1 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
  torchrun --nproc_per_node=2 gen_gs_data.py \
    --root_path "$SCENE" --save_normal --split_sky

# 对应 docker.py run Stage 4: 训练并导出 3DGS。--ssim-lambda 0 用于避开测试环境中
# RTX 5090/sm_120 上不稳定的 fused SSIM kernel 路径。
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
  python -u -m world_gs_trainer default \
    --data-dir "$SCENE/gs_data" \
    --result-dir "$RESULT_DIR" \
    --max-steps 4000 --save-steps 4000 --eval-steps 4000 --ply-steps 4000 \
    --batch-size 1 \
    --save-ply --convert-to-spz --disable-video --disable-viewer \
    --use-scale-regularization --antialiased \
    --depth-loss --normal-loss --sky-depth-from-pcd \
    --use-mask-gaussian --mask-export-stochastic \
    --no-mask-export-anchor-protection --use-anchor-protection --export-mesh \
    --ssim-lambda 0 \
    --strategy.refine-start-iter 150 \
    --strategy.refine-stop-iter 2000 \
    --strategy.refine-every 100 \
    --strategy.refine-scale2d-stop-iter 2000 \
    --strategy.reset-every 99990 \
    --strategy.grow-grad2d 0.0001 \
    --strategy.prune-scale3d 0.1
```

恢复中断任务时常用参数：

| 参数 | 用途 |
|------|------|
| `--skip-existing` | `docker.py run` 参数；转发为底层脚本的 `--skip_exist`，跳过已有 Stage2/Stage3 产物，适合 resume。 |
| `--skip 1,2,3,4` | `docker.py run` 参数；跳过指定 wrapper stage。例如已有全景图时可用 `--skip 1`。 |
| `--local_files_only` | 只使用本地模型缓存，避免运行时访问 Hugging Face。 |
| `--stage3-offload-mode auto` | `docker.py run` 默认；单卡解析为 `group-stream`，多卡解析为 `none` 并启用 FSDP。 |
| `--fsdp` | 手动运行 `video_gen.py` 的多卡参数；`docker.py run --device 0,1` 会自动添加。 |

### 5. 查看生成的 3DGS

```bash
CKPT=$(ls "$RESULT_DIR"/ckpts/ckpt_*_rank0.pt | sort -V | tail -1)

CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
  python -u show_gs.py --port 8081 --gpu_id 0 --ckpt "$CKPT"
```

Dockerfile 默认设置了 `/models` 模型路径、`WS_TEXT_DTYPE=bf16`、`WS_AUX_OFFLOAD=1`、WorldMirror 单卡 `512` fallback，以及 PyTorch CUDA allocator 的 `expandable_segments`。更多本地实测显存、失败记录和长流程 runbook 见 `WORLDGEN_CHANGELOG.md`。


<h1>HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds</h1>

[English](README.md) | [简体中文](README_zh.md)

<p align="center">
  <img src="assets/teaser.png" width="95%" alt="HY-World-2.0 Teaser">
</p>

<div align="center">
  <a href=https://3d.hunyuan.tencent.com/sceneTo3D target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/tencent/HY-World-2.0 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://3d-models.hunyuan.tencent.com/world/ target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px></a>
  <a href=https://arxiv.org/abs/2604.14268 target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
   <a href=https://modelscope.cn/models/Tencent-Hunyuan/HY-World-2.0 target="_blank"><img src=https://img.shields.io/badge/ModelScope-Models-624aff.svg height=22px></a>
  <a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Tencent%20HY-black.svg?logo=x height=22px></a>
 <a href="#community-resources" target="_blank"><img src=https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore height=22px></a>
</div>

<br>
<p align="center">
  <i>"What Is Now Proved Was Once Only Imagined"</i>
</p>

## 🎥 Video
https://github.com/user-attachments/assets/b56f4750-25c9-48fb-83ff-d58526711463

## 🔥 News

- **[May 18, 2026]**: 🤗 Open-source World Generation inference code and WorldStereo 2.0 model weights!
- **[May 11, 2026]**: 🤗 Open-source HY-Pano 2.0 inference code and model weights!
- **[April 16, 2026]**: 🚀 Release HY-World 2.0 technical report & partial codes!
- **[April 16, 2026]**: 🤗 Open-source WorldMirror 2.0 inference code and model weights!


## 📋 Table of Contents
- [📖 Introduction](#-introduction)
- [✨ Highlights](#-highlights)
- [🧩 Architecture](#-architecture)
- [📝 Open-Source Plan](#-open-source-plan)
- [🎁 Model Zoo](#-model-zoo)
- [🤗 Get Started](#-get-started)
- [🔮 Performance](#-performance)
- [🎬 More Examples](#-more-examples)
- [📚 Citation](#-citation)


## 📖 Introduction

**HY-World 2.0** is a multi-modal world model framework for **world generation** and **world reconstruction**. It accepts diverse input modalities — text, single-view images, multi-view images, and videos — and produces 3D world representations (meshes / Gaussian Splattings). It offers two core capabilities:

- **World Generation** (text / single image &rarr; 3D world): syntheses high-fidelity, navigable 3D scenes through a four-stage method —— a) ![Panorama Generation](https://img.shields.io/badge/Panorama_Generation-4285F4?style=flat-square) with HY-Pano 2.0, b) ![Trajectory Planning](https://img.shields.io/badge/Trajectory_Planning-EA4335?style=flat-square) with WorldNav, c) ![World Expansion](https://img.shields.io/badge/World_Expansion-FBBC05?style=flat-square) with WorldStereo 2.0, and d) ![World Composition](https://img.shields.io/badge/World_Composition-34A853?style=flat-square) with WorldMirror 2.0 & 3DGS learning.
- **World Reconstruction** (multi-view images / video &rarr; 3D): Powered by WorldMirror 2.0, a unified feed-forward model that simultaneously predicts depth, surface normals, camera parameters, 3D point clouds, and 3DGS attributes in a single forward pass.

HY-World 2.0 is an **open-source state-of-the-art** world model.  We released all model weights, code, and technical details to facilitate reproducibility and advance research in this field.

### Why 3D World Models?

Existing world models, such as Genie 3, Cosmos, and HY-World 1.5 (WorldPlay+WorldCompass), generate pixel-level videos — essentially "watching a movie" that vanishes once playback ends. **HY-World 2.0 takes a fundamentally different approach**: it directly produces editable, persistent 3D assets (meshes / 3DGS) that can be imported into game engines like Blender/Unity/Unreal Engine/Isaac Sim — more like "building a playable game" than recording a clip. This paradigm shift natively resolves many long-standing pain points of video world models:

|  | Video World Models | 3D World Model (HY-World 2.0) |
|--|---|---|
| **Output** | Pixel videos (non-editable) | Real 3D assets — meshes / 3DGS (fully editable) |
| **Playable Duration** | Limited (typically 1 min) | Unlimited — assets persist permanently |
| **3D Consistency** | No (flickering, artifacts across views) | Native — inherently consistent in 3D |
| **Real-Time Rendering** | Requires per-frame inference; high latency | Consumer GPUs can render in real time |
| **Controllability** | Weak (imprecise character control, no real physics) | Precise — zero-error control, real physics collision, accurate lighting |
| **Inference Cost** | Accumulates with every interaction | One-time generation; rendering cost ≈ 0 |
| **Engine Compatibility** | ✗ Video files only | ✓ Directly importable into Blender / UE / Isaac Engine |
| | $\color{IndianRed}{\textsf{Watch a video, then it's gone}}$ | $\color{RoyalBlue}{\textbf{Build a world, keep it forever}}$ |


<table align="center" style="border: none;">
  <tr>
    <td align="center" width="50%"><img src="assets/screenshot_1.gif" width="100%"></td>
    <td align="center" width="50%"><img src="assets/screenshot_2.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshot_7.gif" width="100%"></td>
    <td align="center" width="50%"><img src="assets/screenshot_8.gif" width="100%"></td>
  </tr>
</table>

<p align="center"><em>All above are <strong>real 3D assets</strong> (not generated videos) and entirely created by HY-World 2.0 -- captured from live real-time interaction.</em></p>

## ✨ Highlights

- **Real 3D Worlds, Not Just Videos**

  Unlike video-only world models (e.g., Genie 3, HY World 1.5), HY-World 2.0 generates **real 3D assets** — 3DGS, meshes, and point clouds — that are freely explorable, editable, and directly importable into **Unity / Unreal Engine / Isaac**. From a single text prompt or image, create navigable 3D worlds with diverse styles: realistic, cartoon, game, and more.

<p align="center">
  <img src="assets/mesh_en.gif" width="95%">
</p>


- **Instant 3D Reconstruction from Photos & Videos**

  Powered by **WorldMirror 2.0**, a unified feed-forward model that predicts dense point clouds, depth maps, surface normals, camera parameters, and 3DGS from multi-view images or casual videos in a single forward pass. Supports flexible-resolution inference (50K–500K pixels) with SOTA accuracy. Capture a video, get a digital twin.

<p align="center">
  <img src="assets/recon_en.gif" width="95%">
</p>

- **Interactive Character Exploration**

  Go beyond viewing — **play inside your generated worlds**. HY-World 2.0 supports first-person navigation and third-person character mode, enabling users to freely explore AI-generated streets, buildings, and landscapes with physics-based collision.  Go to [our product page](https://3d.hunyuan.tencent.com/sceneTo3D) for free try. 

<p align="center">
  <img src="assets/interactive.gif" width="95%">
</p>


## 🧩 Architecture
- **Refer to our tech report for more details**

  A systematic pipeline of HY-World 2.0 — *Panorama Generation* (HY-Pano-2.0) &rarr; *Trajectory Planning* (WorldNav) &rarr; *World Expansion* (WorldStereo 2.0) &rarr; *World Composition* (WorldMirror 2.0 + Splattings Learning) — that automatically transforms text or a single image into a high-fidelity, navigable 3D world (3DGS/mesh outputs).

<p align="center">
  <img src="assets/overview.png" width="95%">
</p>


## 📝 Open-Source Plan

- [x] Technical Report
- [x] WorldMirror 2.0 Code & Model Checkpoints
- [x] Full Inference Code for World Generation (WorldNav + WorldStereo + World Composition)
- [x] Panorama Generation (HY-Pano 2.0) Model & Code
- [x] World Expansion (WorldStereo 2.0) Model & Code


## 🎁 Model Zoo

### World Reconstruction — WorldMirror Series

| Model | Description | Params | Date | Hugging Face |
|-------|-------------|--------|------|--------------|
| WorldMirror-2 [new] | Multi-view / video &rarr; 3D reconstruction | ~1.2B | 2026 | [Download](https://huggingface.co/tencent/HY-World-2.0/tree/main/HY-WorldMirror-2.0) |
| WorldMirror-1 | Multi-view / video &rarr; 3D reconstruction (legacy) | ~1.2B | 2025 | [Download](https://huggingface.co/tencent/HunyuanWorld-Mirror/tree/main) |

### Panorama Generation — HY-Pano Series

| Model | Description | Params | Date | Hugging Face |
|-------|-------------|--------|------|--------------|
| HY-Pano-2 [new] | Text / image → 360° panorama | ~80B | 2026 | [Download](https://huggingface.co/tencent/HY-World-2.0/tree/main/HY-Pano-2.0) |
| HY-Pano-2-Qwen [new] | Text / image → 360° panorama | ~425M | 2026 | [Download](https://huggingface.co/tencent/HY-World-2.0/blob/main/HY-Pano-2.0/pytorch_lora_weights.safetensors) |

### World Expansion — WorldStereo Series

| Model           | Description | Params | Date | Hugging Face |
|-----------------|-------------|-----|------|--------------|
| WorldStereo-2 [new] | Panorama &rarr;  3DGS world |  ~17B  | 2026 | [Download](https://huggingface.co/hanshanxue/WorldStereo/tree/main) |

We recommend referring to our previous works, [WorldStereo](https://github.com/FuchengSu/WorldStereo) and [WorldMirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror), for background knowledge on 3D world generation and reconstruction. 

## 🤗 Get Started

### Install Requirements

We recommend **CUDA 12.8** and **Python 3.11+**. The easiest path is to prepare one shared environment, first make **World Reconstruction (WorldMirror 2.0)** work, and then install the extra components required by **World Generation**.

#### 1. Create the shared environment

```bash
git clone https://github.com/Tencent-Hunyuan/HY-World-2.0
cd HY-World-2.0

conda create -n hyworld2 python=3.11.15
conda activate hyworld2
```

#### 2. Install World Reconstruction dependencies

After this step, the environment is ready for **worldrecon / WorldMirror 2.0**.

```bash
# Base dependencies shared by worldrecon and worldgen
pip install -r requirements.txt

# Recommended: install the custom gsplat variant once for both worldrecon and worldgen
cd hyworld2/worldgen/third_party/gsplat_maskgaussian
pip install -e . --no-build-isolation
cd ../../../../
```

If you only need **worldrecon** and want a simpler fallback, official `gsplat` is also supported:

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

Install **one** FlashAttention backend:

```bash
# Recommended for Hopper GPUs: FlashAttention-3
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
cd ../../
rm -rf flash-attention
```

```bash
# Simpler alternative: FlashAttention-2
pip install flash-attn --no-build-isolation
```

#### 3. Add extra World Generation dependencies

Run the following extra steps only if you need **worldgen**. These commands assume the shared `hyworld2` environment above is already active.

```bash
# Git-based dependencies require torch/CUDA to be installed first
pip install --no-build-isolation -r requirements_git.txt

# recastnavigation is managed as a git submodule
git submodule update --init --recursive

# Recast navmesh extension for trajectory planning
cd hyworld2/worldgen/third_party/navmesh
pip install . --no-build-isolation
cd ../../../../
```

For **HY-Pano-2** installation, please refer to **[hyworld2/panogen/README.md](hyworld2/panogen/README.md)**.

### Code Usage — Panorama Generation (HY-Pano-2)

For full documentation and CLI reference, see **[hyworld2/panogen/README.md](hyworld2/panogen/README.md)**.

We provide a `diffusers`-like Python API for HY-Pano 2.0. Model weights are automatically downloaded from Hugging Face on first run.

```python
from pipeline import HunyuanPanoPipeline

pipeline = HunyuanPanoPipeline.from_pretrained('tencent/HY-World-2.0')
output = pipeline('input.png')
output.save('output_panorama.png')
```

### Code Usage — World Generation (WorldNav, WorldStereo-2, and 3DGS)

The world Generation pipeline turns a panorama scene into a navigable 3D world through five stages:

| Stage | Script | Description |
|-------|--------|-------------|
| 1. Trajectory Planning | `traj_generate.py` | VLM-guided camera trajectory planning with obstacle-aware navigation |
| 2. Trajectory Rendering | `traj_render.py` | Multi-GPU point-cloud rendering along planned trajectories |
| 3. World Expansion | `video_gen.py` | WorldStereo-2 keyframe generation with memory-guided consistency |
| 4. GS Data Preparation | `gen_gs_data.py` | Extract frames, aligned depth, normals, and cameras for 3DGS training |
| 5. 3DGS Training | `world_gs_trainer.py` | Optimize and export the final Gaussian Splatting world |

For full documentation, prerequisites, and CLI arguments, see **[hyworld2/worldgen/README.md](hyworld2/worldgen/README.md)**.

### Code Usage — WorldMirror 2.0
WorldMirror 2.0 supports the following usage modes:

- [Code Usage](#code-usage--worldmirror-20)
- [Gradio App](#gradio-app--worldmirror-20)

We provide a `diffusers`-like Python API for WorldMirror 2.0. Model weights are automatically downloaded from Hugging Face on first run.

```python
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

pipeline = WorldMirrorPipeline.from_pretrained('tencent/HY-World-2.0')
result = pipeline('path/to/images')
```

**With Prior Injection (Camera & Depth):**

```python
result = pipeline(
    'path/to/images',
    prior_cam_path='path/to/prior_camera.json',
    prior_depth_path='path/to/prior_depth/',
)
```

> For the detailed structure of camera/depth priors and how to prepare them, see [Prior Preparation Guide](DOCUMENTATION.md#prior-injection).

**CLI:**

```bash
# Single GPU
python -m hyworld2.worldrecon.pipeline --input_path path/to/images

# Multi-GPU
torchrun --nproc_per_node=2 -m hyworld2.worldrecon.pipeline \
    --input_path path/to/images \
    --use_fsdp --enable_bf16
```

> **Important:** In multi-GPU mode, the number of input images must be **>= the number of GPUs**. For example, with `--nproc_per_node=8`, provide at least 8 images.

### Gradio App — WorldMirror 2.0

We provide an interactive [Gradio](https://www.gradio.app/) web demo for WorldMirror 2.0. Upload images or videos and visualize 3DGS, point clouds, depth maps, normal maps, and camera parameters in your browser.

```bash
# Single GPU
python -m hyworld2.worldrecon.gradio_app

# Multi-GPU
torchrun --nproc_per_node=2 -m hyworld2.worldrecon.gradio_app \
    --use_fsdp --enable_bf16
```

For the full list of Gradio app arguments (port, share, local checkpoints, etc.), see [DOCUMENTATION.md](DOCUMENTATION.md#gradio-app).



## 🔮 Performance

For full benchmark results, please refer to the [technical report](https://3d-models.hunyuan.tencent.com/world/).

### WorldStereo 2.0 — Camera Control

<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="3" align="center">Camera Metrics</th>
      <th colspan="4" align="center">Visual Quality</th>
    </tr>
    <tr>
      <th>RotErr ↓</th><th>TransErr ↓</th><th>ATE ↓</th>
      <th>Q-Align ↑</th><th>CLIP-IQA+ ↑</th><th>Laion-Aes ↑</th><th>CLIP-I ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>SEVA</td><td>1.690</td><td>1.578</td><td>2.879</td><td>3.232</td><td>0.479</td><td>4.623</td><td>77.16</td></tr>
    <tr><td>Gen3C</td><td>0.944</td><td>1.580</td><td>2.789</td><td>3.353</td><td>0.489</td><td>4.863</td><td>82.33</td></tr>
    <tr><td>WorldStereo</td><td>0.762</td><td>1.245</td><td>2.141</td><td>4.149</td><td><b>0.547</b></td><td>5.257</td><td>89.05</td></tr>
    <tr><td><b>WorldStereo 2.0</b></td><td><b>0.492</b></td><td><b>0.968</b></td><td><b>1.768</b></td><td><b>4.205</b></td><td>0.544</td><td><b>5.266</b></td><td><b>89.43</b></td></tr>
  </tbody>
</table>

### WorldStereo 2.0 — Single-View-Generated Reconstruction

<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="4">Tanks-and-Temples</th>
      <th colspan="4">MipNeRF360</th>
    </tr>
    <tr>
      <th>Precision ↑</th>
      <th>Recall ↑</th>
      <th>F1-Score ↑</th>
      <th>AUC ↑</th>
      <th>Precision ↑</th>
      <th>Recall ↑</th>
      <th>F1-Score ↑</th>
      <th>AUC ↑</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">SEVA</td>
      <td>33.59</td>
      <td>35.34</td>
      <td>36.73</td>
      <td>51.03</td>
      <td>22.38</td>
      <td>55.63</td>
      <td>28.75</td>
      <td>46.81</td>
    </tr>
    <tr>
      <td align="left">Gen3C</td>
      <td><u>46.73</u></td>
      <td>25.51</td>
      <td>31.24</td>
      <td>42.44</td>
      <td>23.28</td>
      <td><strong>75.37</strong></td>
      <td>35.26</td>
      <td>52.10</td>
    </tr>
    <tr>
      <td align="left">Lyra</td>
      <td><strong>50.38</strong></td>
      <td>28.67</td>
      <td>32.54</td>
      <td>43.05</td>
      <td>30.02</td>
      <td>58.60</td>
      <td>36.05</td>
      <td>49.89</td>
    </tr>
    <tr>
      <td align="left">FlashWorld</td>
      <td>26.58</td>
      <td>20.72</td>
      <td>22.29</td>
      <td>30.45</td>
      <td>35.97</td>
      <td>53.77</td>
      <td>42.60</td>
      <td>53.86</td>
    </tr>
    <tr>
      <td align="left">WorldStereo 2.0</td>
      <td>43.62</td>
      <td><u>41.02</u></td>
      <td><u>41.43</u></td>
      <td><u>58.19</u></td>
      <td><strong>43.19</strong></td>
      <td><u>65.32</u></td>
      <td><strong>51.27</strong></td>
      <td><strong>65.79</strong></td>
    </tr>
    <tr>
      <td align="left">WorldStereo 2.0 (DMD)</td>
      <td>40.41</td>
      <td><strong>44.41</strong></td>
      <td><strong>43.16</strong></td>
      <td><strong>60.09</strong></td>
      <td><u>42.34</u></td>
      <td>64.83</td>
      <td><u>50.52</u></td>
      <td><u>65.64</u></td>
    </tr>
  </tbody>
</table>

### WorldMirror 2.0 — Point Map Reconstruction

**Point Map Reconstruction on 7-Scenes, NRGBD, and DTU.** We report the mean Accuracy and Completeness of WorldMirror under different input configurations. **Bold** results are best. "L / M / H" denote low / medium / high inference resolution. "+ all priors" denotes injection of camera extrinsics, camera intrinsics, and depth priors.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2" align="center">7-Scenes <sub>(scene)</sub></th>
      <th colspan="2" align="center">NRGBD <sub>(scene)</sub></th>
      <th colspan="2" align="center">DTU <sub>(object)</sub></th>
    </tr>
    <tr>
      <th>Acc. ↓</th><th>Comp. ↓</th>
      <th>Acc. ↓</th><th>Comp. ↓</th>
      <th>Acc. ↓</th><th>Comp. ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="7"><em>WorldMirror 1.0</em></td></tr>
    <tr><td>&nbsp;&nbsp;L</td><td>0.043</td><td>0.055</td><td>0.046</td><td>0.049</td><td>1.476</td><td>1.768</td></tr>
    <tr><td>&nbsp;&nbsp;L + all priors</td><td>0.021</td><td>0.026</td><td>0.022</td><td>0.020</td><td>1.347</td><td>1.392</td></tr>
    <tr><td>&nbsp;&nbsp;M</td><td>0.043</td><td>0.049</td><td>0.041</td><td>0.045</td><td>1.017</td><td>1.780</td></tr>
    <tr><td>&nbsp;&nbsp;M + all priors</td><td>0.018</td><td>0.023</td><td>0.016</td><td>0.014</td><td>0.735</td><td>0.935</td></tr>
    <tr><td>&nbsp;&nbsp;H</td><td>0.079</td><td>0.087</td><td>0.077</td><td>0.093</td><td>2.271</td><td>2.113</td></tr>
    <tr><td>&nbsp;&nbsp;H + all priors</td><td>0.042</td><td>0.041</td><td>0.078</td><td>0.082</td><td>1.773</td><td>1.478</td></tr>
    <tr><td colspan="7"></td></tr>
    <tr><td colspan="7"><em>WorldMirror 2.0</em></td></tr>
    <tr><td>&nbsp;&nbsp;L</td><td>0.041</td><td>0.052</td><td>0.047</td><td>0.058</td><td>1.352</td><td>2.009</td></tr>
    <tr><td>&nbsp;&nbsp;L + all priors</td><td>0.019</td><td>0.024</td><td>0.017</td><td>0.015</td><td>1.100</td><td>1.201</td></tr>
    <tr><td>&nbsp;&nbsp;M</td><td>0.033</td><td>0.046</td><td>0.039</td><td>0.047</td><td>1.005</td><td>1.892</td></tr>
    <tr><td>&nbsp;&nbsp;M + all priors</td><td>0.013</td><td>0.017</td><td><b>0.013</b></td><td><b>0.013</b></td><td>0.690</td><td>0.876</td></tr>
    <tr><td>&nbsp;&nbsp;H</td><td>0.037</td><td>0.040</td><td>0.046</td><td>0.053</td><td>0.845</td><td>1.904</td></tr>
    <tr><td>&nbsp;&nbsp;<b>H + all priors</b></td><td><b>0.012</b></td><td><b>0.016</b></td><td>0.015</td><td>0.016</td><td><b>0.554</b></td><td><b>0.771</b></td></tr>
  </tbody>
</table>
 
### WorldMirror 2.0 — Prior Comparison

**Comparison with Pow3R and MapAnything under Different Prior Conditions.** Results are averaged on 7-Scenes, NRGBD, and DTU datasets. Pow3R (pro) refers to the original Pow3R with Procrustes alignment.


<p align="center">
  <img src="assets/prior_comparison2_wm2.png" width="85%">
</p>




## 🎬 More Examples

<table align="center" style="border: none;">
  <tr>
    <td align="center" width="50%"><img src="assets/screenshot_3.gif" width="100%"></td>
    <td align="center" width="50%"><img src="assets/screenshot_4.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshot_5.gif" width="100%"></td>
    <td align="center" width="50%"><img src="assets/screenshot_6.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshot_9.gif" width="100%"></td>
    <td align="center" width="50%"><img src="assets/screenshot_10.gif" width="100%"></td>
  </tr>
</table>


## 📖 Documentation

For detailed usage guides, parameter references, output format specifications, and prior injection instructions, see **[DOCUMENTATION.md](DOCUMENTATION.md)**.


## 📚 Citation

If you find HunyuanWorld 2.0 useful for your research, please cite:

```bibtex
@article{hyworld22026,
  title={HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds},
  author={Team HY-World},
  journal={arXiv preprint arXiv:2604.14268},
  year={2026}
}

@article{hunyuanworld2025tencent,
    title={HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels},
    author={Team HunyuanWorld},
    year={2025},
    journal={arXiv preprint}
}
```

## 📧 Contact

Please send emails to tengfeiwang12@gmail.com for questions or feedback.


## 🙏 Acknowledgements

We would like to thank [HunyuanWorld 1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0), [WorldMirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror), [WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay), [WorldStereo](https://github.com/FuchengSu/WorldStereo), [HunyuanImage](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0) for their great work.
