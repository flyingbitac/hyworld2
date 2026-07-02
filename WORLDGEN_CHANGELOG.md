# HY-World 2.0 本地修改与显存记录

本文记录当前工作区中为跑通 `examples/worldgen/case000` 的世界生成流水线所做的代码、参数和容器修改。来源包括当前 git diff、`/tmp/claude-1001/-home-zxh-ws-hyworld2/.../tasks/*.output` 中的 Claude 运行日志，以及本轮已验证过的 Stage3-Stage5 / `show_gs` 结果。

## 运行结论

- 已跑通的有效路径：Stage3 使用 2 卡 WorldStereo + 单进程 WorldMirror，Stage4 使用 2 卡，Stage5 使用单卡 3DGS 并设置 `--ssim-lambda 0`。
- FLUX.2 Klein 9B 验证已完成：`hyworld2-pano` 不能直接复用，因为它缺少新版 Diffusers 的 `Flux2KleinPipeline`；容器内新增并验证了独立 `flux2` conda 环境。当前 `docker.py` 只保留 9B 路径，text2img 和 HY-World Qwen img2pano 均已跑通。
- `show_gs` 已能加载 `/workspace/hyworld2/examples/worldgen/case000/gs_results_single/ckpts/ckpt_3999_rank0.pt`。
- Stage5 验证指标：PSNR `25.74`，SSIM `0.748`，LPIPS `0.302`，Gaussians `612349`。
- Panogen Qwen-Image-Edit 已完成阶段级 GPU profile：fresh container 中使用 `--load-strategy balanced`，GPU0 峰值 `5489 MiB` / 平均利用率 `35.13%`，GPU1 峰值 `16453 MiB` / 平均利用率 `0.37%`。
- Stage1 skip-existing 已完成阶段级 GPU profile：单卡，使用 Dockerfile 构建出的 fresh container 和 PyTorch3D CUDA rasterizer，GPU0 峰值 `9801 MiB` / 平均利用率 `13.28%`，GPU1 空闲。
- Stage1 true-VLM 已完成阶段级 GPU profile：fresh container 中临时 scene 从 `panorama.png` 重新生成 trajectory，GPU0 跑 `traj_generate.py`，GPU1 跑 Qwen3.5-4B shim；GPU0 峰值 `11213 MiB` / 平均利用率 `8.15%`，GPU1 峰值 `17995 MiB` / 平均利用率 `2.20%`。
- Stage2 已完成阶段级 GPU profile：fresh container 中 GPU0 单进程点云渲染 + GPU1 Qwen3.5-4B shim caption，GPU0 峰值 `3895 MiB` / 平均利用率 `45.85%`，GPU1 峰值 `18395 MiB` / 平均利用率 `40.55%`。
- Stage3 resume/load 已完成阶段级 GPU profile：2 卡，`video_gen.py --fsdp --skip_exist` 加载 WorldStereo/FSDP 后发现 33 个已有 render 和 `aligned_pcd.ply`，跳过重算；GPU0/GPU1 峰值均为 `23621 MiB`。
- Stage3 full attempt 已完成长跑 GPU profile：fresh container 中复制 case000 到 `/tmp/hyworld_stage3_full` 后运行 `video_gen.py --fsdp` 不带 `--skip_exist`，33 个 WorldStereo 视频、291 个 WorldMirror depth、`aligned_pcd.ply` 和 `global_pcd.ply` 均已产出；命令在长尾阶段运行 `2788.561s` 后由人工 SIGTERM，GPU0 峰值 `32069 MiB` / 平均利用率 `73.32%`，GPU1 峰值 `32107 MiB` / 平均利用率 `74.31%`。
- Stage3 WorldMirror fallback 已完成组件级 GPU profile：单卡，291 张图，target size `512`，GPU0 峰值 `14081 MiB` / 平均利用率 `35.11%`，GPU1 空闲。
- Stage4 已完成阶段级 GPU profile：2 卡，GPU0 峰值 `4958 MiB` / 平均利用率 `27.57%`，GPU1 峰值 `3827 MiB` / 平均利用率 `32.82%`。
- Stage5 已完成阶段级 GPU profile：单卡，GPU0 峰值 `3366 MiB` / 平均利用率 `46.89%`，GPU1 空闲。
- Viewer 已完成常驻 GPU profile：单卡，GPU0 峰值 `1274 MiB` / 平均利用率 `0.27%`，GPU1 空闲。
- 剩余未证明项：从文本到最终 3DGS 的单次自然退出全流程 profile 尚未完成。当前已有 Panogen、Stage1 true-VLM、Stage2 render+caption、Stage3 full attempt、Stage4、Stage5、viewer 的分段峰值证据；Stage3 full attempt 覆盖实际 denoise / WorldMirror / alignment 输出，但命令由人工 SIGTERM 停止，不能证明自然完整退出。按用户要求，不再重跑该长跑阶段。

## FLUX.2 Klein text2img + HY-World img2pano 验证

目标是检查容器内是否已有环境可以跑 `black-forest-labs/FLUX.2-klein-9B`，并验证 text2img 和 HY-World 自带 img2pano。结论如下：

| 组合 | text2img 输出 | img2pano 输出 | 结果 |
| --- | --- | --- | --- |
| `9B` | `/tmp/flux2_klein_tests/base_9b/text2img.png`，`512x512` RGB，像素范围非空 | `/tmp/flux2_klein_tests/base_9b/pano_steps4.png`，`992x512` RGB，像素范围非空 | 通过 |

环境结论：

- `hyworld2` / `hyworld2-pano` 原有环境不适合直接跑 Klein：原先的 `hyworld2-pano` 使用 HY-Pano 依赖栈，未提供 `Flux2KleinPipeline`，并且强行升级 Diffusers 会污染 HY-Pano 依赖。
- 已在 `Dockerfile` 中新增独立 `flux2` 环境，安装 `torch==2.7.1+cu128`、Diffusers main、`transformers==4.57.1`、`accelerate`、`bitsandbytes` 等依赖。
- 容器的 `flux2` import check 通过：`torch 2.7.1+cu128`、`diffusers 0.39.0.dev0`、`transformers 4.57.1`、`bitsandbytes 0.49.2`、`Flux2KleinPipeline`。
- `hyworld2-base:v1.0` 镜像 tag 已更新为包含 `flux2` 环境、`scripts/flux2_klein_text2img.py`、README 和本 changelog 的版本；从镜像直接 `docker run --rm --user root --entrypoint bash ...` 验证 `Flux2KleinPipeline` 导入和脚本 `--help` 均通过。

命令和参数结论：

- text2img 使用 `scripts/flux2_klein_text2img.py`，默认模型路径为 `/models/FLUX.2-klein-9B`，测试参数为 `--height 512 --width 512 --steps 1 --placement offload`。
- HY-World img2pano 使用 `hyworld2/panogen/pipeline_with_qwen_image.py`、`/models/Qwen/Qwen-Image-Edit-2509` 和 `/models/HY-World-2.0/HY-Pano-2.0` LoRA。
- `--load-strategy cpu-offload` 在 Qwen pano 推理中占到约 `31.3 GiB` 并 OOM；切换为脚本支持的 `--load-strategy sequential-offload` 后通过。
- `--num-inference-steps 1` 虽能保存文件，但输出为全黑图，并伴随 scheduler NaN warning，不能作为 img2pano 跑通证据；最终有效验证使用 `--num-inference-steps 4`。
- 直接用 `conda run` 会缓冲命令输出，长推理时容易看起来像卡住；文档命令统一使用 `conda run --no-capture-output` 和 `python -u`。
- `balanced + HY-Pano LoRA` 已定位为不稳定组合：第一步 transformer conditional forward 的 `noise_pred_cond` 即全 NaN/Inf。代码已禁用 `balanced` 并提示使用 `sequential-offload`。
- 最终验证后 GPU 显存回到空闲状态：GPU0 `4 MiB`，GPU1 `3 MiB`，无残留计算进程。

## 容器内从条件图 + prompt 生成 3D 场景

下面是一条已按当前 Dockerfile 默认值整理过的最小操作路径。当前仓库里的 `hyworld2/panogen/pipeline_with_qwen_image.py` 和 `hyworld2/panogen/pipeline.py` 都是 image-conditioned panorama 入口：需要一张输入图作为条件图，再用 `--prompt` 控制场景内容、风格和细节；它们不是纯 text-to-panorama 入口。若只想从纯文字开始，需要先用其他 text-to-image 工具生成一张条件图，或直接准备一张已有 panorama 并跳到 Stage1。

宿主机上先构建、启动并进入容器：

```bash
cd /home/zxh/ws/hyworld2
python docker.py build
python docker.py start
python docker.py verify
python docker.py enter
```

进入容器后，先设置本次 scene 路径和 prompt。`INPUT_IMAGE` 换成你的条件图；输出的全景图固定保存为 `$SCENE/panorama.png`，后续 Stage1-5 都复用同一个 `$SCENE`。

```bash
cd /workspace/hyworld2

SCENE=/workspace/hyworld2/examples/worldgen/my_prompt_scene
RESULT_DIR=$SCENE/gs_results_prompt
INPUT_IMAGE=/workspace/hyworld2/examples/worldgen/case000/panorama.png
PROMPT="a realistic sunny mountain village with stone paths, trees, and distant snow peaks"

mkdir -p "$SCENE"
```

生成全景图。推荐先用 `sequential-offload` 和 4 steps 验证输出非空；如果你已有 `panorama.png`，把已有文件放到 `$SCENE/panorama.png` 后跳过这一步。

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

启动 Stage1/2 需要的 OpenAI-compatible VLM server。建议让它占用 GPU1，Stage1/2 的主进程用 GPU0。

```bash
cd /workspace/hyworld2

CUDA_VISIBLE_DEVICES=1 PORT=8000 scripts/launch_vlm.sh \
  > /tmp/hyworld_vlm.log 2>&1 &
VLM_PID=$!

tail -f /tmp/hyworld_vlm.log
```

另开一个 shell 进入同一个容器，或 `Ctrl-C` 停止 `tail` 后继续执行。Stage1 生成导航目标、navmesh、上视角和 reconstruction iteration；`--force_vlm` 会强制重新请求 VLM。

```bash
cd /workspace/hyworld2/hyworld2/worldgen

CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
  python -u traj_generate.py \
    --target_path "$SCENE" \
    --llm_addr localhost \
    --llm_port 8000 \
    --llm_name Qwen/Qwen3.5-4B \
    --apply_nav_traj \
    --apply_up_route \
    --apply_recon_iteration \
    --force_vlm
```

Stage2 渲染轨迹并让 VLM 写 caption。这里用单进程 GPU0；VLM server 继续占用 GPU1。

```bash
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
  torchrun --nproc_per_node=1 traj_render.py \
    --target_path "$SCENE" \
    --llm_addr localhost \
    --llm_port 8000 \
    --llm_name Qwen/Qwen3.5-4B
```

Stage3 用 WorldStereo 扩展视频并调用 WorldMirror 生成 generation bank。Dockerfile 已默认设置本地模型路径、`WS_TEXT_DTYPE=bf16`、`WS_AUX_OFFLOAD=1` 和 WorldMirror 单卡 512 fallback；通常不需要再手动 export。

```bash
CUDA_VISIBLE_DEVICES=0,1 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
  torchrun --nproc_per_node=2 video_gen.py \
    --target_path "$SCENE" \
    --fsdp \
    --local_files_only
```

Stage4 生成 3DGS 训练数据。

```bash
CUDA_VISIBLE_DEVICES=0,1 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
  torchrun --nproc_per_node=2 gen_gs_data.py \
    --root_path "$SCENE" \
    --save_normal \
    --split_sky
```

Stage5 单卡训练 3DGS。当前 RTX 5090/sm_120 环境下 `fused_ssim` 不稳定，所以用 `--ssim-lambda 0`；`--max-steps 4000` 是本轮验证过的较快配置，可以按质量需求调高。

```bash
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
  python -u -m world_gs_trainer default \
    --data-dir "$SCENE/gs_data" \
    --result-dir "$RESULT_DIR" \
    --max-steps 4000 \
    --save-steps 4000 \
    --eval-steps 4000 \
    --ply-steps 4000 \
    --save-ply \
    --convert-to-spz \
    --disable-video \
    --disable-viewer \
    --use-scale-regularization \
    --antialiased \
    --depth-loss \
    --normal-loss \
    --sky-depth-from-pcd \
    --use-mask-gaussian \
    --mask-export-stochastic \
    --no-mask-export-anchor-protection \
    --use-anchor-protection \
    --export-mesh \
    --ssim-lambda 0 \
    --strategy.refine-start-iter 150 \
    --strategy.refine-stop-iter 2000 \
    --strategy.refine-every 100 \
    --strategy.refine-scale2d-stop-iter 2000 \
    --strategy.reset-every 99990 \
    --strategy.grow-grad2d 0.0001 \
    --strategy.prune-scale3d 0.1
```

查看训练结果：

```bash
CKPT=$(ls "$RESULT_DIR"/ckpts/ckpt_*_rank0.pt | sort -V | tail -1)

CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/bin/conda run --no-capture-output -n hyworld2 \
  python -u show_gs.py \
    --port 8081 \
    --gpu_id 0 \
    --ckpt "$CKPT"
```

常用参数说明：

| 参数 / 环境变量 | 作用 |
| --- | --- |
| `SCENE` | scene 工作目录；必须包含或生成 `panorama.png`，后续中间产物都会写到这里 |
| `PROMPT` | panogen 使用的文本描述；会追加到 Qwen-Image-Edit panorama 正向模板 |
| `INPUT_IMAGE` | panogen 条件图；当前 CLI 不能只给纯文本 prompt |
| `--load-strategy sequential-offload` | Panogen 推荐加载方式；本轮 4-step 验证非空输出通过。`balanced` 的历史 profile 能跑完，但在高分辨率 40-step prompt 场景里观察到 NaN/全黑输出 |
| `--llm_addr` / `--llm_port` / `--llm_name` | Stage1/2 调用 OpenAI-compatible VLM server 的地址、端口和模型名 |
| `--force_vlm` | Stage1 强制重新请求 VLM；复用已有 `objects.json` 时可去掉 |
| `--skip_exist` | Stage1/3 可用于断点续跑，已有产物会跳过 |
| `--fsdp` | Stage3 启用 WorldStereo FSDP，两卡路径用它降低单卡权重压力 |
| `--local_files_only` | Stage3 禁止在线下载，只使用 Dockerfile 默认的 `/models` 本地权重 |
| `WORLDMIRROR_NPROC_PER_NODE=1` | Dockerfile 默认值；让 WorldMirror 走单进程 fallback，避开 2 卡 FSDP illegal memory access |
| `WORLDMIRROR_TARGET_SIZE=512` | Dockerfile 默认值；降低 WorldMirror 后处理显存 |
| `--ssim-lambda 0` | Stage5 跳过当前环境不稳定的 fused SSIM CUDA kernel |

## 变更清单

| 类别 | 文件 / 参数 | 修改 | 目的 | 显存影响 |
| --- | --- | --- | --- | --- |
| 容器 | `Dockerfile` | 新增 CUDA 12.8、conda 环境、worldgen 依赖、`requirements_git.txt` 源码安装 pytorch3d、rtree，并复用 `hyworld2-pano` 运行 VLM shim | 让镜像构建后具备 Stage1-Stage5 所需依赖 | 不是显存优化 |
| 容器 | `Dockerfile` | 安装 `requirements_git.txt` 时设置 `FORCE_CUDA=1 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4` | 确保 PyTorch3D 编译 CUDA rasterizer，避免 Stage1/2 点云渲染报 `Not compiled with GPU support`；限制编译并发避免 CPU 全满 | 不是显存优化；保证使用 GPU rasterization |
| 容器 | `Dockerfile` | 设置 `PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`，并用 `mjun0812/flash-attention-prebuild-wheels` 的 `flash_attn-2.8.2+cu128torch2.7-cp311` wheel 替代源码编译 | 加速镜像构建，避免 flash-attn 本地 CUDA 编译占用大量 CPU | 不是显存优化；构建时间/CPU 优化 |
| 容器 | `Dockerfile` | 默认 `TORCH_CUDA_ARCH_LIST=8.0;8.9;12.0`、`CMAKE_CUDA_ARCHITECTURES=80;89;120` | 同一个镜像构建 CUDA 扩展时覆盖 A100、RTX 4090 和 RTX 5090 | 不是显存优化；扩大运行 GPU 架构兼容范围 |
| 容器 | `Dockerfile` | 安装 `libglm-dev` | gsplat CUDA 扩展需要 `<glm/...>` 头文件；用系统包替代未跟踪的 30MB `csrc/third_party/glm` clone | 不是显存优化 |
| 容器 | `Dockerfile` | 默认 `HF_HOME=/models/.cache/huggingface`、`HUGGINGFACE_HUB_CACHE=/models/.cache/huggingface/hub` | 使用只读模型挂载中的预下载权重，避免运行时下载 | 不是显存优化 |
| 容器 | `Dockerfile` | 默认 `SAM3_REPO_ID=/models/sam3`、`WORLDSTEREO_REPO=/models/WorldStereo`、`WORLDMIRROR_MODEL=/models/HY-World-2.0`、`CAMERA_SELECTOR_MODEL=facebook/dinov2-base` | 消除运行 Stage1/3 时必须手动指定本地模型路径的问题 | 不是显存优化 |
| 容器 | `Dockerfile` | 默认 `WS_TEXT_DTYPE=bf16`、`WS_AUX_OFFLOAD=1`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 将低显存 WorldStereo 路径固化为容器默认 | UMT5 注释记录约 `25 -> 12.5 GiB/card`；MoGe/SAM3 offload 日志中 WorldStereo 前常驻约 `21.3 GiB/card` |
| 容器 | `Dockerfile` | 默认 `WORLDMIRROR_NPROC_PER_NODE=1`、`WORLDMIRROR_TARGET_SIZE=512`、`WORLDMIRROR_CUDA_VISIBLE_DEVICES=0` | 避开 2 卡 FSDP WorldMirror 在 291 张图、832 target size 上的 NCCL/CUDA illegal memory access | 失败路径 OOM/illegal memory；单卡 512 profile 峰值 `14081 MiB` |
| 容器启动 | `docker.py` | 不再把 `HF_HOME` / hub cache 覆盖到 `/cache/huggingface` | 保持 Dockerfile 默认的 `/models/.cache/huggingface`，否则 Wan/MoGe/SAM3/WorldStereo 解析会失败 | 不是显存优化 |
| 容器启动 | `docker.py` | 保留单一 `python docker.py ...` 命令入口，默认镜像为 `hyworld2-base:v1.0` | 降低镜像选择和容器启动复杂度 | 不是显存优化 |
| 容器启动 | `docker.py pull` | 从 `crpi-jq3nu6qbricb9zcb.cn-beijing.personal.cr.aliyuncs.com/zxh_in_bitac/hyworld2:<tag>` 拉取镜像，并默认 retag 为本地 `hyworld2-base:v1.0` | 允许直接使用已发布镜像，不必每台机器本地编译 | 不是显存优化 |
| 容器启动 | `docker.py verify` | 新增 PyTorch3D CUDA rasterizer runtime check，并显式校验 worldgen runtime env defaults | import check 无法发现 CPU-only PyTorch3D；runtime env check 确认 Dockerfile 构建出的容器无需手工再设置本地模型路径和低显存参数 | 不是显存优化 |
| 离线模型 | `worldstereo_wrapper.py` | `_resolve_local_snapshot()` 将 Wan base model repo id 解析到本地 HF snapshot | 避免离线/镜像环境下 diffusers 尝试访问 Hugging Face | 不是显存优化 |
| WorldStereo | `worldstereo_wrapper.py` | `WS_TEXT_DTYPE=bf16` 控制 UMT5/CLIP 加载 dtype | 降低文本和图像编码器权重占用 | 组件估算：UMT5 约节省 `12.5 GiB/card`，CLIP 约减半；需保留全流程实测 |
| WorldStereo | `worldstereo_wrapper.py` | FSDP aux encoder 使用 bf16 mixed precision，且 `WS_AUX_OFFLOAD=1` 时启用 `CPUOffloadPolicy()` | 避免 FSDP forward 中 fp32 compute copy 翻倍占用；编码器不在 denoise loop 常驻 GPU | 注释记录 text+image encoder offload 可释放约 `6 GiB/card` |
| WorldStereo | `worldstereo_wrapper.py` | 非 FSDP 路径安装 encode-aware offload | 单卡/非分布式时也能在 encode 后释放辅助编码器 | 预计释放编码器常驻显存，待采样 |
| WorldStereo | `pipeline_dmd_keyframe.py` | `WS_AUX_OFFLOAD=1` 时 denoise loop 前把 VAE 移到 CPU，decode 前移回 GPU | VAE 在 denoise loop 空闲，避免与 transformer 激活叠加 | 注释记录约释放 `2.7 GiB/card` |
| Stage3 | `video_gen.py` | `SAM3_REPO_ID`、`WORLDSTEREO_REPO` 改为环境可配置 | 支持容器内本地权重默认值 | 不是显存优化 |
| Stage3 | `video_gen.py` | `WS_AUX_OFFLOAD=1` 时 WorldStereo denoise 前把 MoGe/SAM3 移到 CPU，之后移回 | MoGe/SAM3 只在 retrieval/update_memory 用，denoise 时不应常驻 GPU | 注释记录约释放 `4.4 GiB/card`；Claude 日志中 denoise 前 allocated 约 `21.3 GiB/card` |
| Stage3 | `video_gen.py` | WorldMirror 前 `del worldstereo` 并清空 CUDA cache | WorldMirror 子进程会加载独立模型，避免与 WorldStereo FSDP 模型叠加 | 注释记录 WorldStereo transformer 约 `14 GiB/card`；实际避免了 WorldMirror OOM/冲突 |
| Stage3 | `retrieval_wm.py` | SAM3 / DINOv2 / WorldMirror model path 支持环境变量 | 支持本地 gated/cached 模型 | 不是显存优化 |
| Stage3 | `retrieval_wm.py` | WorldMirror 支持 `WORLDMIRROR_NPROC_PER_NODE`、`WORLDMIRROR_TARGET_SIZE`、`WORLDMIRROR_USE_FSDP`、`WORLDMIRROR_ENABLE_BF16`、`WORLDMIRROR_CUDA_VISIBLE_DEVICES` | 允许在 2 卡 FSDP 失败时走单进程 512 target size fallback | 单卡 512 profile 峰值 `14081 MiB`；2 卡 832 失败 |
| Stage3 | `retrieval_wm.py` | 单进程 WorldMirror 子进程清理 `RANK` / `WORLD_SIZE` 等分布式环境变量 | 避免 `python -m worldrecon.pipeline` 误判为分布式运行 | 不是显存优化 |
| 分布式兼容 | `src/sp_utils/communications.py` | 用 `all_gather` 实现 `_all_to_all_single_via_gather()` 替代 NCCL `all_to_all_single` | 避开 Blackwell sm_120 + torch 2.7 上 NCCL all-to-all illegal memory access | 不是显存降低，可能增加临时通信显存；为兼容性修复 |
| WorldMirror | `inference_utils.py` | skyseg ONNX 缺失或 onnxruntime 初始化失败时，`source=auto` 回退到模型预测或全 sky mask | 避免离线缺少 `skyseg.onnx` 时失败 | 不是显存优化 |
| Stage5 | `world_gs_trainer.py` | `cfg.ssim_lambda <= 0` 时跳过 `fused_ssim` CUDA kernel | RTX 5090/sm_120 上 fused SSIM kernel 不兼容时仍可训练 | 不是主要显存优化；成功路径使用 `--ssim-lambda 0` |
| Stage5 | `gsplat/cuda/_backend.py` | `shutil.rmtree(..., ignore_errors=True)` | 预编译 CUDA13 扩展导入失败后，多进程 JIT fallback 的清理 race 不再中断 | 不是显存优化 |
| Stage1/2 | `traj_generate.py` | HF cache 跟随 `HF_HOME` / `HUGGINGFACE_HUB_CACHE`，SAM3 repo 可配置 | 支持容器默认模型挂载和本地 SAM3 | 不是显存优化 |
| Stage1/2 | `traj_generate.py` / `traj_render.py` | CLI help 从 vLLM-specific 改为 OpenAI-compatible VLM server；Stage2 timer 文案同步改名 | 与 `scripts/launch_vlm.sh` 的 transformers/FastAPI shim 保持一致，避免保留过时 vLLM 入口语义 | 不是显存优化 |
| Stage1/2 | `scripts/vlm_server.py` / `scripts/launch_vlm.sh` | 用 `hyworld2-pano` 环境中的 transformers + FastAPI 实现 OpenAI-compatible Qwen3.5-4B shim，并删除旧的 `scripts/launch_vllm.sh` | vLLM/FlashInfer 在 Blackwell 上不可用时，Stage1/2 VLM 仍可服务，且不再保留 vLLM/torch2.11/cu13 运行路径 | Stage2 caption profile 中 GPU1 峰值 `18395 MiB` |
| Stage1/2 文档 | `hyworld2/worldgen/README.md` | 将 VLM 前置条件从必须 vLLM 改为 OpenAI-compatible server，并给出 `scripts/launch_vlm.sh` 示例 | 文档与 Dockerfile 内置 transformers shim 保持一致 | 不是显存优化 |
| Pano | `panogen/pipeline_with_qwen_image.py` | 新增 `--load-strategy`：`cuda`、`balanced`、`cpu-offload`、`sequential-offload` | 允许全景生成在低显存设备上使用 diffusers offload | `balanced` profile 中 GPU0 峰值 `5489 MiB`，GPU1 峰值 `16453 MiB`；默认仍为 `cuda` |
| Pano | `panogen/README.md` | PyTorch 安装说明从 CUDA 11.8 改为 CUDA 12.8 | 与 Dockerfile / Blackwell 环境一致 | 不是显存优化 |
| 仓库卫生 | `.gitignore` / `.dockerignore` | 忽略模型、示例输出、checkpoint、视频等大文件 | 防止生成产物进入 git | 不是显存优化 |
| 仓库卫生 | `scripts/profile_gpu.py` | SIGINT/SIGTERM 时转发终止给被测命令的整个进程组并仍写 JSON summary，添加 `interrupted: true` | 长跑 profile 被人工停止时不再只留下 CSV，也不残留 torchrun worker；Stage3 full attempt 这类测量可复现记录峰值 | 不是显存优化 |
| Docker wrapper | `docker.py` / `scripts/profile_gpu.py` | `docker.py run --profile` 按 wrapper stage 生成 per-stage CSV/JSON，并汇总为 `profile.json` / `profile.md`；`scripts/profile_gpu.py` 支持 `--gpus` 只采样 `--device` 指定的物理卡 | 将历史手工 profile 流程产品化，失败或跳过 stage 也能在最终报告中定位 | 不是显存优化 |
| 仓库卫生 | `scripts/test_vlm.py` | 删除一次性手工 VLM 探测脚本 | 该脚本不属于 Docker verify 或 pipeline 入口，保留会增加维护噪音 | 不是显存优化 |

## 已知失败路径

- Stage3 WorldMirror：`torchrun --nproc_per_node=2 ... --target_size 832 --use_fsdp --enable_bf16` 在 291 张图上出现 NCCL watchdog / CUDA illegal memory access，部分日志也显示接近 31 GiB 卡容量后 OOM。
- Stage3 full natural-exit profile：`video_gen.py --fsdp --local_files_only` 的 full attempt 已产出 33 个 WorldStereo 视频、291 个 WorldMirror depth、`aligned_pcd.ply` 和 `global_pcd.ply`，但在两卡仍接近满载时由人工 SIGTERM 停止；这证明峰值区间，不证明自然退出。用户已明确要求不再重跑该长跑阶段。
- Stage5 2 卡训练：在 gsplat rasterizer / NCCL 路径上仍会触发 CUDA illegal memory access。当前可用路径为单卡训练。
- Stage5 默认 SSIM：`fused_ssim` CUDA kernel 在当前 RTX 5090/sm_120 环境不可用，需 `--ssim-lambda 0`。
- Stage3 alignment：case000 的 generated videos 被判定为 outlier，generation-bank `aligned_pcd.ply` 为 dummy-sized；Stage4/5 仍能从其他数据生成可训练 `gs_data`。

## 显存测量待办

完整测量应覆盖：

1. Panogen：已采样 Qwen-Image-Edit backend，`--load-strategy balanced`，40 steps，模型从 `/models` 本地挂载读取。
2. Stage1：已采样 `traj_generate.py --force_vlm` 的真实 Qwen3.5-4B shim 请求、SAM3、navmesh、up-route 和 recon-iteration 路径；另保留 `--skip_exist` 对 case000 现有输出的 Dockerfile fresh-container 验证。
3. Stage2：已采样 `traj_render.py` 单进程 GPU0 渲染 + GPU1 Qwen3.5-4B shim caption；还未采样多卡渲染配置。
4. Stage3：已采样 full attempt 的实际 WorldStereo video denoise、WorldMirror depth 和 alignment/global-pcd 输出峰值；但命令未自然退出，记录为 `returncode 143` interrupted attempt。另保留 `--skip_exist` load/resume 和 standalone WorldMirror fallback profile。
5. Stage4：`gen_gs_data.py --save_normal --split_sky`。
6. Stage5：单卡 `world_gs_trainer ... --ssim-lambda 0`。
7. Viewer：`show_gs.py` 常驻显存。

测量标准：用 `scripts/profile_gpu.py` 包住每个阶段命令，每秒采样 `nvidia-smi` 的 `memory.used` 和 `utilization.gpu`，按阶段输出每张卡峰值、平均利用率、是否单卡/双卡，以及对应命令。

已采样：

| 阶段 | 命令摘要 | GPU | 峰值显存 | 平均 GPU 利用率 | 证据 |
| --- | --- | --- | --- | --- | --- |
| Panogen Qwen-Image-Edit | `pipeline_with_qwen_image.py --image case000/panorama.png --height 960 --width 1952 --num-inference-steps 40 --load-strategy balanced --pretrained-model-name-or-path /models/Qwen/Qwen-Image-Edit-2509 --lora-path /models/HY-World-2.0` | 0 | `5489 MiB` | `35.13%` | `examples/worldgen/case000/gpu_profile_panogen_qwen_balanced.json` |
| Panogen Qwen-Image-Edit | 同上；GPU1 hosts balanced-loaded model weights | 1 | `16453 MiB` | `0.37%` | `examples/worldgen/case000/gpu_profile_panogen_qwen_balanced.json` |
| Stage1 skip-existing | `python -u traj_generate.py --target_path case000 --apply_nav_traj --apply_up_route --apply_recon_iteration --skip_exist` | 0 | `9801 MiB` | `13.28%` | `examples/worldgen/case000/gpu_profile_stage1_skip.json` |
| Stage1 skip-existing | 同上 | 1 | `3 MiB` | `0.00%` | `examples/worldgen/case000/gpu_profile_stage1_skip.json` |
| Stage1 true-VLM | `CUDA_VISIBLE_DEVICES=0 python -u traj_generate.py --target_path /tmp/hyworld_stage1_vlm --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm --llm_addr localhost --llm_port 8000 --llm_name Qwen/Qwen3.5-4B` with Qwen3.5-4B shim on GPU1 | 0 | `11213 MiB` | `8.15%` | `examples/worldgen/case000/gpu_profile_stage1_vlm.json` |
| Stage1 true-VLM | 同上；GPU1 runs `scripts/launch_vlm.sh` / `scripts/vlm_server.py` | 1 | `17995 MiB` | `2.20%` | `examples/worldgen/case000/gpu_profile_stage1_vlm.json` |
| Stage2 render+caption | `CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 traj_render.py --target_path case000 --llm_addr localhost --llm_port 8000 --llm_name Qwen/Qwen3.5-4B` with Qwen3.5-4B shim on GPU1 | 0 | `3895 MiB` | `45.85%` | `examples/worldgen/case000/gpu_profile_stage2.json` |
| Stage2 render+caption | 同上；GPU1 runs `scripts/launch_vlm.sh` / `scripts/vlm_server.py` | 1 | `18395 MiB` | `40.55%` | `examples/worldgen/case000/gpu_profile_stage2.json` |
| Stage3 resume/load | `torchrun --nproc_per_node=2 video_gen.py --target_path case000 --fsdp --skip_exist --local_files_only` | 0 | `23621 MiB` | `0.42%` | `examples/worldgen/case000/gpu_profile_stage3_skip_load.json` |
| Stage3 resume/load | 同上 | 1 | `23621 MiB` | `1.31%` | `examples/worldgen/case000/gpu_profile_stage3_skip_load.json` |
| Stage3 full attempt | `torchrun --nproc_per_node=2 video_gen.py --target_path /tmp/hyworld_stage3_full --fsdp --local_files_only` | 0 | `32069 MiB` | `73.32%` | `examples/worldgen/case000/gpu_profile_stage3_full.json` |
| Stage3 full attempt | 同上；interrupted after 33 WorldStereo videos, 291 WorldMirror depth files, `aligned_pcd.ply`, and `global_pcd.ply` were observed | 1 | `32107 MiB` | `74.31%` | `examples/worldgen/case000/gpu_profile_stage3_full.json` |
| Stage3 WorldMirror fallback | `python -m hyworld2.worldrecon.pipeline ... --target_size 512 --enable_bf16 --disable_heads normal points gs` | 0 | `14081 MiB` | `35.11%` | `examples/worldgen/case000/gpu_profile_worldmirror.json` |
| Stage3 WorldMirror fallback | 同上 | 1 | `3 MiB` | `0.00%` | `examples/worldgen/case000/gpu_profile_worldmirror.json` |
| Stage4 | `torchrun --nproc_per_node=2 gen_gs_data.py --root_path case000 --save_normal --split_sky` | 0 | `4958 MiB` | `27.57%` | `examples/worldgen/case000/gpu_profile_stage4.json` |
| Stage4 | 同上 | 1 | `3827 MiB` | `32.82%` | `examples/worldgen/case000/gpu_profile_stage4.json` |
| Stage5 | `python -u -m world_gs_trainer ... --max-steps 4000 --ssim-lambda 0 --disable-viewer` | 0 | `3366 MiB` | `46.89%` | `examples/worldgen/case000/gpu_profile_stage5.json` |
| Stage5 | 同上 | 1 | `3 MiB` | `0.00%` | `examples/worldgen/case000/gpu_profile_stage5.json` |
| Viewer | `show_gs.py --ckpt gs_results_profile_stage5/ckpts/ckpt_3999_rank0.pt` | 0 | `1274 MiB` | `0.27%` | `examples/worldgen/case000/gpu_profile_viewer.json` |
| Viewer | 同上 | 1 | `3 MiB` | `0.00%` | `examples/worldgen/case000/gpu_profile_viewer.json` |

Stage5 profile 输出质量指标：

- Result dir: `examples/worldgen/case000/gs_results_profile_stage5`
- PSNR `27.0623`
- SSIM `0.7633`
- LPIPS `0.2907`
- Gaussians `618927`

注意：Stage5 第一次 profile 尝试在当前旧容器中失败，因为清理掉未跟踪 `csrc/third_party/glm` 后，容器还没有 `libglm-dev`。Dockerfile 已声明 `libglm-dev`，并已在当前容器内同步安装后重跑成功。

## 容器构建验证记录

- 第一次重建失败：Docker build context 为 `32.93GB`，根分区被打满。已新增 `.dockerignore`，上下文降至 `263.58MB`。
- 第二次重建进展：CUDA 12.8、`hyworld2`、`hyworld2-pano`、gsplat、flash-attn、pytorch3d、MoGe、fused-ssim、rtree 和 build-time import checks 均通过。
- 第二次重建失败点：Dockerfile 后置层尝试从清华 PyPI 镜像 `pip install --force-reinstall pytorch3d`，但 PyPI 没有 `pytorch3d` 包。该层与前面已成功执行的 `requirements_git.txt` 源码安装重复，已删除。
- 第三次重建中止点：Dockerfile 的 VLM shim 层注释声称使用 transformers shim，但实际执行 `pip install vllm`。这会重新引入已拒绝的 vLLM/FlashInfer/torch 版本风险并显著增加镜像体积，已改为由 `scripts/launch_vlm.sh` 直接使用镜像内 `hyworld2-pano` 环境运行 `scripts/vlm_server.py`。
- 第四次重建成功：`docker build` 产出 `hyworld2-base:v1.0`，build-time import checks 通过：`hyworld2` 环境可导入 `torch/diffusers/transformers/recast/gsplat/worldrecon.pipeline`，`hyworld2-pano` 环境可导入 `pipeline/pipeline_with_qwen_image`。
- Fresh container 验证成功：`python docker.py stop && python docker.py start && python docker.py verify` 通过；`docker.py verify` 会显式断言容器内 `HF_HOME=/models/.cache/huggingface`、`HUGGINGFACE_HUB_CACHE=/models/.cache/huggingface/hub`、`HF_ENDPOINT=https://hf-mirror.com`、`SAM3_REPO_ID=/models/sam3`、`WORLDSTEREO_REPO=/models/WorldStereo`、`WORLDMIRROR_MODEL=/models/HY-World-2.0`、`WS_TEXT_DTYPE=bf16`、`WS_AUX_OFFLOAD=1`、`WORLDMIRROR_NPROC_PER_NODE=1`、`WORLDMIRROR_TARGET_SIZE=512`、`WORLDMIRROR_CUDA_VISIBLE_DEVICES=0` 均为默认值。
- VLM shim 轻量验证成功：`hyworld2-pano` 环境可导入 `fastapi`、`uvicorn`、`AutoModelForImageTextToText`、`AutoProcessor`；未在验证中加载完整 Qwen3.5-4B 权重。
- 第五次重建成功：Dockerfile 使用清华 PyPI 镜像和预编译 flash-attn wheel 后，`docker build` 成功产出 `hyworld2-base:v1.0`；日志确认 `flash-attn-2.8.2+cu128torch2.7` 由 wheel 安装成功，`requirements_git.txt`、PyTorch3D CUDA build、navmesh 和 build-time import checks 均通过。
- Fresh container GPU rasterizer 验证成功：`python docker.py stop && python docker.py start && python docker.py verify` 通过，`PyTorch3D CUDA rasterizer` runtime check 输出 `pytorch3d cuda rasterizer ok (1, 4, 4, 2)`。
- 镜像构建成功：`python docker.py build` 产出 `hyworld2-base:v1.0`；容器为 Ubuntu `24.04`、CUDA `12.8` compiler build；build-time import checks 通过 `hyworld2` / `hyworld2-pano`。
- Fresh container 验证成功：`python docker.py stop && python docker.py start && python docker.py verify` 通过；`hyworld2` / `hyworld2-pano` 均看到 CUDA，PyTorch3D CUDA rasterizer 输出 `pytorch3d cuda rasterizer ok (1, 4, 4, 2)`，模型挂载 `/models` 正常。
- 图生场景短冒烟成功：在 `hyworld2-base` 中启动 README 的 `pipeline_with_qwen_image.py` 条件图 + prompt 入口，使用 `/models/Qwen/Qwen-Image-Edit-2509` 和 `/models/HY-World-2.0/HY-Pano-2.0`，`--num-inference-steps 1`；观察到 GPU0 显存占用约 `15371 MiB` 后按要求停止，进程清理后 GPU 显存恢复。
- Fresh container Panogen profile 验证成功：`pipeline_with_qwen_image.py` 使用 `/models/Qwen/Qwen-Image-Edit-2509` 和 `/models/HY-World-2.0/HY-Pano-2.0`，`--load-strategy balanced`，40 steps 返回码 `0`，输出 `1920 x 960` PNG。总耗时 `718.449s`；GPU0 峰值 `5489 MiB`、平均利用率 `35.13%`，GPU1 峰值 `16453 MiB`、平均利用率 `0.37%`。日志出现 diffusers 后处理 `invalid value encountered in cast` warning，输出文件仅作为 profile 验证产物未纳入 git。
- Fresh container Stage1 profile 验证成功：用刚构建的镜像启动的容器运行 `traj_generate.py --apply_nav_traj --apply_up_route --apply_recon_iteration --skip_exist`，返回码 `0`；GPU0 峰值 `9801 MiB`、平均利用率 `13.28%`，GPU1 峰值 `3 MiB`、平均利用率 `0.00%`。
- Fresh container Stage1 true-VLM profile 验证成功：在容器 `/tmp/hyworld_stage1_vlm` 复制 case000 `panorama.png` 后启动 `scripts/launch_vlm.sh`，运行 `traj_generate.py --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm`，返回码 `0`；完成 meta information VLM labeling、objects VLM labeling、SAM3 segmentation、navmesh path planning、aerial route 和 recon eloop。总耗时 `88.747s`；GPU0 峰值 `11213 MiB`、平均利用率 `8.15%`，GPU1 峰值 `17995 MiB`、平均利用率 `2.20%`。临时 scene 约 `958MB`，已从容器 `/tmp` 删除。
- Fresh container Stage2 profile 验证成功：启动 `scripts/launch_vlm.sh` 后运行 `traj_render.py`，返回码 `0`；33 条轨迹完成点云渲染，28 个 VLM caption 请求完成。GPU0 峰值 `3895 MiB`、平均利用率 `45.85%`；GPU1 峰值 `18395 MiB`、平均利用率 `40.55%`。
- Fresh container Stage3 full attempt 验证记录：在容器 `/tmp/hyworld_stage3_full` 复制 case000 后运行 `video_gen.py --fsdp --local_files_only` 不带 `--skip_exist`。采样 `2788.561s` 后人工 SIGTERM，profile summary 标记 `interrupted: true` / `returncode: 143`；停止前已观察到 33 个 `worldstereo-memory-dmd_result.mp4`、291 个 WorldMirror `depth_*.npy`、`aligned_pcd.ply` 和 `global_pcd.ply`。GPU0 峰值 `32069 MiB`、平均利用率 `73.32%`；GPU1 峰值 `32107 MiB`、平均利用率 `74.31%`。临时副本约 `2.4GB`，已从容器 `/tmp` 删除。
