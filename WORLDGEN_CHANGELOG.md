# HY-World 2.0 本地修改与显存记录

本文记录当前工作区中为跑通 `examples/worldgen/case000` 的世界生成流水线所做的代码、参数和容器修改。来源包括当前 git diff、`/tmp/claude-1001/-home-zxh-ws-hyworld2/.../tasks/*.output` 中的 Claude 运行日志，以及本轮已验证过的 Stage3-Stage5 / `show_gs` 结果。

## 运行结论

- 已跑通的有效路径：Stage3 使用 2 卡 WorldStereo + 单进程 WorldMirror，Stage4 使用 2 卡，Stage5 使用单卡 3DGS 并设置 `--ssim-lambda 0`。
- `show_gs` 已能加载 `/workspace/hyworld2/examples/worldgen/case000/gs_results_single/ckpts/ckpt_3999_rank0.pt`。
- Stage5 验证指标：PSNR `25.74`，SSIM `0.748`，LPIPS `0.302`，Gaussians `612349`。
- Stage3 resume/load 已完成阶段级 GPU profile：2 卡，`video_gen.py --fsdp --skip_exist` 加载 WorldStereo/FSDP 后发现 33 个已有 render 和 `aligned_pcd.ply`，跳过重算；GPU0/GPU1 峰值均为 `23621 MiB`。
- Stage3 WorldMirror fallback 已完成组件级 GPU profile：单卡，291 张图，target size `512`，GPU0 峰值 `14081 MiB` / 平均利用率 `35.11%`，GPU1 空闲。
- Stage4 已完成阶段级 GPU profile：2 卡，GPU0 峰值 `4958 MiB` / 平均利用率 `27.57%`，GPU1 峰值 `3827 MiB` / 平均利用率 `32.82%`。
- Stage5 已完成阶段级 GPU profile：单卡，GPU0 峰值 `3366 MiB` / 平均利用率 `46.89%`，GPU1 空闲。
- Viewer 已完成常驻 GPU profile：单卡，GPU0 峰值 `1274 MiB` / 平均利用率 `0.27%`，GPU1 空闲。
- 仍需补充：从文本到最终 3DGS 的完整 GPU 峰值采样。当前已有 Stage3 resume/load、Stage4、Stage5、viewer 实测，组件级/阶段日志级和失败路径证据。

## 变更清单

| 类别 | 文件 / 参数 | 修改 | 目的 | 显存影响 |
| --- | --- | --- | --- | --- |
| 容器 | `Dockerfile` | 新增 CUDA 12.8、conda 环境、worldgen 依赖、`requirements_git.txt` 源码安装 pytorch3d、rtree，并复用 `hyworld2-pano` 运行 VLM shim | 让镜像构建后具备 Stage1-Stage5 所需依赖 | 不是显存优化 |
| 容器 | `Dockerfile` | 安装 `libglm-dev` | gsplat CUDA 扩展需要 `<glm/...>` 头文件；用系统包替代未跟踪的 30MB `csrc/third_party/glm` clone | 不是显存优化 |
| 容器 | `Dockerfile` | 默认 `HF_HOME=/models/.cache/huggingface`、`HUGGINGFACE_HUB_CACHE=/models/.cache/huggingface/hub` | 使用只读模型挂载中的预下载权重，避免运行时下载 | 不是显存优化 |
| 容器 | `Dockerfile` | 默认 `SAM3_REPO_ID=/models/sam3`、`WORLDSTEREO_REPO=/models/WorldStereo`、`WORLDMIRROR_MODEL=/models/HY-World-2.0`、`CAMERA_SELECTOR_MODEL=facebook/dinov2-base` | 消除运行 Stage1/3 时必须手动指定本地模型路径的问题 | 不是显存优化 |
| 容器 | `Dockerfile` | 默认 `WS_TEXT_DTYPE=bf16`、`WS_AUX_OFFLOAD=1`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 将低显存 WorldStereo 路径固化为容器默认 | UMT5 注释记录约 `25 -> 12.5 GiB/card`；MoGe/SAM3 offload 日志中 WorldStereo 前常驻约 `21.3 GiB/card` |
| 容器 | `Dockerfile` | 默认 `WORLDMIRROR_NPROC_PER_NODE=1`、`WORLDMIRROR_TARGET_SIZE=512`、`WORLDMIRROR_CUDA_VISIBLE_DEVICES=0` | 避开 2 卡 FSDP WorldMirror 在 291 张图、832 target size 上的 NCCL/CUDA illegal memory access | 失败路径 OOM/illegal memory；单卡 512 profile 峰值 `14081 MiB` |
| 容器启动 | `docker.py` | 不再把 `HF_HOME` / hub cache 覆盖到 `/cache/huggingface` | 保持 Dockerfile 的 `/models/.cache/huggingface` 默认，否则 Wan/MoGe/SAM3/WorldStereo 解析会失败 | 不是显存优化 |
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
| Stage1/2 | `scripts/vlm_server.py` / `scripts/launch_vlm.sh` | 用 `hyworld2-pano` 环境中的 transformers + FastAPI 实现 OpenAI-compatible Qwen3-VL shim，并删除旧的 `scripts/launch_vllm.sh` | vLLM/FlashInfer 在 Blackwell 上不可用时，Stage1/2 VLM 仍可服务，且不再保留 vLLM/torch2.11/cu13 运行路径 | 使用 1 张 GPU；峰值待测 |
| Pano | `panogen/pipeline_with_qwen_image.py` | 新增 `--load-strategy`：`cuda`、`balanced`、`cpu-offload`、`sequential-offload` | 允许全景生成在低显存设备上使用 diffusers offload | 待单独采样；默认仍为 `cuda` |
| Pano | `panogen/README.md` | PyTorch 安装说明从 CUDA 11.8 改为 CUDA 12.8 | 与 Dockerfile / Blackwell 环境一致 | 不是显存优化 |
| 仓库卫生 | `.gitignore` / `.dockerignore` | 忽略模型、示例输出、checkpoint、视频等大文件 | 防止生成产物进入 git | 不是显存优化 |

## 已知失败路径

- Stage3 WorldMirror：`torchrun --nproc_per_node=2 ... --target_size 832 --use_fsdp --enable_bf16` 在 291 张图上出现 NCCL watchdog / CUDA illegal memory access，部分日志也显示接近 31 GiB 卡容量后 OOM。
- Stage5 2 卡训练：在 gsplat rasterizer / NCCL 路径上仍会触发 CUDA illegal memory access。当前可用路径为单卡训练。
- Stage5 默认 SSIM：`fused_ssim` CUDA kernel 在当前 RTX 5090/sm_120 环境不可用，需 `--ssim-lambda 0`。
- Stage3 alignment：case000 的 generated videos 被判定为 outlier，generation-bank `aligned_pcd.ply` 为 dummy-sized；Stage4/5 仍能从其他数据生成可训练 `gs_data`。

## 显存测量待办

完整测量应覆盖：

1. Panogen：文本到 panorama。
2. Stage1：`traj_generate.py` + VLM shim，记录 VLM GPU 与主进程 GPU。
3. Stage2：`traj_render.py` 多卡渲染。
4. Stage3：补充完整重算 profile；当前已采样 `video_gen.py --fsdp --skip_exist` 的 WorldStereo/FSDP load + resume skip 路径，以及 standalone WorldMirror fallback，尚未覆盖实际 video denoise 和 alignment 的完整峰值。
5. Stage4：`gen_gs_data.py --save_normal --split_sky`。
6. Stage5：单卡 `world_gs_trainer ... --ssim-lambda 0`。
7. Viewer：`show_gs.py` 常驻显存。

测量标准：用 `scripts/profile_gpu.py` 包住每个阶段命令，每秒采样 `nvidia-smi` 的 `memory.used` 和 `utilization.gpu`，按阶段输出每张卡峰值、平均利用率、是否单卡/双卡，以及对应命令。

已采样：

| 阶段 | 命令摘要 | GPU | 峰值显存 | 平均 GPU 利用率 | 证据 |
| --- | --- | --- | --- | --- | --- |
| Stage3 resume/load | `torchrun --nproc_per_node=2 video_gen.py --target_path case000 --fsdp --skip_exist --local_files_only` | 0 | `23621 MiB` | `0.42%` | `examples/worldgen/case000/gpu_profile_stage3_skip_load.json` |
| Stage3 resume/load | 同上 | 1 | `23621 MiB` | `1.31%` | `examples/worldgen/case000/gpu_profile_stage3_skip_load.json` |
| Stage3 WorldMirror fallback | `python -m hyworld2.worldrecon.pipeline ... --target_size 512 --enable_bf16 --disable_heads normal points gs` | 0 | `14081 MiB` | `35.11%` | `examples/worldgen/case000/gpu_profile_worldmirror.json` |
| Stage3 WorldMirror fallback | 同上 | 1 | `3 MiB` | `0.00%` | `examples/worldgen/case000/gpu_profile_worldmirror.json` |
| Stage4 | `torchrun --nproc_per_node=2 gen_gs_data.py --root_path case000 --save_normal --split_sky` | 0 | `4958 MiB` | `27.57%` | `examples/worldgen/case000/gpu_profile_stage4.json` |
| Stage4 | 同上 | 1 | `3827 MiB` | `32.82%` | `examples/worldgen/case000/gpu_profile_stage4.json` |
| Stage5 | `python -m world_gs_trainer ... --max_steps 4000 --ssim-lambda 0 --disable-viewer` | 0 | `3366 MiB` | `46.89%` | `examples/worldgen/case000/gpu_profile_stage5.json` |
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
- 第四次重建成功：`docker build` 产出 `hyworld2-isaaclab:3.0.0-beta2`，build-time import checks 通过：`hyworld2` 环境可导入 `torch/diffusers/transformers/recast/gsplat/worldrecon.pipeline`，`hyworld2-pano` 环境可导入 `pipeline/pipeline_with_qwen_image`。
- Fresh container 验证成功：`python docker.py stop && python docker.py start && python docker.py verify` 通过；容器内 `HF_HOME=/models/.cache/huggingface`、`HUGGINGFACE_HUB_CACHE=/models/.cache/huggingface/hub`、`SAM3_REPO_ID=/models/sam3`、`WORLDSTEREO_REPO=/models/WorldStereo`、`WORLDMIRROR_MODEL=/models/HY-World-2.0`、`WS_TEXT_DTYPE=bf16`、`WS_AUX_OFFLOAD=1`、`WORLDMIRROR_NPROC_PER_NODE=1`、`WORLDMIRROR_TARGET_SIZE=512` 均为默认值。
- VLM shim 轻量验证成功：`hyworld2-pano` 环境可导入 `fastapi`、`uvicorn`、`AutoModelForImageTextToText`、`AutoProcessor`；未在验证中加载完整 Qwen3-VL 权重。
