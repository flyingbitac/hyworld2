能跑，但要分清 **“原始 BF16 权重直接跑”** 和 **“量化/卸载后跑”**。

**结论：**

| GPU           |       显存 | 能不能跑 FLUX.2-dev 推理      | 推荐方式                                             |
| ------------- | -------: | ----------------------- | ------------------------------------------------ |
| RTX 4090      |     24GB | **能跑**，但不能直接全 BF16 装进显存 | 4-bit / FP8 + CPU offload，最好 remote text encoder |
| RTX 5090      |     32GB | **能跑，而且比 4090 舒服很多**    | 4-bit 最稳；FP8/ComfyUI 也可，但仍可能需要 offload           |
| 80GB 卡，如 H100 |     80GB | BF16 也要 CPU offload 才稳  | 官方说 H100 全组件同时放不下                                |
| H200/B200 级别  | 141GB/更大 | 才比较接近“直接全塞显存”           | BF16 全量更合适                                       |

FLUX.2-dev 是 **32B 参数**模型，模型卡明确说它是 32 billion parameter rectified flow transformer。官方 Diffusers 博客也写得很清楚：FLUX.2 的 DiT + Mistral3 Small text encoder 如果不做任何 offload，推理需要 **超过 80GB VRAM**；一个 H100 上启用 CPU offload 的示例仍需要约 **62GB**。([Hugging Face][1])

所以：

**4090/5090 都不能按原始 BF16 全量舒舒服服直接跑。**
但官方专门给了 **RTX 4090 / RTX 5090 这类 24–32GB 显存卡**的方案：用 4-bit 量化。BFL 的 GitHub 文档写明，24–32GB VRAM 可以用 4-bit quantization；其中 “4-bit transformer + remote text encoder” 大约 **18GB VRAM**，“4-bit transformer + 4-bit text encoder” 大约 **20GB VRAM**。([GitHub][2])

显卡规格上，4090 是 **24GB GDDR6X**，5090 是 **32GB GDDR7**，所以官方把它们归到 “Lower VRAM 24–32G - RTX 4090 and 5090” 这档是合理的。([NVIDIA][3])

我的建议：

**4090：**
可以跑，但最好别折腾 BF16 原版。用：

```python
repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
```

再配合：

```python
pipe.enable_model_cpu_offload()
```

如果能接受 Hugging Face 的 remote text encoder，就用 remote text encoder 版本，显存压力最低，大概 18GB VRAM。这样 4090 的 24GB 是够的。缺点是依赖网络和 HF token。

**5090：**
也建议 4-bit 或 FP8，不建议幻想 BF16 全量装进 32GB。5090 的 32GB 会比 4090 更宽裕，尤其是高分辨率、多参考图、ComfyUI workflow 复杂时更不容易爆显存。但完整 BF16 仍然远超 32GB。

**ComfyUI：**
也能跑。NVIDIA 官方博客说 FLUX.2 已针对 RTX GPU 做了 FP8 量化和 ComfyUI 优化，FP8 可降低 VRAM 需求并提升性能，但完整模型原始加载约 90GB，低显存模式也约 64GB，因此消费卡主要依赖 FP8 + weight streaming/offload。([NVIDIA Blog][4])

一句话：**如果你只是本地推理，4090 能跑，5090 更稳；但都应该走 4-bit/FP8 + offload，而不是原始 BF16 全量。** 如果你想做 LoRA 训练，4090/5090 也不是完全没戏，但会比推理麻烦很多，基本必须上 QLoRA/FP8、latent cache、remote text encoder、gradient checkpointing 这些省显存手段。

[1]: https://huggingface.co/black-forest-labs/FLUX.2-dev "black-forest-labs/FLUX.2-dev · Hugging Face"
[2]: https://github.com/black-forest-labs/flux2/blob/main/docs/flux2_dev_hf.md "flux2/docs/flux2_dev_hf.md at main · black-forest-labs/flux2 · GitHub"
[3]: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/?utm_source=chatgpt.com "GeForce RTX 4090 Graphics Cards for Gaming"
[4]: https://blogs.nvidia.com/blog/rtx-ai-garage-flux-2-comfyui/?utm_source=chatgpt.com "FLUX.2 Image Generation Models Now Released"


结论：**FLUX.2-klein-4B 和 9B 在 4090/5090 上比 FLUX.2-dev 好跑很多**。它们不是 dev 的量化版，而是小模型家族，默认就是 **4-step distilled**，所以用法、步数、显存策略都不一样。

| 模型                  |                4090 24GB |              5090 32GB | 推荐用法                     |
| ------------------- | -----------------------: | ---------------------: | ------------------------ |
| **FLUX.2-klein-4B** |                      很轻松 |                    很轻松 | BF16 直接跑或 offload        |
| **FLUX.2-klein-9B** | 能跑，但建议 CPU offload / FP8 | BF16 可跑但显存紧，offload 更稳 | BF16+offload，或 FP8/NVFP4 |
| **FLUX.2-dev 32B**  |       必须 4-bit / offload |    也建议 4-bit / offload | `Flux2Pipeline` + 量化     |

官方 model card 写，**klein-4B 约 13GB VRAM**，可在 RTX 3090/4070 及以上跑；**klein-9B 约 29GB VRAM**，官方说 RTX 4090 及以上可用。4090 是 24GB，5090 是 32GB，所以 9B 在 4090 上不要理解成“全模型都塞进显存”，而应理解成“配合 CPU offload / 量化可用”；5090 跑 9B BF16 更接近原生可用，但 29GB 对 32GB 也很贴边。([Hugging Face][1])

和 FLUX.2-dev 最大的用法区别是：**Klein 用 `Flux2KleinPipeline`，不用你手动加载 Mistral text encoder，也不需要 remote text encoder**。官方 Klein 示例直接从模型仓库加载 pipeline，然后 `num_inference_steps=4`、`guidance_scale=1.0`；而 dev 的官方示例是 `Flux2Pipeline`，并给了 4-bit + remote text encoder 的 4090 路线。([Hugging Face][2])

## 4090/5090 都能跑的本地伪代码

```python
import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image

# 二选一：
# model_id = "black-forest-labs/FLUX.2-klein-4B"
model_id = "black-forest-labs/FLUX.2-klein-9B"

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
)

# 稳妥写法：4090 / 5090 都建议先这样跑通
# 特别是 9B on 4090，基本应该开
pipe.enable_model_cpu_offload()

# 如果是 4B，而且你想追求速度，可以尝试改成：
# pipe.to(device)
#
# 如果是 9B：
# - 4090：不建议直接 pipe.to("cuda")
# - 5090：可以尝试 pipe.to("cuda")，但显存很贴边，失败就换回 offload

prompt = """
A cinematic photo of a small robot repairing a vintage radio on a wooden desk,
warm morning light, shallow depth of field, realistic, highly detailed.
"""

with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=4,   # Klein 默认核心区别：4 steps
        guidance_scale=1.0,      # Klein 官方示例是 1.0
        generator=torch.Generator(device=device).manual_seed(42),
    ).images[0]

image.save("flux2_klein_output.png")
```

## 图像编辑 / 参考图用法

```python
input_image = load_image("./input.png").convert("RGB")

prompt = """
Turn this object into a matte black industrial drone while preserving the camera angle,
lighting, and background.
"""

with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        image=input_image,       # 单图编辑
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=torch.Generator(device=device).manual_seed(123),
    ).images[0]

image.save("flux2_klein_edit.png")
```

多参考图大概这样：

```python
ref1 = load_image("./object.png").convert("RGB")
ref2 = load_image("./style.png").convert("RGB")

image = pipe(
    prompt="Combine the object from the first image with the style of the second image.",
    image=[ref1, ref2],
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=1.0,
).images[0]
```

## 和 FLUX.2-dev 的关键差异

**第一，模型规模不同。** FLUX.2-dev 是 **32B**，Klein-9B 是 **9B flow model + 8B Qwen3 text embedder**，Klein-4B 是 **4B rectified flow transformer**。Klein 是为了低延迟和消费级硬件设计的，dev 则是最大质量、无延迟约束时优先。([Hugging Face][3])

**第二，步数差很多。** Klein 4B/9B 是 step-distilled，官方建议 distilled 版本用于生产和实时生成，默认 **4 steps**；FLUX.2-dev 通常是 28–50 steps 这种量级。([GitHub][4])

**第三，加载方式更简单。** dev 在 4090/5090 上通常要走 `diffusers/FLUX.2-dev-bnb-4bit`、手动 text encoder / transformer、CPU offload；Klein 直接：

```python
pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
```

**第四，4B 的商业使用更宽松。** Klein-4B 是 Apache 2.0；Klein-9B 和 FLUX.2-dev 是 FLUX Non-Commercial License，并且 9B Hugging Face 页面需要先同意条款才能下载。([Hugging Face][1])

## 我的建议

你如果只是想在 **4090/5090 本地生成/编辑图**：

**优先试 `FLUX.2-klein-4B`**。它的质量不一定追上 dev，但速度、显存、部署体验明显更好，而且 4090 上压力小很多。

如果你想要更好的质量，又不想用 dev 那种 32B 大模型：

**试 `FLUX.2-klein-9B`**。4090 上用 `enable_model_cpu_offload()`；5090 上可以先试 BF16 直接上 GPU，失败再 offload。官方还发布了 Klein 的 FP8 / NVFP4 版本，BFL 说 FP8 最多可减少约 40% VRAM，NVFP4 最多可减少约 55% VRAM，尤其适合 RTX 5080/5090 这类卡。([Black Forest Labs][5])

[1]: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B "black-forest-labs/FLUX.2-klein-4B · Hugging Face"
[2]: https://huggingface.co/black-forest-labs/FLUX.2-klein-9B "black-forest-labs/FLUX.2-klein-9B · Hugging Face"
[3]: https://huggingface.co/black-forest-labs/FLUX.2-dev "black-forest-labs/FLUX.2-dev · Hugging Face"
[4]: https://github.com/black-forest-labs/flux2 "GitHub - black-forest-labs/flux2: Official inference repo for FLUX.2 models · GitHub"
[5]: https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence "FLUX.2 [klein]: Towards Interactive Visual Intelligence | Black Forest Labs"
