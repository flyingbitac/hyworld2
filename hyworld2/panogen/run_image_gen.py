"""
HunyuanImage-3.0 Panorama Image Generation Script

Usage:
    # Basic panorama generation
    python run_image_gen.py --image input.png

    # Specify prompt and output path
    python run_image_gen.py --image input.png \\
        --prompt "Expand this image to a 360-degree equirectangular panorama. Maintain realistic style." \\
        --save output_panorama.png

    # Customize inference steps, task type, and system prompt
    python run_image_gen.py --image input.png \\
        --diff-infer-steps 75 --bot-task think_recaption --use-system-prompt en_unified

    # Reproducible generation with a fixed seed
    python run_image_gen.py --image input.png --seed 42 --reproduce

    # Use Taylor Cache to speed up sampling
    python run_image_gen.py --image input.png \\
        --use-taylor-cache --taylor-cache-interval 5 --taylor-cache-order 2
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from hunyuan_image_3 import HunyuanImage3ForCausalMM


def circular_blend_edges(image, blend_width=32):
    """Blend the left and right edges of an image for seamless panorama."""
    image = np.array(image)
    for x in range(blend_width):
        image[:, x, :] = (
            image[:, -blend_width + x, :] * (1 - x / blend_width) +
            image[:, x, :] * (x / blend_width)
        )
    return Image.fromarray(image[:, :-blend_width].astype(np.uint8))


def parse_args():
    parser = argparse.ArgumentParser("Commandline arguments for running HunyuanImage-3 panorama locally")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, default="Expand this image to a 360-degree equirectangular panorama.", help="Prompt to run")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens to generate")
    parser.add_argument("--model-id", type=str,
                        default="/apdcephfs_zwfy/share_303204533/josephzuo/workspace/Panorama/checkpoints/HYImage_3_sft_20260409/hf_convert",
                        help="Path to the model")
    parser.add_argument("--attn-impl", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"],
                        help="Attention implementation. 'flash_attention_2' requires flash attention to be installed.")
    parser.add_argument("--moe-impl", type=str, default="eager", choices=["eager", "flashinfer"],
                        help="MoE implementation. 'flashinfer' requires FlashInfer to be installed.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Use None for random seed.")
    parser.add_argument("--diff-infer-steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--height", type=int, default=960, help="Height of the generated panorama image.")
    parser.add_argument("--width", type=int, default=1952, help="Width of the generated panorama image.")
    parser.add_argument(
        "--use-system-prompt",
        type=str,
        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "en_unified", "custom"],
        default="en_unified",
        help=(
            "Use system prompt. 'None' means no system prompt; 'dynamic' means "
            "the system prompt is determined by --bot-task; 'en_vanilla', "
            "'en_recaption', 'en_think_recaption' and 'en_unified' are four "
            "predefined system prompts; 'custom' means using the custom system "
            "prompt. When using 'custom', --system-prompt must be provided. "
            "Default to load from the model generation config."
        )
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt. Used when --use-system-prompt is 'custom'."
    )
    parser.add_argument(
        "--bot-task",
        type=str,
        choices=["image", "auto", "recaption", "think_recaption"],
        default="think_recaption",
        help=(
            "Type of task for the model. 'image' for direct image generation; "
            "'auto' for text generation; 'recaption' for re-write->image; "
            "'think_recaption' for think->re-write->image. "
            "Default to load from the model generation config."
        )
    )
    parser.add_argument("--save", type=str, default=None, help="Path to save the generated image (default: <input_stem>_panorama.png)")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level")
    parser.add_argument("--blend-width", type=int, default=32, help="Edge blending width for seamless panorama.")

    parser.add_argument("--reproduce", action="store_true", help="Whether to reproduce the results")
    parser.add_argument(
        "--infer-align-image-size",
        action="store_true",
        help="Whether to align the target image size to the src image size."
    )

    # ======================== Taylor Cache ========================
    parser.add_argument("--use-taylor-cache", action="store_true", help="Use Taylor Cache when sampling.")
    parser.add_argument("--taylor-cache-interval", type=int, default=5, help="Interval of Taylor Cache.")
    parser.add_argument("--taylor-cache-order", type=int, default=2, help="Order of Taylor Cache.")
    parser.add_argument(
        "--taylor-cache-enable-first-enhance",
        action="store_true",
        help="Enable first enhance when using Taylor Cache."
    )
    parser.add_argument(
        "--taylor-cache-first-enhance-steps",
        type=int,
        default=3,
        help="First enhance steps when using Taylor Cache (>2)."
    )
    parser.add_argument(
        "--taylor-cache-enable-tailing-enhance",
        action="store_true",
        help="Enable tailing enhance when using Taylor Cache."
    )
    parser.add_argument(
        "--taylor-cache-tailing-enhance-steps",
        type=int,
        default=1,
        help="Tailing enhance steps when using Taylor Cache."
    )
    parser.add_argument(
        "--taylor-cache-low-freqs-order",
        type=int,
        default=2,
        help="Low freqs order when using Taylor Cache."
    )
    parser.add_argument(
        "--taylor-cache-high-freqs-order",
        type=int,
        default=2,
        help="High freqs order when using Taylor Cache."
    )

    return parser.parse_args()


def set_reproducibility(enable, global_seed=None, benchmark=None):
    import torch
    if enable:
        # Configure the seed for reproducibility
        import random
        random.seed(global_seed)
        # Seed the RNG for Numpy
        import numpy as np
        np.random.seed(global_seed)
        # Seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(global_seed)
    # Set following debug environment variable
    # See the link for details: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    if enable:
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Cudnn benchmarking
    torch.backends.cudnn.benchmark = (not enable) if benchmark is None else benchmark
    # Use deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = enable
    torch.use_deterministic_algorithms(enable)


def main(args):
    if args.reproduce:
        set_reproducibility(args.reproduce, global_seed=args.seed)

    if not Path(args.image).exists():
        raise ValueError(f"Input image does not exist: {args.image}")
    if not Path(args.model_id).exists():
        raise ValueError(f"Model path {args.model_id} does not exist")

    # Ensure the panorama instruction is always present in the prompt
    PANO_INSTRUCTION = "Expand this image to a 360-degree equirectangular panorama."
    if PANO_INSTRUCTION not in args.prompt:
        args.prompt = f"{PANO_INSTRUCTION} {args.prompt}".strip()
        print(f"[Info] Panorama instruction prepended. Final prompt: {args.prompt}")

    # Set default output filename if not provided
    if args.save is None:
        input_stem = Path(args.image).stem
        args.save = f"{input_stem}_panorama.png"

    kwargs = dict(
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        moe_impl=args.moe_impl,
        moe_drop_tokens=True,
    )
    model = HunyuanImage3ForCausalMM.from_pretrained(args.model_id, **kwargs)
    model.load_tokenizer(args.model_id)

    print("Start generating panorama image...")
    print(f"Input image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print(f"Output size: {args.height}x{args.width}")
    print(f"Inference steps: {args.diff_infer_steps}")
    print(f"Random seed: {args.seed}")
    print(f"Task type: {args.bot_task}")
    print(f"System prompt: {args.use_system_prompt}")

    cot_text, samples = model.generate_image(
        prompt=args.prompt,
        image=[args.image],
        seed=args.seed,
        image_size=[args.height, args.width],
        use_system_prompt=args.use_system_prompt,
        system_prompt=args.system_prompt,
        bot_task=args.bot_task,
        diff_infer_steps=args.diff_infer_steps,
        verbose=args.verbose,
        max_new_tokens=args.max_new_tokens,
        infer_align_image_size=args.infer_align_image_size,
        use_taylor_cache=args.use_taylor_cache,
        taylor_cache_interval=args.taylor_cache_interval,
        taylor_cache_order=args.taylor_cache_order,
        taylor_cache_enable_first_enhance=args.taylor_cache_enable_first_enhance,
        taylor_cache_first_enhance_steps=args.taylor_cache_first_enhance_steps,
        taylor_cache_enable_tailing_enhance=args.taylor_cache_enable_tailing_enhance,
        taylor_cache_tailing_enhance_steps=args.taylor_cache_tailing_enhance_steps,
        taylor_cache_low_freqs_order=args.taylor_cache_low_freqs_order,
        taylor_cache_high_freqs_order=args.taylor_cache_high_freqs_order,
    )

    # Edge blending post-processing
    output = circular_blend_edges(samples[0], args.blend_width)

    # Save output image
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    output.save(args.save)
    print(f"Image saved to {args.save}")
    if cot_text:
        print(f"Reasoning trace: {cot_text}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
