# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch
from typing import Optional, Dict, Any
import warnings
from types import SimpleNamespace
from torchvision.transforms import Normalize
import torch.nn.functional as F
from loguru import logger


class HFDinoBackbone(torch.nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        config = self.model.config
        self.embed_dim = config.hidden_size
        self.num_register_tokens = getattr(config, "num_register_tokens", 0)
        self.patch_embed = SimpleNamespace(patch_size=getattr(config, "patch_size", "N/A"))

    def forward_features(self, input_img):
        output = self.model(pixel_values=input_img, return_dict=True)
        tokens = output.last_hidden_state
        patch_start = 1 + self.num_register_tokens
        return {
            "x_prenorm": tokens,
            "x_norm_clstoken": tokens[:, 0],
            "x_norm_patchtokens": tokens[:, patch_start:],
        }

    def get_intermediate_layers(self, input_img, intermediate_layers, return_class_token=True):
        output = self.model(
            pixel_values=input_img,
            return_dict=True,
            output_hidden_states=True,
        )
        layers = []
        for layer_index in intermediate_layers:
            tokens = output.hidden_states[layer_index]
            patch_tokens = tokens[:, 1 + self.num_register_tokens:]
            if return_class_token:
                layers.append((patch_tokens, tokens[:, 0]))
            else:
                layers.append(patch_tokens)
        return layers


class Dino(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 224,
        repo_or_dir: Optional[str] = None,
        dino_model: str = "dinov2_vitb14",
        source: str = "local",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        # for backward compatible
        prenorm_features: bool = False,
        freeze_backbone: bool = True,
        prune_network: bool = False,  # False for backward compatible
    ):
        super().__init__()
        if backbone_kwargs is None:
            backbone_kwargs = {}
        if repo_or_dir is None:
            repo_or_dir = os.environ.get("DINO_MODEL", "/models/dinov2-base")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            logger.info(f"Loading DINO model: {dino_model} from {repo_or_dir} (source: {source})")
            if backbone_kwargs:
                logger.info(f"DINO backbone kwargs: {backbone_kwargs}")

            if source == "local" and not os.path.exists(os.path.join(repo_or_dir, "hubconf.py")):
                self.backbone = HFDinoBackbone(repo_or_dir)
            else:
                self.backbone = torch.hub.load(
                    repo_or_dir=repo_or_dir,
                    model=dino_model,
                    source=source,
                    verbose=False,
                    **backbone_kwargs,
                )
            
            # Log model properties after loading
            logger.info(f"Loaded DINO model - type: {type(self.backbone)}, "
                        f"embed_dim: {self.backbone.embed_dim}, "
                        f"patch_size: {getattr(self.backbone.patch_embed, 'patch_size', 'N/A')}")


        self.resize_input_size = (input_size, input_size)
        self.embed_dim = self.backbone.embed_dim
        self.input_size = input_size
        self.input_channels = 3
        self.normalize_images = normalize_images
        self.prenorm_features = prenorm_features
        self.register_buffer('mean', torch.as_tensor([[0.485, 0.456, 0.406]]).view(-1, 1, 1), persistent=False)
        self.register_buffer('std', torch.as_tensor([[0.229, 0.224, 0.225]]).view(-1, 1, 1), persistent=False)

        # freeze
        if freeze_backbone:
            self.requires_grad_(False)
            self.eval()
        elif not prune_network:
            logger.warning(
                "Unfreeze encoder w/o prune parameter may lead to error in ddp/fp16 training"
            )

        if prune_network:
            self._prune_network()

    def _preprocess_input(self, x):
        _resized_images = torch.nn.functional.interpolate(
            x,
            size=self.resize_input_size,
            mode="bilinear",
            align_corners=False,
        )

        if x.shape[1] == 1:
            _resized_images = _resized_images.repeat(1, 3, 1, 1)

        if self.normalize_images:
            _resized_images = _resized_images.sub_(self.mean).div_(self.std)

        return _resized_images

    def _forward_intermediate_layers(
        self, input_img, intermediate_layers, cls_token=True
    ):
        return self.backbone.get_intermediate_layers(
            input_img,
            intermediate_layers,
            return_class_token=cls_token,
        )

    def _forward_last_layer(self, input_img):
        output = self.backbone.forward_features(input_img)
        if self.prenorm_features:
            features = output["x_prenorm"]
            tokens = F.layer_norm(features, features.shape[-1:])
        else:
            tokens = torch.cat(
                [
                    output["x_norm_clstoken"].unsqueeze(1),
                    output["x_norm_patchtokens"],
                ],
                dim=1,
            )
        return tokens

    def forward(self, x, **kwargs):
        _resized_images = self._preprocess_input(x)
        tokens = self._forward_last_layer(_resized_images)
        return tokens.to(x.dtype)

    def _prune_network(self):
        """
        Ran this script:
        out = model(input)
        loss = out.sum()
        loss.backward()

        for name, p in dino_model.named_parameters():
            if p.grad is None:
                print(name)
        model.zero_grad()
        """
        self.backbone.mask_token = None
        if self.prenorm_features:
            self.backbone.norm = torch.nn.Identity()


class DinoForMasks(torch.nn.Module):
    def __init__(
        self,
        backbone: Dino,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.embed_dim

    def forward(self, image, mask):
        return self.backbone.forward(mask)
