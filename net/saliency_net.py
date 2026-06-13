#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPT front-end using a DINOv3 ViT backbone (Hugging Face OR Torch Hub).

This version **does NOT use a separate timm ViT**. Instead, it:
- runs DINOv3,
- grabs 4 transformer layers (hidden_states),
- converts those token sequences to feature maps,
- builds a 4-level pyramid (/4, /8, /16, /32),
- and feeds that into the standard DPT decoder.

Quick notes:
- HF path: pass an access token via env (HF_TOKEN or HUGGING_FACE_HUB_TOKEN) or hf_token=...
- This version does NOT require AutoImageProcessor (works with older Transformers),
  but if it’s available we’ll use its mean/std automatically.
- Torch Hub path STILL requires a local/URL checkpoint for dinov3_weights.
"""

import os
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Your DPT import (unchanged, from your repo)
from DPT.dpt.models import DPT  # noqa: E402

# HF imports (processor is optional)
try:
    from transformers import AutoModel  # type: ignore
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

try:
    from transformers import AutoImageProcessor  # type: ignore
    _HAS_IMAGE_PROCESSOR = True
except Exception:
    _HAS_IMAGE_PROCESSOR = False


class OpenVAMSaliencyNet(DPT):
    """
    DPT front-end that uses a DINOv3 ViT backbone, taps several DINO transformer
    blocks directly (no extra timm ViT), and constructs a 4-level pyramid
    (/4, /8, /16, /32) before the standard DPT refinenets.
    """

    def __init__(
        self,
        # Torch Hub path (if use_hf=False)
        dinov3_repo_dir_or_name: str = "facebookresearch/dinov3",
        dinov3_variant: str = "dinov3_vitb16",       # e.g. dinov3_vitb16
        dinov3_weights: Optional[str] = None,        # REQUIRED for Torch Hub path

        # Hugging Face path
        use_hf: bool = False,
        hf_model_name: Optional[str] = None,         # e.g. "facebook/dinov3-vitb16-pretrain-lvd1689m"
        hf_torch_dtype: Optional[torch.dtype] = None,
        hf_token: Optional[str] = None,              # pass token directly or via env
        hf_local_files_only: bool = False,           # True: load DINO from HF cache only (no Hub HTTP)

        # DPT + backbone settings
        apply_dino_norm: bool = True,                # ImageNet mean/std if not using HF processor stats
        backbone: str = "vitb_rn50_384",             # used by base DPT to size scratch, etc.
        features: int = 256,
        readout: str = "project",
        channels_last: bool = False,
        use_bn: bool = False,
        enable_attention_hooks: bool = False,
        upsample_output_to_input_res: bool = True,
        out_channels: int = 1,
        use_sigmoid: bool = True,
        freeze_dino: bool = True,
        freeze_vit: bool = False,                    # kept for API compatibility, unused
    ):
        # ----- Build the DPT head -----
        head_layers = [
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
        ]
        if upsample_output_to_input_res:
            head_layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        head_layers += [
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        ]
        if use_sigmoid:
            head_layers.append(nn.Sigmoid())
        head = nn.Sequential(*head_layers)

        # Base DPT (we will NOT use its encoder forward, only its scratch/decoder)
        super().__init__(
            head=head,
            features=features,
            backbone=backbone,
            readout=readout,
            channels_last=channels_last,
            use_bn=use_bn,
            enable_attention_hooks=enable_attention_hooks,
        )
        
        # 🔴 This removes the unused timm backbone completely
        if hasattr(self, "pretrained"):
            del self.pretrained
            
        # ===== Choose HF or Torch Hub backbone =====
        self.use_hf_dino = bool(use_hf)
        self.hf_processor = None  # we’ll try to load it if available
        self.hf_norm_from_processor = False

        if self.use_hf_dino:
            assert _HAS_TRANSFORMERS, "Install `transformers` to use Hugging Face DINOv3 (pip install transformers)."

            # Map Hub-style variants to HF ids if user didn’t pass one
            _vit_map = {
                "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "dinov3_vith14": "facebook/dinov3-vith14-pretrain-lvd1689m",
            }
            hf_id = hf_model_name or _vit_map.get(dinov3_variant)
            if hf_id is None:
                raise ValueError("Provide `hf_model_name` or choose a mapped variant.")

            token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

            # Load the ViT backbone
            try:
                self.dino = AutoModel.from_pretrained(
                    hf_id,
                    torch_dtype=hf_torch_dtype,
                    token=token,
                    local_files_only=hf_local_files_only,
                )
            except Exception as e:
                msg = str(e).lower()
                if "401" in msg or "403" in msg or "gated" in msg or "not in the authorized list" in msg:
                    raise PermissionError(
                        "HF reports restricted/gated access. Ensure access is granted on the model page and a valid token is set."
                    ) from e
                raise

            if freeze_dino:
                self.dino.eval()
                for p in self.dino.parameters():
                    p.requires_grad = False

            # Config-driven properties
            self.patch_size = int(getattr(self.dino.config, "patch_size", 16))
            self.num_register_tokens = int(getattr(self.dino.config, "num_register_tokens", 4))

            # Try to read mean/std from the (new) HF image processor. If it’s not available
            # in your transformers version, we gracefully fall back to ImageNet stats.
            if _HAS_IMAGE_PROCESSOR and apply_dino_norm:
                try:
                    self.hf_processor = AutoImageProcessor.from_pretrained(
                        hf_id, token=token, local_files_only=hf_local_files_only
                    )
                    mean = torch.tensor(self.hf_processor.image_mean).view(1, 3, 1, 1)
                    std = torch.tensor(self.hf_processor.image_std).view(1, 3, 1, 1)
                    self.register_buffer("hf_mean", mean)
                    self.register_buffer("hf_std", std)
                    self.hf_norm_from_processor = True
                except Exception:
                    self.hf_processor = None  # proceed with ImageNet stats

        else:
            # Torch Hub path (requires weights path/URL)
            import os as _os

            def _load_dino(repo_or_dir, variant, weights, **hub_kwargs):
                if not weights:
                    raise ValueError("Torch Hub DINOv3 requires `dinov3_weights` (local path or URL).")
                return torch.hub.load(repo_or_dir, variant, weights=weights, **hub_kwargs)

            hub_kwargs = {"source": "local"} if _os.path.isdir(dinov3_repo_dir_or_name) else {}
            try:
                self.dino = _load_dino(dinov3_repo_dir_or_name, dinov3_variant, dinov3_weights, **hub_kwargs)
            except Exception:
                self.dino = _load_dino("facebookresearch/dinov3", dinov3_variant, dinov3_weights)

            if freeze_dino:
                for p in self.dino.parameters():
                    p.requires_grad = False

            ps = getattr(self.dino, "patch_size", None)
            if ps is None and hasattr(self.dino, "patch_embed"):
                ps = getattr(self.dino.patch_embed, "patch_size", None)
            if isinstance(ps, (tuple, list)):
                ps = ps[0]
            self.patch_size = int(ps) if ps is not None else 16
            self.num_register_tokens = int(
                getattr(self.dino, "num_register_tokens", 0)
                or getattr(getattr(self.dino, "config", object()), "num_register_tokens", 0)
                or 4
            )

        # Normalization buffers
        # If we got HF processor stats, use those; else default to ImageNet stats.
        use_imagenet_stats = apply_dino_norm and (not self.use_hf_dino or not self.hf_norm_from_processor)
        if use_imagenet_stats:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("dino_mean", mean)
            self.register_buffer("dino_std", std)

        # ----- Choose which DINO transformer layers to tap -----
        # We’ll take 4 roughly evenly-spaced transformer blocks.
        # DINO HF models expose num_hidden_layers in the config.
        num_layers = getattr(getattr(self.dino, "config", object()), "num_hidden_layers", 12)
        self.dino_layers = [
            max(1, int(num_layers * 0.25)),
            max(1, int(num_layers * 0.5)),
            max(1, int(num_layers * 0.75)),
            num_layers,
        ]

        # ----- Adapters into DPT scratch (DINO dim -> DPT channels) -----
        in_ch_l1 = self.scratch.layer1_rn.in_channels
        in_ch_l2 = self.scratch.layer2_rn.in_channels
        in_ch_l3 = self.scratch.layer3_rn.in_channels
        in_ch_l4 = self.scratch.layer4_rn.in_channels

        # We don’t know DINO hidden_size at construction time -> LazyConv2d
        self.proj_to_l1 = nn.LazyConv2d(in_ch_l1, kernel_size=1)
        self.proj_to_l2 = nn.LazyConv2d(in_ch_l2, kernel_size=1)
        self.proj_to_l3 = nn.LazyConv2d(in_ch_l3, kernel_size=1)
        self.proj_to_l4 = nn.LazyConv2d(in_ch_l4, kernel_size=1)

    @staticmethod
    def _safe_div_size(H: int, W: int, div: int) -> Tuple[int, int]:
        return (max(1, H // div), max(1, W // div))

    def _dino_tokens_to_map(self, tokens: torch.Tensor, H_img: int, W_img: int) -> torch.Tensor:
        """
        tokens: [B, T, D] -> map: [B, D, h, w]
        Drops CLS + register tokens if present.
        """
        B, T, D = tokens.shape
        n_drop = 1 + max(getattr(self, "num_register_tokens", 0), 0)
        patch_tok = tokens[:, n_drop:, :] if T > n_drop else tokens

        h = max(1, H_img // self.patch_size)
        w = max(1, W_img // self.patch_size)
        N = patch_tok.shape[1]

        # If token count doesn't match expected hw, fall back to sqrt heuristic
        if h * w != N:
            h = int(round(math.sqrt(N)))
            w = max(1, N // h)

        return patch_tok.permute(0, 2, 1).contiguous().view(B, D, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W]
        returns: [B,out_channels,H,W] if head has x2 upsample; else [B,out_channels,H/2,W/2]
        """
        B, _, H_img, W_img = x.shape

        # ========= A) DINOv3 encode -> transformer hidden states =========
        hidden = None
        tokens_last = None
        H_eff, W_eff = H_img, W_img  # effective size (after padding) used for token->map

        if self.use_hf_dino:
            # Prefer HF processor stats if available, else ImageNet stats
            if self.hf_norm_from_processor:
                x_norm = (x - self.hf_mean) / self.hf_std
            else:
                x_norm = (x - self.dino_mean) / self.dino_std

            # Pad to multiples of patch size so token grid lines up with input
            H_eff = math.ceil(H_img / self.patch_size) * self.patch_size
            W_eff = math.ceil(W_img / self.patch_size) * self.patch_size
            pad = (0, W_eff - W_img, 0, H_eff - H_img)  # (w_left, w_right, h_top, h_bottom)
            x_pad = F.pad(x_norm, pad, mode="reflect") if (pad[1] or pad[3]) else x_norm

            with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.dino.parameters())):
                # IMPORTANT: ask DINO to return all hidden states
                out = self.dino(pixel_values=x_pad, output_hidden_states=True)

            tokens_last = out.last_hidden_state          # [B, 1+reg+N, D]
            hidden = out.hidden_states                   # tuple length = num_layers + 1

        else:
            # Torch Hub path (no hidden_states by default -> we’ll just use last tokens 4x)
            x_in = (x - self.dino_mean) / self.dino_std  # ImageNet stats
            dino_out = self.dino
            with torch.set_grad_enabled(self.training and any(p.requires_grad for p in dino_out.parameters())):
                if hasattr(dino_out, "forward_features"):
                    feats = dino_out.forward_features(x_in)
                    if isinstance(feats, dict):
                        if "x_norm_patchtokens" in feats:
                            tokens = feats["x_norm_patchtokens"]
                        elif "x" in feats and getattr(feats["x"], "dim", lambda: 0)() == 3:
                            tokens = feats["x"]
                        else:
                            raise RuntimeError("Unexpected DINOv3 forward_features() output.")
                    else:
                        tokens = feats
                else:
                    out = dino_out(x_in)
                    if isinstance(out, dict) and "last_hidden_state" in out:
                        tokens = out["last_hidden_state"]
                    elif hasattr(out, "last_hidden_state"):
                        tokens = out.last_hidden_state
                    elif torch.is_tensor(out) and out.dim() == 3:
                        tokens = out
                    else:
                        raise RuntimeError("Unknown DINOv3 output format.")

            tokens_last = tokens  # [B, 1+reg+N, D]

        # ========= B) Convert 4 chosen DINO layers into feature maps =========
        feats = []

        if hidden is not None:
            # HF path: use real intermediate transformer layers
            for idx in self.dino_layers:
                # hidden_states[0] is embeddings; clamp to valid range [1, L]
                idx = int(idx)
                idx = min(max(idx, 1), len(hidden) - 1)
                tokens_l = hidden[idx]  # [B, 1+reg+N, D]
                m = self._dino_tokens_to_map(tokens_l, H_eff, W_eff)  # [B, D, h, w]
                feats.append(m)
        else:
            # Torch Hub fallback: we only have final tokens -> reuse them 4 times
            m = self._dino_tokens_to_map(tokens_last, H_eff, W_eff)
            feats = [m, m, m, m]

        if len(feats) != 4:
            raise RuntimeError(f"Expected 4 tapped features, got {len(feats)} (HF layers: {getattr(self, 'dino_layers', None)})")

        # ========= C) Build multi-scale pyramid (/4, /8, /16, /32) =========
        target_sizes = [
            self._safe_div_size(H_img, W_img, 4),
            self._safe_div_size(H_img, W_img, 8),
            self._safe_div_size(H_img, W_img, 16),
            self._safe_div_size(H_img, W_img, 32),
        ]
        f1 = F.interpolate(feats[0], size=target_sizes[0], mode="bilinear", align_corners=False)
        f2 = F.interpolate(feats[1], size=target_sizes[1], mode="bilinear", align_corners=False)
        f3 = F.interpolate(feats[2], size=target_sizes[2], mode="bilinear", align_corners=False)
        f4 = F.interpolate(feats[3], size=target_sizes[3], mode="bilinear", align_corners=False)

        # Project to DPT channels + fusion
        l1_rn = self.scratch.layer1_rn(self.proj_to_l1(f1))
        l2_rn = self.scratch.layer2_rn(self.proj_to_l2(f2))
        l3_rn = self.scratch.layer3_rn(self.proj_to_l3(f3))
        l4_rn = self.scratch.layer4_rn(self.proj_to_l4(f4))

        p4 = self.scratch.refinenet4(l4_rn)
        p3 = self.scratch.refinenet3(p4, l3_rn)
        p2 = self.scratch.refinenet2(p3, l2_rn)
        p1 = self.scratch.refinenet1(p2, l1_rn)

        out = self.scratch.output_conv(p1)

        # # If you want to strictly remove padding from HF path, you can crop here:
        # if self.use_hf_dino and (pad[1] or pad[3]):
        #     out = out[..., :H_img, :W_img]

        return out


# # ---------------- Example usage ----------------
if __name__ == "__main__":
    # === Hugging Face path (recommended) ===
    # Make sure you’ve requested/accepted access on the model page and set a token
    # via the HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) environment variable.
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    model = OpenVAMSaliencyNet(
        use_hf=True,
        hf_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        hf_token=HF_TOKEN,
        backbone="vitb_rn50_384",
        features=256,
        readout="project",
        upsample_output_to_input_res=True,
        out_channels=1,
        freeze_dino=False,
        freeze_vit=False,  # unused but kept for compatibility
    ).eval()

    x = torch.randn(2, 3, 256, 256)
    
    # =========================================================================
    # PRINT LAYER SHAPES IN ORDER (Encoder -> Decoder flow)
    # =========================================================================
    print("\n" + "="*80)
    print("LAYER SHAPES IN ORDER (Encoder -> Decoder)")
    print("="*80)
    
    with torch.no_grad():
        B, _, H_img, W_img = x.shape
        print(f"\n{'[INPUT]':=^80}")
        print(f"  Input image:                    {list(x.shape)}")
        
        # ===== ENCODER: DINO Backbone =====
        print(f"\n{'[DINO ENCODER]':=^80}")
        
        # Normalization
        if model.hf_norm_from_processor:
            x_norm = (x - model.hf_mean) / model.hf_std
        else:
            x_norm = (x - model.dino_mean) / model.dino_std
        print(f"  After normalization:            {list(x_norm.shape)}")
        
        # Padding
        H_eff = math.ceil(H_img / model.patch_size) * model.patch_size
        W_eff = math.ceil(W_img / model.patch_size) * model.patch_size
        pad = (0, W_eff - W_img, 0, H_eff - H_img)
        x_pad = F.pad(x_norm, pad, mode="reflect") if (pad[1] or pad[3]) else x_norm
        print(f"  After padding:                  {list(x_pad.shape)}")
        
        # DINO forward
        out = model.dino(pixel_values=x_pad, output_hidden_states=True)
        print(f"  DINO last_hidden_state:         {list(out.last_hidden_state.shape)}")
        print(f"  DINO hidden_states count:       {len(out.hidden_states)} layers")
        
        # Print ALL DINO layer shapes
        print(f"\n{'[ALL DINO LAYERS]':=^80}")
        for i, hidden_state in enumerate(out.hidden_states):
            marker = " <-- EXTRACTED" if i in model.dino_layers else ""
            if i == 0:
                print(f"  Layer {i:2d} (embeddings):          {list(hidden_state.shape)}{marker}")
            else:
                print(f"  Layer {i:2d} (transformer block):   {list(hidden_state.shape)}{marker}")
        
        # Print extracted layer indices
        print(f"\n  Extracted DINO layers:          {model.dino_layers}")
        
        # ===== DINO -> Feature Maps =====
        print(f"\n{'[DINO FEATURES -> FEATURE MAPS]':=^80}")
        feats = []
        hidden = out.hidden_states
        for i, idx in enumerate(model.dino_layers):
            idx = min(max(idx, 1), len(hidden) - 1)
            tokens_l = hidden[idx]
            print(f"  Layer {idx} tokens:                {list(tokens_l.shape)}")
            m = model._dino_tokens_to_map(tokens_l, H_eff, W_eff)
            print(f"  Layer {idx} -> feature map:        {list(m.shape)}")
            feats.append(m)
        
        # ===== MULTI-SCALE PYRAMID =====
        print(f"\n{'[MULTI-SCALE PYRAMID]':=^80}")
        target_sizes = [
            model._safe_div_size(H_img, W_img, 4),
            model._safe_div_size(H_img, W_img, 8),
            model._safe_div_size(H_img, W_img, 16),
            model._safe_div_size(H_img, W_img, 32),
        ]
        f1 = F.interpolate(feats[0], size=target_sizes[0], mode="bilinear", align_corners=False)
        f2 = F.interpolate(feats[1], size=target_sizes[1], mode="bilinear", align_corners=False)
        f3 = F.interpolate(feats[2], size=target_sizes[2], mode="bilinear", align_corners=False)
        f4 = F.interpolate(feats[3], size=target_sizes[3], mode="bilinear", align_corners=False)
        print(f"  f1 (scale /4):                  {list(f1.shape)}")
        print(f"  f2 (scale /8):                  {list(f2.shape)}")
        print(f"  f3 (scale /16):                 {list(f3.shape)}")
        print(f"  f4 (scale /32):                 {list(f4.shape)}")
        
        # ===== PROJECTION LAYERS =====
        print(f"\n{'[PROJECTION LAYERS]':=^80}")
        proj1 = model.proj_to_l1(f1)
        proj2 = model.proj_to_l2(f2)
        proj3 = model.proj_to_l3(f3)
        proj4 = model.proj_to_l4(f4)
        print(f"  proj_to_l1(f1):                 {list(proj1.shape)}")
        print(f"  proj_to_l2(f2):                 {list(proj2.shape)}")
        print(f"  proj_to_l3(f3):                 {list(proj3.shape)}")
        print(f"  proj_to_l4(f4):                 {list(proj4.shape)}")
        
        # ===== DPT SCRATCH LAYERS =====
        print(f"\n{'[DPT SCRATCH LAYERS]':=^80}")
        l1_rn = model.scratch.layer1_rn(proj1)
        l2_rn = model.scratch.layer2_rn(proj2)
        l3_rn = model.scratch.layer3_rn(proj3)
        l4_rn = model.scratch.layer4_rn(proj4)
        print(f"  layer1_rn:                      {list(l1_rn.shape)}")
        print(f"  layer2_rn:                      {list(l2_rn.shape)}")
        print(f"  layer3_rn:                      {list(l3_rn.shape)}")
        print(f"  layer4_rn:                      {list(l4_rn.shape)}")
        
        # ===== DECODER: REFINENET =====
        print(f"\n{'[DECODER: REFINENET]':=^80}")
        p4 = model.scratch.refinenet4(l4_rn)
        print(f"  refinenet4(l4_rn):              {list(p4.shape)}")
        p3 = model.scratch.refinenet3(p4, l3_rn)
        print(f"  refinenet3(p4, l3_rn):          {list(p3.shape)}")
        p2 = model.scratch.refinenet2(p3, l2_rn)
        print(f"  refinenet2(p3, l2_rn):          {list(p2.shape)}")
        p1 = model.scratch.refinenet1(p2, l1_rn)
        print(f"  refinenet1(p2, l1_rn):          {list(p1.shape)}")
        
        # ===== OUTPUT CONV + HEAD =====
        # Note: In DPT, the head is integrated into scratch.output_conv
        print(f"\n{'[OUTPUT CONV + HEAD]':=^80}")
        print("  (Head layers are integrated into output_conv)")
        
        # Print the structure of output_conv to show all head layers
        print("\n  output_conv structure:")
        for i, layer in enumerate(model.scratch.output_conv):
            print(f"    [{i}] {layer}")
        
        y = model.scratch.output_conv(p1)
        print(f"\n  output_conv(p1):                {list(y.shape)}")
        
        print(f"\n{'[FINAL OUTPUT]':=^80}")
        print(f"  Output:                         {list(y.shape)}")
    
    print("\n" + "="*80)
    print("PARAMETER SUMMARY")
    print("="*80)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")