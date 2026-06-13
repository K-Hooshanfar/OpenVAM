#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-VL ViT + DPT Decoder with TEXT CONDITIONING (No DINOv3).

Architecture:
    Image → Qwen VL ViT ─┬─→ Intermediate layers [L¼, L½, L¾] ─→ DPT Decoder ─→ Dense Output
                          │                                              ↑
                          └─→ Built-in PatchMerger ─┐                   │
                                                     ├─→ Qwen LLM ─→ Visual Tokens (text-conditioned)
    Text → Tokenize → Embed ─────────────────────────┘

Key difference from OpenVAM:
- No DINOv3 backbone; the Qwen VL visual encoder (ViT) is used for ALL image features.
- Intermediate ViT block outputs feed DPT layers 1–3.
- Final ViT tokens (after built-in PatchMerger) feed the Qwen LLM → DPT layer 4.
- Compatible with train_salience_and_text_lora.py via --model_type qwen_vit.
"""

import os
import math
import contextlib
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from DPT.dpt.models import DPT

try:
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    _QWEN25_VL_AVAILABLE = True
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
    _QWEN25_VL_AVAILABLE = False

try:
    from transformers import Qwen3VLForConditionalGeneration
    _QWEN3_VL_AVAILABLE = True
except ImportError:
    Qwen3VLForConditionalGeneration = None
    _QWEN3_VL_AVAILABLE = False

_HAS_TRANSFORMERS = _QWEN25_VL_AVAILABLE or _QWEN3_VL_AVAILABLE

# Reuse prompt definitions and helpers from the existing module
from net.openvam import (
    DATASET_TYPE_PROMPTS,
    get_dataset_type_from_name,
    _get_qwen_vl_class,
    Qwen2RMSNorm,
    Qwen2_5_VLPatchMerger,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class QwenViTDPTWithText(DPT):
    """
    Saliency model that uses Qwen VL's built-in ViT as the sole visual backbone.
    No DINOv3. DPT decoder is attached directly to the ViT intermediate layers.

    Layers 1–3 of the DPT decoder receive intermediate ViT features tapped at
    block indices [n//4, n//2, 3n//4].  Layer 4 receives ViT final tokens passed
    through the Qwen LLM (text-conditioned).
    """

    def __init__(
        self,
        qwen_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        hf_token: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        # DPT decoder
        backbone: str = "vitb_rn50_384",
        features: int = 256,
        readout: str = "project",
        channels_last: bool = False,
        use_bn: bool = False,
        enable_attention_hooks: bool = False,
        upsample_output_to_input_res: bool = True,
        out_channels: int = 1,
        use_sigmoid: bool = True,
        # Text conditioning
        max_text_length: int = 1024,
        # Freeze flags
        freeze_qwen_vit: bool = False,
        freeze_qwen_lm: bool = True,
        freeze_projector: bool = False,
    ):
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

        super().__init__(
            head=head,
            features=features,
            backbone=backbone,
            readout=readout,
            channels_last=channels_last,
            use_bn=use_bn,
            enable_attention_hooks=enable_attention_hooks,
        )
        if hasattr(self, "pretrained"):
            del self.pretrained

        assert _HAS_TRANSFORMERS, "Install transformers>=4.45.0 with Qwen2.5-VL support"

        self.token = hf_token or os.getenv("HF_TOKEN")
        self.dtype = dtype or torch.bfloat16
        self.max_text_length = max_text_length

        # ------------------------------------------------------------------
        # 1. Load Qwen VL model
        # ------------------------------------------------------------------
        QwenVLClass = _get_qwen_vl_class(qwen_model_name)
        print(f"[1/3] Loading Qwen-VL ({QwenVLClass.__name__}): {qwen_model_name}")

        qwen_full = QwenVLClass.from_pretrained(
            qwen_model_name,
            torch_dtype=self.dtype,
            token=self.token,
            trust_remote_code=True,
        )

        # -- Dimension metadata --
        self.qwen_dim = (
            getattr(qwen_full.config, "hidden_size", None)
            or getattr(qwen_full.config.text_config, "hidden_size", 2048)
        )
        self.vocab_size = (
            getattr(qwen_full.config, "vocab_size", None)
            or getattr(qwen_full.config.text_config, "vocab_size", 151936)
        )
        vc = getattr(qwen_full.config, "vision_config", None)
        self.spatial_merge = getattr(vc, "spatial_merge_size", getattr(vc, "merge_size", 2)) if vc else 2
        self.qwen_vision_dim = getattr(vc, "hidden_size", getattr(vc, "embed_dim", 1280)) if vc else 1280

        # -- Extract ViT visual encoder --
        vl_model = qwen_full.model
        if not hasattr(vl_model, "visual"):
            raise AttributeError("Qwen-VL model.visual not found; unsupported model variant.")
        self.qwen_visual = vl_model.visual  # keep the whole ViT

        # Patch / temporal sizes from the visual encoder config
        self.qwen_patch_size = getattr(self.qwen_visual.config, "patch_size", 14)
        self.qwen_temporal_patch_size = getattr(self.qwen_visual.config, "temporal_patch_size", 2)

        if freeze_qwen_vit:
            self.qwen_visual.eval()
            for p in self.qwen_visual.parameters():
                p.requires_grad = False

        print(f"      ViT dim={self.qwen_vision_dim}, patch_size={self.qwen_patch_size}, "
              f"temporal_patch_size={self.qwen_temporal_patch_size}")

        # -- Extract Qwen LLM backbone --
        if not hasattr(vl_model, "language_model"):
            raise AttributeError("Qwen-VL model.language_model not found.")
        lm = vl_model.language_model
        lm_inner = lm.model if hasattr(lm, "model") else lm
        self.qwen_backbone = lm_inner
        self.embed_tokens = lm_inner.embed_tokens
        self.qwen_layers = lm_inner.layers
        self.qwen_norm = lm_inner.norm

        if freeze_qwen_lm:
            for p in self.embed_tokens.parameters():
                p.requires_grad = False
            for p in self.qwen_layers.parameters():
                p.requires_grad = False
            for p in self.qwen_norm.parameters():
                p.requires_grad = False

        # Keep full model for text generation
        self.qwen_full_model = qwen_full

        print(f"      LLM dim={self.qwen_dim}, {len(self.qwen_layers)} layers")

        # ------------------------------------------------------------------
        # 2. PatchMerger: ViT final tokens → Qwen LLM dimension
        #    (the ViT's built-in merger projects to lm_dim; we use a separate
        #     one here so the merged token count / grid dims are explicit)
        # ------------------------------------------------------------------
        print(f"[2/3] Building PatchMerger (vision_dim={self.qwen_vision_dim} → lm_dim={self.qwen_dim})")
        self.patch_merger = Qwen2_5_VLPatchMerger(
            dim=self.qwen_dim,
            context_dim=self.qwen_vision_dim,
            spatial_merge_size=self.spatial_merge,
        )
        if freeze_projector:
            for p in self.patch_merger.parameters():
                p.requires_grad = False

        # ------------------------------------------------------------------
        # 3. DPT projections (ViT intermediate features → DPT scratch layers)
        # ------------------------------------------------------------------
        print(f"[3/3] Building DPT projections")
        in_ch_l1 = self.scratch.layer1_rn.in_channels
        in_ch_l2 = self.scratch.layer2_rn.in_channels
        in_ch_l3 = self.scratch.layer3_rn.in_channels
        in_ch_l4 = self.scratch.layer4_rn.in_channels
        print(f"      Scratch channels: L1={in_ch_l1}, L2={in_ch_l2}, L3={in_ch_l3}, L4={in_ch_l4}")

        # LazyConv2d: will infer input channels from qwen_vision_dim on first forward
        self.proj_to_l1 = nn.LazyConv2d(in_ch_l1, kernel_size=1)
        self.proj_to_l2 = nn.LazyConv2d(in_ch_l2, kernel_size=1)
        self.proj_to_l3 = nn.LazyConv2d(in_ch_l3, kernel_size=1)
        self.proj_to_l4 = nn.Conv2d(self.qwen_dim, in_ch_l4, kernel_size=1)

        self.patch_merger = self.patch_merger.to(self.dtype)
        self.proj_to_l1 = self.proj_to_l1.to(self.dtype)
        self.proj_to_l2 = self.proj_to_l2.to(self.dtype)
        self.proj_to_l3 = self.proj_to_l3.to(self.dtype)
        self.proj_to_l4 = self.proj_to_l4.to(self.dtype)
        self.scratch = self.scratch.to(self.dtype)

        # Image normalization (CLIP-style used by Qwen VL)
        self.register_buffer(
            "img_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

        # Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                qwen_model_name, token=self.token, local_files_only=True
            )
        except (OSError, ValueError):
            self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, token=self.token)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.default_prompt = DATASET_TYPE_PROMPTS["natural_scene"]["user"]

        print(f"\nQwenViTDPTWithText ready (no DINOv3, ViT-only backbone)")

    # ------------------------------------------------------------------
    # rebind_lora_references  (needed by train_salience_and_text_lora.py)
    # ------------------------------------------------------------------
    def rebind_lora_references(self):
        vl_model = self.qwen_full_model.model
        if not hasattr(vl_model, "language_model"):
            return
        lm = vl_model.language_model
        try:
            from peft import PeftModel
            is_peft = isinstance(lm, PeftModel)
        except ImportError:
            is_peft = False

        if is_peft:
            lm_inner = lm.model if hasattr(lm, "model") else lm
            self.qwen_backbone = lm_inner
            self.embed_tokens = lm_inner.embed_tokens
            self.qwen_layers = lm_inner.layers
            self.qwen_norm = lm_inner.norm
            print("  LoRA references rebound (PeftModel detected)")
        else:
            lm_inner = lm.model if hasattr(lm, "model") else lm
            self.qwen_backbone = lm_inner
            self.embed_tokens = lm_inner.embed_tokens
            self.qwen_layers = lm_inner.layers
            self.qwen_norm = lm_inner.norm

    # ------------------------------------------------------------------
    # Patchify: images → Qwen VL ViT input format
    # ------------------------------------------------------------------
    def _patchify_for_qwen_vit(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Convert (B, 3, H, W) images in [0,1] to Qwen VL ViT input.

        Returns:
            pixel_values : (B * grid_h * grid_w, C * tps * ps * ps)
            grid_thw     : (B, 3)  — [temporal=1, grid_h, grid_w] per image
            grid_h, grid_w
        """
        B, C, H, W = images.shape
        ps = self.qwen_patch_size
        tps = self.qwen_temporal_patch_size
        device = images.device

        mean = self.img_mean.to(device=device, dtype=images.dtype)
        std = self.img_std.to(device=device, dtype=images.dtype)
        x = (images - mean) / std

        H_pad = math.ceil(H / ps) * ps
        W_pad = math.ceil(W / ps) * ps
        if H != H_pad or W != W_pad:
            x = F.pad(x, (0, W_pad - W, 0, H_pad - H), mode="reflect")

        grid_h = H_pad // ps
        grid_w = W_pad // ps
        n_patches = grid_h * grid_w

        # Duplicate along temporal axis: (B, C, H_pad, W_pad) → (B, C*tps, H_pad, W_pad)
        x_t = x.unsqueeze(2).expand(-1, -1, tps, -1, -1).contiguous()
        x_t = x_t.view(B, C * tps, H_pad, W_pad)

        # Spatial unfold → (B, C*tps, grid_h, grid_w, ps, ps)
        x_t = x_t.unfold(2, ps, ps).unfold(3, ps, ps)
        # → (B, grid_h, grid_w, C*tps, ps, ps)
        x_t = x_t.permute(0, 2, 3, 1, 4, 5).contiguous()
        # → (B * n_patches, C * tps * ps * ps)
        pixel_values = x_t.view(B * n_patches, C * tps * ps * ps)

        grid_thw = torch.tensor(
            [[1, grid_h, grid_w]], dtype=torch.long, device=device
        ).expand(B, 3).contiguous()

        return pixel_values, grid_thw, grid_h, grid_w

    # ------------------------------------------------------------------
    # Encode image with Qwen VL ViT + intermediate feature extraction
    # ------------------------------------------------------------------
    def encode_vit(
        self, pixel_values: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, int, int]:
        """
        Run Qwen VL ViT on a batch of images and collect:
          - 3 intermediate feature maps (for DPT layers 1–3)
          - final patch tokens before merger (for text conditioning → DPT layer 4)

        Returns:
            vit_features : list of 3 tensors, each (B, vit_dim, grid_h, grid_w)
            patch_tokens : (B, n_patches, vit_dim) — final ViT tokens before merger
            grid_h, grid_w
        """
        B = pixel_values.shape[0]
        pv_flat, grid_thw, grid_h, grid_w = self._patchify_for_qwen_vit(pixel_values)
        n_patches = grid_h * grid_w
        vis = self.qwen_visual

        # Cast input to ViT dtype
        pv_flat = pv_flat.to(dtype=self.dtype)

        # Patch embedding
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in vis.parameters())):
            hidden = vis.patch_embed(pv_flat)  # (B * n_patches, vit_dim)

            # Positional embeddings & cu_seqlens for window/full attention
            rotary_pos_emb = vis.rot_pos_emb(grid_thw)
            seqlens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2])  # (B,)
            cu_seqlens = seqlens.cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

            n_blocks = len(vis.blocks)
            # Hook points at ¼, ½, ¾ depth
            hook_at = {n_blocks // 4, n_blocks // 2, 3 * n_blocks // 4}
            vit_feats_flat = []

            for i, block in enumerate(vis.blocks):
                hidden = block(hidden, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
                if (i + 1) in hook_at:
                    vit_feats_flat.append(hidden.clone())

        # Reshape flat features → spatial maps (B, vit_dim, grid_h, grid_w)
        vit_features = []
        for feat_flat in vit_feats_flat:
            vit_dim = feat_flat.shape[-1]
            feat = feat_flat.view(B, n_patches, vit_dim)          # (B, N, D)
            feat = feat.permute(0, 2, 1).contiguous().view(B, vit_dim, grid_h, grid_w)
            vit_features.append(feat)

        # Final patch tokens (before merger) for LLM conditioning
        patch_tokens = hidden.view(B, n_patches, -1)  # (B, N, vit_dim)

        return vit_features, patch_tokens, grid_h, grid_w

    # ------------------------------------------------------------------
    # Helpers reused from OpenVAM
    # (inline here so the class is self-contained)
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_div(H: int, W: int, div: int) -> Tuple[int, int]:
        return (max(1, H // div), max(1, W // div))

    def _tokens_to_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        B, N, D = tokens.shape
        return tokens.permute(0, 2, 1).contiguous().view(B, D, h, w)

    # ------------------------------------------------------------------
    # Text encoding (identical interface to existing model)
    # ------------------------------------------------------------------
    def encode_text(self, text_prompt, B, device, dataset_type=None):
        """Tokenise text_prompt and embed it through Qwen embed_tokens."""
        if dataset_type and dataset_type in DATASET_TYPE_PROMPTS:
            prompts = DATASET_TYPE_PROMPTS[dataset_type]
            messages = [
                {"role": "system", "content": prompts["system"]},
                {"role": "user",   "content": prompts["user"]},
            ]
        else:
            messages = [{"role": "user", "content": text_prompt or self.default_prompt}]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(
            [text] * B,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        ).to(device)

        with torch.no_grad():
            text_embeds = self.embed_tokens(enc.input_ids).to(dtype=self.dtype)
        return text_embeds, enc.attention_mask, enc.input_ids.shape[1]

    # ------------------------------------------------------------------
    # Fuse text + visual through Qwen LLM
    # ------------------------------------------------------------------
    def _build_mrope_position_ids(self, num_visual_tokens, num_text_tokens,
                                   grid_h, grid_w, device, vision_first=True):
        """Simplified M-RoPE: sequential IDs with vision/text offset."""
        num_vis = grid_h * grid_w
        vis_t = torch.zeros(num_vis, device=device, dtype=torch.long)
        vis_h = torch.arange(grid_h, device=device).repeat_interleave(grid_w)
        vis_w = torch.arange(grid_w, device=device).repeat(grid_h)
        offset = max(grid_h, grid_w) + 1
        txt_ids = torch.arange(num_text_tokens, device=device) + offset

        if vision_first:
            vis_pos = torch.stack([vis_t, vis_h, vis_w], dim=0)              # [3, N_vis]
            txt_pos = txt_ids.unsqueeze(0).expand(3, -1)                     # [3, N_txt]
            pos = torch.cat([vis_pos, txt_pos], dim=1)                       # [3, N_total]
        else:
            txt_pos = txt_ids.unsqueeze(0).expand(3, -1)
            vis_pos = torch.stack([vis_t + offset, vis_h + offset, vis_w + offset], dim=0)
            pos = torch.cat([txt_pos, vis_pos], dim=1)
        return pos.unsqueeze(1)  # [3, 1, N_total]

    def fuse_text_and_vision(self, visual_tokens, text_embeds, text_mask,
                              grid_h, grid_w, order="vision_first"):
        B = visual_tokens.shape[0]
        device = visual_tokens.device
        N_vis = visual_tokens.shape[1]
        N_txt = text_embeds.shape[1]

        if order == "vision_first":
            combined = torch.cat([visual_tokens, text_embeds], dim=1)
            attn_mask = torch.cat(
                [torch.ones(B, N_vis, device=device, dtype=torch.long), text_mask], dim=1
            )
        else:
            combined = torch.cat([text_embeds, visual_tokens], dim=1)
            attn_mask = torch.cat(
                [text_mask, torch.ones(B, N_vis, device=device, dtype=torch.long)], dim=1
            )

        pos_ids = self._build_mrope_position_ids(
            N_vis, N_txt, grid_h, grid_w, device, vision_first=(order == "vision_first")
        ).expand(3, B, -1)

        outputs = self.qwen_backbone(
            inputs_embeds=combined,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            return_dict=True,
            use_cache=False,
        )
        hs = outputs.last_hidden_state
        if order == "vision_first":
            return hs[:, :N_vis, :]
        return hs[:, N_txt:, :]

    # ------------------------------------------------------------------
    # Text generation (for qualitative checks in training loop)
    # ------------------------------------------------------------------
    def generate_text(self, pixel_values, text_prompt, max_new_tokens=100,
                      do_sample=False, **kwargs):
        """Thin wrapper that keeps the same API as OpenVAM."""
        from net.openvam import DATASET_TYPE_PROMPTS as _P
        messages = [{"role": "user", "content": text_prompt or self.default_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(text, return_tensors="pt").to(pixel_values.device)
        with torch.no_grad():
            out = self.qwen_full_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **kwargs,
            )
        new_ids = out[0][enc.input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def get_text_generation_logits(self, pixel_values, input_ids, attention_mask=None):
        """Return LM logits for teacher-forced training (text loss)."""
        B = pixel_values.shape[0]
        device = pixel_values.device
        _, patch_tokens, grid_h, grid_w = self.encode_vit(pixel_values)
        visual_tokens, merged_h, merged_w = self.patch_merger(patch_tokens, grid_h, grid_w)
        text_embeds = self.embed_tokens(input_ids).to(dtype=self.dtype)
        if attention_mask is None:
            attention_mask = torch.ones(B, input_ids.shape[1], device=device, dtype=torch.long)
        full_seq = self.fuse_text_and_vision(
            visual_tokens, text_embeds, attention_mask,
            grid_h=merged_h, grid_w=merged_w,
            order="vision_first",
        )
        # Apply full sequence: vision + text
        N_vis = visual_tokens.shape[1]
        # Re-run to get text portion logits
        combined = torch.cat([visual_tokens, text_embeds], dim=1)
        attn_mask = torch.cat(
            [torch.ones(B, N_vis, device=device, dtype=torch.long), attention_mask], dim=1
        )
        pos_ids = self._build_mrope_position_ids(
            N_vis, input_ids.shape[1], merged_h, merged_w, device, vision_first=True
        ).expand(3, B, -1)
        outputs = self.qwen_backbone(
            inputs_embeds=combined,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            return_dict=True,
            use_cache=False,
        )
        hs = outputs.last_hidden_state[:, N_vis:, :]  # text positions only
        return self._apply_lm_head(hs)

    def _apply_lm_head(self, hidden_states):
        head_dtype = next(self.qwen_norm.parameters()).dtype
        hidden_states = hidden_states.to(dtype=head_dtype)
        normalized = self.qwen_norm(hidden_states)
        if hasattr(self.qwen_full_model, "lm_head"):
            return self.qwen_full_model.lm_head(normalized)
        lm = self.qwen_full_model.model.language_model
        if hasattr(lm, "lm_head"):
            return lm.lm_head(normalized)
        raise AttributeError("Could not find lm_head in Qwen model.")

    # ------------------------------------------------------------------
    # Main forward  (same signature as OpenVAM)
    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        text_prompt: Optional[str] = None,
        dataset_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values : (B, 3, H, W) in [0, 1]
            text_prompt  : optional text string for conditioning
            dataset_type : one of natural_scene / webpage / e_commerce

        Returns:
            saliency map (B, 1, H, W)
        """
        B, _, H_img, W_img = pixel_values.shape
        device = pixel_values.device

        # 1. Encode text
        text_embeds, text_mask, _ = self.encode_text(
            text_prompt or self.default_prompt, B, device, dataset_type=dataset_type
        )

        # 2. Encode with Qwen VL ViT (intermediate features + final tokens)
        vit_features, patch_tokens, grid_h, grid_w = self.encode_vit(pixel_values)

        # 3. Project final ViT tokens through PatchMerger → Qwen LM dimension
        visual_tokens, merged_h, merged_w = self.patch_merger(patch_tokens, grid_h, grid_w)

        # 4. Fuse text + vision through Qwen LLM (vision first)
        fused_visual = self.fuse_text_and_vision(
            visual_tokens, text_embeds, text_mask,
            grid_h=merged_h, grid_w=merged_w,
            order="vision_first",
        )

        # 5. Reshape fused tokens → feature map for DPT layer 4
        qwen_map = self._tokens_to_map(fused_visual, merged_h, merged_w)

        # 6. Build multi-scale pyramid
        target_sizes = [
            self._safe_div(H_img, W_img, 4),
            self._safe_div(H_img, W_img, 8),
            self._safe_div(H_img, W_img, 16),
            self._safe_div(H_img, W_img, 32),
        ]

        # DPT layers 1–3: ViT intermediate features
        f1 = F.interpolate(self.proj_to_l1(vit_features[0]), size=target_sizes[0], mode="bilinear", align_corners=False)
        f2 = F.interpolate(self.proj_to_l2(vit_features[1]), size=target_sizes[1], mode="bilinear", align_corners=False)
        f3 = F.interpolate(self.proj_to_l3(vit_features[2]), size=target_sizes[2], mode="bilinear", align_corners=False)
        # DPT layer 4: Qwen LLM-conditioned features
        f4 = F.interpolate(self.proj_to_l4(qwen_map), size=target_sizes[3], mode="bilinear", align_corners=False)

        # Cast to scratch dtype
        scratch_dtype = next(self.scratch.layer1_rn.parameters()).dtype
        f1 = f1.to(dtype=scratch_dtype)
        f2 = f2.to(dtype=scratch_dtype)
        f3 = f3.to(dtype=scratch_dtype)
        f4 = f4.to(dtype=scratch_dtype)

        # 7. DPT decoder
        l1_rn = self.scratch.layer1_rn(f1)
        l2_rn = self.scratch.layer2_rn(f2)
        l3_rn = self.scratch.layer3_rn(f3)
        l4_rn = self.scratch.layer4_rn(f4)

        p4 = self.scratch.refinenet4(l4_rn)
        p3 = self.scratch.refinenet3(p4, l3_rn)
        p2 = self.scratch.refinenet2(p3, l2_rn)
        p1 = self.scratch.refinenet1(p2, l1_rn)

        return self.scratch.output_conv(p1)
