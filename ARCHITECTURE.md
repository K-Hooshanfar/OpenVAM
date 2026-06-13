# Model Architecture: OpenVAM

**OpenVAM** (Open-World Visual Attention Modeling with VLMs) jointly predicts *where* people look — a dense saliency map \(\hat{S}\in\mathbb{R}^{H\times W}\) — and *what/why* it is salient — a grounded explanation sequence \(\hat{Y}=\{y_t\}_{t=1}^{T}\), for an input image \(I\in\mathbb{R}^{H\times W\times 3}\).

It follows a **decoupled-but-aligned** design that separates *where* attention is (dense saliency) from *what/why* it is (text), while forcing both heads to condition on the same image and a data-type prompt:

- a dedicated **dense visual pathway** — DINOv3 **Visual Encoder** + coarse-to-fine **Saliency Decoder** — gives stable, spatially precise localization that is never entangled with language dynamics;
- an instruction-following **Vision-Language Semantic Head** — Qwen-VL + a lightweight visual adapter — generates grounded explanations conditioned on a data-type instruction and improves cross-domain robustness.

| Paper term | Code symbol | Location |
|------------|-------------|----------|
| OpenVAM (full model) | `OpenVAM` | `net/openvam.py` |
| Stage-I saliency-only network | `OpenVAMSaliencyNet` | `net/saliency_net.py` |
| Visual Encoder | DINOv3 ViT-B/16 backbone | `net/openvam.py` |
| ConvUpConv projection / feature pyramid | `proj_to_l1` … `proj_to_l4` | `net/openvam.py` |
| Saliency Decoder (RefineNet, DPT-style) | `layer1_rn`…`layer4_rn`, `refinenet1`…`refinenet4`, `output_conv` | `net/openvam.py`, `DPT/` |
| Visual adapter | `VisualAdapter` (+ `Qwen2_5_VLPatchMerger`) | `net/openvam.py` |
| Vision-Language Semantic Head | Qwen-VL transformer + `lm_head` | `net/openvam.py` |
| Qwen-native-ViT variant | `QwenViTDPTWithText` | `net/openvam_qwen_vit.py` |

---

## High-Level Overview

```
                          ┌─────────────────────────────────────────────────────────┐
   Image [B,3,256,256]    │                    SALIENCY PATH                        │
          │               │                                                         │
          ▼               │                                                         │
  ┌───────────────┐       │                                                         │
  │   DINOv3      │       │                                                         │
  │  (ViT-B/16)   │───────┤                                                         │
  │               │       │                                                         │
  └───────┬───────┘       │         ┌──────────────────────────────────┐             │
          │               │         │     Saliency Decoder             │             │
          │               │         │                                  │             │
          ├── L3 feats ──►├────────►│  proj_to_l1 → layer1_rn ──────┐ │             │
          ├── L6 feats ──►├────────►│  proj_to_l2 → layer2_rn ───┐  │ │             │
          ├── L9 feats ──►├────────►│  proj_to_l3 → layer3_rn ─┐ │  │ │             │
          │               │         │                           │ │  │ │             │
          │               │         │                           ▼ ▼  ▼ │             │
          │               │    ┌───►│  proj_to_l4 → layer4_rn   │ │  │ │             │
          │               │    │    │       │                   │ │  │ │             │
          │               │    │    │       ▼                   │ │  │ │             │
          │               │    │    │  refinenet4               │ │  │ │             │
          │               │    │    │       │                   │ │  │ │             │
          │               │    │    │       ▼                   │ │  │ │             │
          │               │    │    │  refinenet3 ◄─────────────┘ │  │ │             │
          │               │    │    │       │                     │  │ │             │
          │               │    │    │       ▼                     │  │ │             │
          │               │    │    │  refinenet2 ◄───────────────┘  │ │             │
          │               │    │    │       │                        │ │             │
          │               │    │    │       ▼                        │ │             │
          │               │    │    │  refinenet1 ◄─────────────────┘ │             │
          │               │    │    │       │                         │             │
          │               │    │    │       ▼                         │             │
          │               │    │    │  output_conv                    │  Saliency   │
          │               │    │    └───────┼─────────────────────────┘  Map        │
          │               │    │            ▼                              [B,1,H,W]│
          │               │    │    Saliency Map Output ──────────────────────────►─┘
          │               │    │
          ▼               │    │
  ┌───────────────┐       │    │
  │ Last hidden   │       │    │
  │ state (L12)   │       │    │       ┌─────────────────────────────────────────┐
  │ patch tokens  │       │    │       │           CROSS-MODAL FUSION            │
  │ [B, 256, 768] │       │    │       │                                         │
  └───────┬───────┘       │    │       │                                         │
          │               │    │       │                                         │
          ▼               │    │       │                                         │
  ┌───────────────┐       │    │       │                                         │
  │ VisualAdapter │       │    │       │                                         │
  │ Linear        │       │    │       │                                         │
  │ 768 → 1536    │       │    │       │                                         │
  └───────┬───────┘       │    │       │                                         │
          │               │    │       │                                         │
          ▼               │    │       │                                         │
  ┌───────────────┐       │    │       │                                         │
  │ PatchMerger   │       │    │       │                                         │
  │ (2×2 spatial  │       │    │       │                                         │
  │  merge + MLP) │       │    │       │                                         │
  │ → [B,64,1536] │       │    │       │                                         │
  └───────┬───────┘       │    │       │                                         │
          │               │    │       │                                         │
          │  Visual       │    │       │                                         │
          │  Tokens       │    │       │                                         │
          │               │    │       │                                         │
          ▼               │    │       │                                         │
  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─│─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐│
  │  Concat: [Visual Tokens, Text Embeddings]                                   ││
  │          [B, N_vis + N_text, 1536]                                          ││
  │                        │    │       │                                        ││
  │          ┌─────────────┘    │       │                                        ││
             │                  │       │                                        │
  │          ▼                  │       │                                        ││
     ┌───────────────┐         │       │                                        │
  │  │  Qwen LLM     │         │       │                                        ││
     │  Transformer   │         │       │                                        │
  │  │  Layers (×28)  │         │       │                                        ││
     │  + M-RoPE      │         │       │                                        │
  │  │  positions     │         │       │                                        ││
     └───────┬────────┘         │       │                                        │
  │          │                  │       │                                        ││
             │ Full sequence    │       │                                        │
  │          │ hidden states    │       │                                        ││
             │                  │       │                                        │
  │          ├──► Visual part ──┼───────┘  Text-conditioned visual features      ││
             │    [B,N_vis,dim] │           reshaped → [B, dim, h, w]            │
  │          │                  │           fed into Saliency Decoder as L4 ─────┘│
             │                  │
  │          └──► Text part ────┼──► qwen_norm → lm_head → Text Logits           │
                  [B,N_txt,dim] │    (grounded explanation, Stage III)
  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

Both heads operate on the same image and are optimized jointly. The Saliency Decoder consumes raw DINOv3 features for levels 1–3 and the **VLM-fused L4** feature, so dense localization stays spatially precise while still being conditioned on text.

---

## Detailed Component Breakdown

### 1. Visual Encoder — DINOv3 (ViT-B/16)

The dense pathway is decoupled from the VLM's native vision tower. We tap DINOv3 hidden states at layers \(\ell\in\{3,6,9,12\}\): earlier blocks preserve fine spatial detail and local contrast (critical for fixation localization), while deeper blocks encode higher-level semantics and context.

```
Input Image [B, 3, 256, 256]
      │
      ▼
 Normalize (ImageNet mean/std)
      │
      ▼
 Patch Embedding (16×16 patches → 256 tokens)
      │
      ▼
 12 Transformer Blocks
      │
      ├── Block  3 output → dino_features[0]  [B, 768, 16, 16]  (→ decoder L1, fine detail)
      ├── Block  6 output → dino_features[1]  [B, 768, 16, 16]  (→ decoder L2)
      ├── Block  9 output → dino_features[2]  [B, 768, 16, 16]  (→ decoder L3)
      └── Block 12 output → patch_tokens       [B, 256, 768]     (→ Visual Adapter / VLM fusion)
```

Layers \(L_3,L_6,L_9\) form a compact hierarchy for the Saliency Decoder; \(L_{12}\) — the most semantically abstract representation — is reserved for cross-modal fusion so the saliency features are not entangled with language dynamics. The supplementary backbone ablation shows DINOv3 ViT-B is the best accuracy/efficiency trade-off, so it is the default.

### 2. Visual Adapter (replaces Qwen's native Vision Embedding)

A lightweight adapter injects DINOv3's \(L_{12}\) patch tokens into the language context.

```
patch_tokens [B, 256, 768]
      │
      ▼
 VisualAdapter: nn.Linear(768 → 1536)       ... project to Qwen vision dim
      │
      ▼
 adapted_tokens [B, 256, 1536]
      │
      ▼
 PatchMerger (Qwen-VL style):               ... spatial 2×2 merge
   ├── RMSNorm
   ├── Reshape to 2×2 blocks
   ├── Concat 4 neighbors → [B, 64, 6144]
   └── MLP: Linear(6144, 6144) → GELU → Linear(6144, 1536)
      │
      ▼
 visual_tokens [B, 64, 1536]                ... ready for the Qwen transformer
```

### 3. Text Encoding (data-type prompt)

The explanation is conditioned on a **data-type instruction** \(\pi(\cdot)\) — `Generic` for natural scenes, `UI/Webpage` for web/UI layouts, and `E-commerce` for commercial imagery — which keeps the explanation style and grounding consistent across domains. Prompts are resolved per sample from the merged dataset id (see `get_dataset_type_from_name` / `DATASET_TYPE_PROMPTS`).

```
Data-type prompt (e.g. "List the 2-6 most visually salient objects/regions, descending...")
      │
      ▼
 Chat Template (system + user roles, ChatML format)
      │
      ▼
 Tokenizer (Qwen BPE, not trainable)
      │
      ▼
 input_ids [B, N_text]
      │
      ▼
 embed_tokens (Qwen word embedding, frozen)
      │
      ▼
 text_embeds [B, N_text, 1536]
```

### 4. Cross-Modal Fusion — Qwen-VL Transformer (Semantic Head)

```
 visual_tokens [B, 64, 1536]     text_embeds [B, N_text, 1536]
           │                              │
           └──────────┬───────────────────┘
                      ▼
              Concatenate (vision_first order)
              [B, 64 + N_text, 1536]
                      │
                      ▼
              M-RoPE Position IDs
              (3D: temporal, height, width for vision;
               1D sequential for text)
                      │
                      ▼
           ┌──────────────────────┐
           │  Qwen Transformer    │
           │  Layer 1             │  ← LoRA adapters (Stage III)
           │  ...                 │
           │  Layer 28            │  ← LoRA adapters (Stage III)
           └──────────┬──────────┘
                      │
              Full hidden states
              [B, 64 + N_text, 1536]
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
  Visual part               Text part
  [B, 64, 1536]            [B, N_text, 1536]
         │                         │
         ▼                         ▼
  reshape to spatial        qwen_norm → lm_head
  [B, 1536, 8, 8]          → Text Logits [B, N_text, vocab]
         │                  (grounded explanation)
         ▼
  Fed into the Saliency
  Decoder as L4 features
```

The VLM is **not** the primary source of dense spatial features; it functions as an auxiliary semantic head that improves interpretability and domain robustness without destabilizing saliency learning.

### 5. Saliency Decoder (coarse-to-fine, DPT-style)

Given the four-level pyramid, a RefineNet-style decoder progressively fuses features via skip connections — \(\mathbf{z}^{(4)}=r_4(\mathbf{f}^{(4)})\), \(\mathbf{z}^{(k)}=r_k(\mathbf{f}^{(k)},\uparrow(\mathbf{z}^{(k+1)}))\) for \(k=3,2,1\) — and a lightweight head produces the final map.

```
                  DINO intermediate features          Qwen fused visual map
                         │   │   │                           │
                         ▼   ▼   ▼                           ▼
Scale /4:   dino_feat[0] → proj_to_l1 (1×1 conv) → resize → f1  [B, 256, 64, 64]
Scale /8:   dino_feat[1] → proj_to_l2 (1×1 conv) → resize → f2  [B, 256, 32, 32]
Scale /16:  dino_feat[2] → proj_to_l3 (1×1 conv) → resize → f3  [B, 256, 16, 16]
Scale /32:  qwen_map     → proj_to_l4 (1×1 conv) → resize → f4  [B, 256,  8,  8]
                                                                      │
                ┌─────────────────────────────────────────────────────┘
                ▼
        layer4_rn(f4) ──────────────────────────────────► refinenet4
                                                               │
        layer3_rn(f3) ──────────────────────────────────► refinenet3
                                                               │
        layer2_rn(f2) ──────────────────────────────────► refinenet2
                                                               │
        layer1_rn(f1) ──────────────────────────────────► refinenet1
                                                               │
                                                               ▼
                                                          output_conv
                                                        ┌─────────────┐
                                                        │ Conv 256→128│
                                                        │ ReLU        │
                                                        │ Upsample ×2 │
                                                        │ Conv 128→32 │
                                                        │ ReLU        │
                                                        │ Conv 32→1   │
                                                        │ Sigmoid     │
                                                        └──────┬──────┘
                                                               │
                                                               ▼
                                                     Saliency Map [B, 1, 256, 256]
```

The paper writes the per-level projection as a `ConvUpConv` block \(g_k(\cdot)=\mathrm{Conv}_{3\times3}(\uparrow(\mathrm{Conv}_{1\times1}(\cdot)))\); in code this is the `proj_to_l*` + resize stage feeding each `layer*_rn`.

## Key Design Choices

- **Levels 1–3 come from DINOv3 blocks 3/6/9** — pure-vision features, no text influence — for spatially precise localization.
- **Level 4 comes from the Qwen transformer output** — visual features that have attended to the text tokens via causal attention — making the saliency prediction **text-conditioned**.
- **Qwen's native vision encoder is removed** and replaced by DINOv3 + the Visual Adapter, decoupling dense prediction from the VLM's vision tower.

---

## Loss Functions

OpenVAM is trained with a composite saliency objective that combines distributional and structural terms. Let \(S^{g}\) be the ground-truth saliency map, \(F^{g}\) the ground-truth fixation map, and \(\hat{S}\) the prediction:

\[
\mathcal{L}_{\text{sal}} =
\lambda_{1}\mathcal{L}_{\text{KL}}(S^{g},\hat{S})
+\lambda_{2}\mathcal{L}_{\text{CC}}(S^{g},\hat{S})
+\lambda_{3}\mathcal{L}_{\text{SIM}}(S^{g},\hat{S})
+\lambda_{4}\mathcal{L}_{\text{NSS}}(F^{g},\hat{S})
+\lambda_{5}\mathcal{L}_{\text{MSE}}(S^{g},\hat{S})
\]

Dissimilarity terms (KL, MSE) are minimized and similarity terms (CC, SIM, NSS) are maximized. All terms are implemented in `utils/losses.py`.

| Term | Direction | Role |
|------|-----------|------|
| **KL divergence** | minimize | Distributional alignment; penalizes missing mass where \(S^g\) is large (\(\epsilon=2.2\times10^{-16}\)). |
| **CC** (correlation coefficient) | maximize | Global structural agreement; insensitive to affine rescaling of \(\hat{S}\). |
| **SIM** (histogram intersection) | maximize | Rewards overlap / correct spread of saliency mass; robust to small shifts. |
| **NSS** (normalized scanpath saliency) | maximize | Enforces high predicted saliency at fixation locations. |
| **MSE** | minimize | Dense per-pixel penalty that stabilizes optimization. |

**Stage-wise objectives.** Stage I and II optimize \(\mathcal{L}^{(\mathrm{I,II})}=\mathcal{L}_{\text{sal}}\). Stage III adds the autoregressive token-level cross-entropy text loss while keeping \(\mathcal{L}_{\text{sal}}\) as a fixed localization constraint (no gradients into the frozen saliency branch): \(\mathcal{L}^{(\mathrm{III})}=\alpha\,\mathcal{L}_{\text{sal}}+\beta\,\mathcal{L}_{\text{text}}\).

---

## Three-Stage Training

| Stage | What is learned | Trainable | Frozen |
|-------|-----------------|-----------|--------|
| **Stage I** — saliency localization | A strong, domain-agnostic localization prior using saliency supervision only (no language). | Visual Encoder + Saliency Decoder | — |
| **Stage II** — attach the VLM | Introduce text conditioning without destabilizing the LM; continue saliency training in the VLM representation space. | Visual Encoder, Saliency Decoder, Visual Adapter | Qwen transformer, `embed_tokens`, `lm_head` |
| **Stage III** — adapt the language side | Sharpen grounded explanations while preserving localization. | Visual Adapter, LoRA on the Qwen transformer, `lm_head`, required norms | Visual Encoder, Saliency Decoder |

### Frozen vs Trainable (per component)

| Component                        | Stage I         | Stage II        | Stage III                 |
|----------------------------------|-----------------|-----------------|---------------------------|
| **DINOv3 ViT-B/16**              | Trainable       | Trainable       | Frozen*                   |
| **VisualAdapter** (Linear Proj.) | —               | Trainable (new) | Trainable*                |
| **PatchMerger** (Spatial Merge)  | —               | Trainable (new) | Trainable*                |
| **proj_to_l1/l2/l3** (1×1 conv)  | Trainable       | Trainable       | Frozen*                   |
| **proj_to_l4** (1×1 conv)        | —               | Trainable (new) | Frozen*                   |
| **Saliency Decoder** (layer_rn, refinenet) | Trainable | Trainable    | Frozen*                   |
| **output_conv** (saliency head)  | Trainable       | Trainable       | Frozen*                   |
| **embed_tokens** (word emb.)     | —               | Frozen          | Frozen                    |
| **Qwen Transformer** (base)      | —               | Frozen          | Frozen                    |
| **Qwen Transformer** (LoRA)      | —               | —               | **Trainable**             |
| **qwen_norm** (final LN)         | —               | Frozen          | Frozen*                   |
| **lm_head** (text output)        | —               | Frozen          | **Trainable**             |
| **Tokenizer**                    | —               | Fixed           | Fixed                     |

\* In Stage III the saliency pathway (encoder + decoder) is frozen so localization is preserved; only the visual adapter and the LoRA-adapted language side are updated. Freezing behavior is configurable via command-line flags (`--freeze_dino`, `--freeze_dpt`, `--freeze_projector`, `--train_qwen_norm`, etc.).

---

## Model Variants

The Saliency Decoder and DINOv3 encoder are shared; variants differ only in the Qwen-VL backbone used for the semantic head:

| Variant | Qwen-VL size |
|---------|--------------|
| OpenVAM-3B | Qwen2.5-VL-3B |
| OpenVAM-4B | Qwen3-VL-4B |
| OpenVAM-7B | Qwen2.5-VL-7B |
| OpenVAM-8B | Qwen3-VL-8B |

Stage III uses LoRA (default \(r=16,\ \alpha=32,\ \text{dropout}=0.05\)) on the language transformer, LM head, and required norms. Full per-stage hyperparameters (learning rate, warmup, epochs, and the loss weights \(\lambda_*\)) are listed in the paper's supplementary "Training hyperparameters" table; for the default OpenVAM-3B these are roughly Stage I lr \(7.4\times10^{-5}\) with \((\lambda_{\text{MSE}},\lambda_{\text{KLD}},\lambda_{\text{CC}},\lambda_{\text{SIM}},\lambda_{\text{NSS}})=(2.99,12,2.15,1.69,3.17)\), and Stage III lr \(1\times10^{-4}\) with \(\alpha=0.05,\ \beta=3.0\).
