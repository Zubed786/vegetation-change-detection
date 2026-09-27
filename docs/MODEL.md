# ChangeFormerV6 Model Architecture & Pretrained Checkpoint

## 1. Overview
**ChangeFormerV6** is a state-of-the-art bi-temporal remote-sensing change-detection architecture combining a hierarchical Transformer Encoder with a Convolutional / Multi-Scale Feature Fusion Decoder. It processes two co-registered optical images $T1$ and $T2$ simultaneously to predict learned surface changes.

- **Paper**: *A Transformer-Based Siamese Network for Change Detection in High-Resolution Remote Sensing Images* (Bandara & Patel, IEEE TGRS).
- **Official Repository**: [https://github.com/wgcban/ChangeFormer](https://github.com/wgcban/ChangeFormer)

---

## 2. Pretrained Checkpoint Used
- **Checkpoint**: `best_ckpt.pt` (492 MB uncompressed state dictionary)
- **Model Variant**: `ChangeFormerV6` (`embed_dim = 256`, `output_nc = 2`)
- **Training Dataset**: **LEVIR-CD** (Large-scale building change detection dataset with 637 bi-temporal image pairs of $1024 \times 1024$ at 0.5m resolution).
- **Reported Benchmark Accuracy**: `best_val_acc = 0.9495` (94.95% validation accuracy).
- **Official Download Source**:
  ```
  https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip
  ```
- **Local Location**:
  `c:\projects\checkpoints\ChangeFormer_LEVIR\CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256\best_ckpt.pt`

---

## 3. Network Architecture & Inputs

```
T1 Image (B, 3, H, W) ───► [ Transformer Encoder ] ───► Feature Pyramids [fx1] ──┐
                                                                                 ▼
                                                                           [ Transformer Decoder ] ──► Change Map (B, 2, H, W)
                                                                                 ▲
T2 Image (B, 3, H, W) ───► [ Transformer Encoder ] ───► Feature Pyramids [fx2] ──┘
```

- **Encoder**: Siamese Hierarchical Transformer Encoder with 4 stages (`embed_dims = [64, 128, 320, 512]`, `depths = [3, 3, 4, 3]`).
- **Decoder**: Multi-scale cross-attention and convolutional fusing layers combining spatial and contextual difference tokens.
- **Input Format**: Two 3-channel optical arrays $(B, 3, H, W)$.
- **Preprocessing & Normalization**:
  $$x_{\text{norm}} = \frac{(x / 255.0) - 0.5}{0.5} \in [-1.0, 1.0]$$
- **Output Format**:
  Deep supervision list of 5 multi-scale predictions. The final prediction `preds[-1]` has shape $(B, 2, H, W)$:
  - Channel 0: Logit for Class 0 (Unchanged).
  - Channel 1: Logit for Class 1 (Changed).
  $$\text{Change Probability} = \text{Softmax}(\text{Logits})[:, 1, :, :]$$

---

## 4. Inference Procedure & Tiling Strategy
- **Standard Resolutions ($256 \times 256$)**: Direct forward pass through encoder-decoder.
- **High-Resolution Rasters ($512 \times 512$, e.g., CDVQA / SECOND)**:
  Processed via non-overlapping $256 \times 256$ quadrant tiles and seamless spatial stitching to prevent downsampling artifacts.
- **Hardware Requirements**:
  - **CPU Inference**: Fully supported out-of-the-box using PyTorch CPU wheel. Typical inference takes $\sim 1.5 - 2.5\text{ seconds}$ per $256 \times 256$ tile on modern multi-core laptop CPUs.
  - **GPU Inference**: Automatically enabled if CUDA is available (`cuda:0`).
  - **Memory Footprint**: Approximately 1.2 GB RAM during model loading and inference.

---

## 5. Critical Model Limitations

> [!WARNING]
> **ChangeFormerV6 is a Generic Change Detector, NOT a Vegetation Classifier**:
> 1. **Training Domain**: ChangeFormerV6 in this checkpoint was trained exclusively on **LEVIR-CD building change imagery** (detecting construction, building appearance/disappearance, and earthworks).
> 2. **Generic Learned Change Signal**: The model outputs a binary indication of whether surface reflectance and spatial structures changed. It **does not** know whether that change was deforestation, agricultural harvesting, crop growth, or construction.
> 3. **Vegetation Classification Decoupling**: In this platform, vegetation classification is handled strictly by the separate **NDVI analytical stage** when multispectral data is available. ChangeFormer output is combined with the vegetation masks rather than being interpreted as vegetation change directly.
