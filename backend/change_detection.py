"""
ChangeFormerV6 Bi-Temporal Change Detection Module.
Provides generic learned temporal change detection using the official
pretrained ChangeFormerV6 checkpoint trained on LEVIR-CD.

IMPORTANT: ChangeFormerV6 detects generic surface changes (optical alterations).
It is NOT a vegetation-specific classifier. In Sentinel-2 mode, vegetation
semantics are determined by the subsequent NDVI analytical stage.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

# Ensure changeformer_repo is accessible for importing the exact model layers
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_PATH = PROJECT_ROOT / "changeformer_repo"
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

try:
    from models.ChangeFormer import ChangeFormerV6
except ImportError:
    # If not in path, try relative
    from changeformer_repo.models.ChangeFormer import ChangeFormerV6


CHECKPOINT_DIR = (
    PROJECT_ROOT / "checkpoints" / "ChangeFormer_LEVIR" /
    "CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256"
)
CHECKPOINT_FILE = CHECKPOINT_DIR / "best_ckpt.pt"


@dataclass
class ChangeDetectionResult:
    change_mask: np.ndarray        # 2D uint8 binary mask (0 = unchanged, 1 = changed)
    change_prob: np.ndarray        # 2D float32 probability map in [0.0, 1.0]
    total_pixels: int
    changed_pixels: int
    change_percentage: float
    device_used: str
    model_name: str = "ChangeFormerV6 (LEVIR-CD Pretrained)"


class ChangeDetector:
    _instance: Optional["ChangeDetector"] = None

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = ChangeFormerV6(embed_dim=256)

        ckpt_file = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_FILE
        if ckpt_file.exists():
            checkpoint = torch.load(str(ckpt_file), map_location=self.device, weights_only=False)
            if "model_G_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_G_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.checkpoint_loaded = True
        else:
            self.checkpoint_loaded = False

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def get_instance(cls) -> "ChangeDetector":
        if cls._instance is None:
            cls._instance = ChangeDetector()
        return cls._instance

    def preprocess_image(self, img_array: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Normalizes image array to [-1.0, 1.0] as expected by ChangeFormer (mean=0.5, std=0.5).
        Input should be (H, W, 3) or (H, W) in uint8 [0, 255] or float [0, 1].
        """
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] > 3:
            # Extract first 3 bands (e.g. RGB)
            img_array = img_array[:, :, :3]

        if target_size is not None and (img_array.shape[0] != target_size[0] or img_array.shape[1] != target_size[1]):
            pil_img = Image.fromarray(img_array.astype(np.uint8) if img_array.max() > 1.0 else (img_array * 255).astype(np.uint8))
            pil_img = pil_img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
            img_array = np.array(pil_img)

        # Ensure float in [0, 1]
        if img_array.dtype == np.uint8 or img_array.max() > 1.0:
            arr = img_array.astype(np.float32) / 255.0
        else:
            arr = img_array.astype(np.float32)

        # Standard ChangeFormer normalization: (x - 0.5) / 0.5
        norm_arr = (arr - 0.5) / 0.5
        # Convert (H, W, C) -> (1, C, H, W)
        tensor = torch.from_numpy(norm_arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor

    def detect_changes(
        self,
        t1_img: np.ndarray,
        t2_img: np.ndarray,
        threshold: float = 0.5,
        min_region_size: int = 16,
    ) -> ChangeDetectionResult:
        """
        Runs ChangeFormerV6 inference on the T1 and T2 images.
        Supports sliding window / tiling for high resolutions or direct inference for standard sizes.
        Post-processes binary change mask with morphological cleanup.
        """
        orig_h, orig_w = t1_img.shape[:2]

        # For standard evaluation/demo sizes (e.g. 256x256, 512x512)
        # We perform inference at 256x256 tiles or resized depending on dimensions
        if orig_h == 256 and orig_w == 256:
            t1_tensor = self.preprocess_image(t1_img)
            t2_tensor = self.preprocess_image(t2_img)
            with torch.no_grad():
                preds = self.model(t1_tensor, t2_tensor)
                final_logits = preds[-1]  # (1, 2, 256, 256)
                probs = F.softmax(final_logits, dim=1)[:, 1, :, :].squeeze(0).cpu().numpy()
        elif orig_h == 512 and orig_w == 512:
            # 512x512 images (e.g. SECOND/CDVQA dataset standard resolution):
            # Process in four 256x256 non-overlapping quadrants for maximum fidelity
            probs = np.zeros((512, 512), dtype=np.float32)
            for y_start in [0, 256]:
                for x_start in [0, 256]:
                    crop1 = t1_img[y_start:y_start+256, x_start:x_start+256]
                    crop2 = t2_img[y_start:y_start+256, x_start:x_start+256]
                    t1_tensor = self.preprocess_image(crop1)
                    t2_tensor = self.preprocess_image(crop2)
                    with torch.no_grad():
                        preds = self.model(t1_tensor, t2_tensor)
                        sub_prob = F.softmax(preds[-1], dim=1)[:, 1, :, :].squeeze(0).cpu().numpy()
                        probs[y_start:y_start+256, x_start:x_start+256] = sub_prob
        else:
            # General arbitrary resolution: resize to multiple of 256 or interpolate
            t1_tensor = self.preprocess_image(t1_img, target_size=(256, 256))
            t2_tensor = self.preprocess_image(t2_img, target_size=(256, 256))
            with torch.no_grad():
                preds = self.model(t1_tensor, t2_tensor)
                sub_prob = F.softmax(preds[-1], dim=1)[:, 1, :, :].squeeze(0).cpu().numpy()
                # Upsample back to original size
                prob_img = Image.fromarray((sub_prob * 255).astype(np.uint8))
                prob_img = prob_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
                probs = np.array(prob_img).astype(np.float32) / 255.0

        # Apply configurable detection threshold
        binary_mask = (probs >= threshold).astype(np.uint8)

        # Morphological post-processing: remove noise and fill small pinholes
        if min_region_size > 1:
            # Small binary opening to disconnect thin noise
            binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((2, 2))).astype(np.uint8)
            # Remove connected components smaller than min_region_size
            labeled, num_features = ndimage.label(binary_mask)
            if num_features > 0:
                sizes = ndimage.sum(binary_mask, labeled, range(num_features + 1))
                mask_sizes = sizes < min_region_size
                remove_pixel = mask_sizes[labeled]
                binary_mask[remove_pixel] = 0

        total_pixels = int(binary_mask.size)
        changed_pixels = int(np.count_nonzero(binary_mask))
        change_pct = float(changed_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

        return ChangeDetectionResult(
            change_mask=binary_mask,
            change_prob=probs,
            total_pixels=total_pixels,
            changed_pixels=changed_pixels,
            change_percentage=round(change_pct, 4),
            device_used=str(self.device),
        )
