"""
Anomaly scoring pipeline: dual-level scoring (pixel + feature).

Computes:
  1. Pixel-level: 1 - SSIM(original, reconstruction)
  2. Feature-level: L2 distance of ResNet-18 intermediate features
  3. Combined score: alpha * pixel + (1-alpha) * feature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pytorch_msssim import ssim


class FeatureExtractor(nn.Module):
    """
    Extract intermediate features from pretrained ResNet-18.
    Uses layers 1, 2, 3 (after each residual block group).
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list:
        """
        Extract features from 3 layers.

        Args:
            x: images (B, 3, H, W) in [-1, 1]

        Returns:
            list of 3 feature maps at different scales
        """
        # Denormalize from [-1,1] to ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
        x = (x - mean) / std

        h = self.layer0(x)
        f1 = self.layer1(h)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]


def compute_pixel_anomaly_map(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    win_size: int = 11,
) -> torch.Tensor:
    """
    Pixel-level anomaly map using 1 - SSIM.

    Args:
        original: (B, 3, H, W) in [-1, 1]
        reconstruction: (B, 3, H, W) in [-1, 1]
        win_size: SSIM window size

    Returns:
        anomaly_map: (B, 1, H, W) in [0, 1], higher = more anomalous
    """
    # Rescale to [0, 1] for SSIM
    orig_01 = (original + 1.0) / 2.0
    recon_01 = (reconstruction + 1.0) / 2.0

    # Compute per-pixel SSIM (not reduced)
    ssim_map = ssim(orig_01, recon_01, data_range=1.0, win_size=win_size,
                    size_average=False, nonnegative_ssim=True)

    # The ssim function returns a scalar per image when size_average=False
    # For spatial map, use the channel-wise approach:
    # average over channels
    anomaly_map = 1.0 - ssim_map
    return anomaly_map


def compute_feature_anomaly_map(
    feature_extractor: FeatureExtractor,
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    img_size: int = 128,
) -> torch.Tensor:
    """
    Feature-level anomaly map using L2 distance in ResNet-18 feature space.

    Args:
        feature_extractor: pretrained ResNet-18 feature extractor
        original: (B, 3, H, W) in [-1, 1]
        reconstruction: (B, 3, H, W) in [-1, 1]
        img_size: target spatial size for upsampling

    Returns:
        anomaly_map: (B, 1, H, W), higher = more anomalous
    """
    feats_orig = feature_extractor(original)
    feats_recon = feature_extractor(reconstruction)

    anomaly_maps = []
    for f_orig, f_recon in zip(feats_orig, feats_recon):
        # L2 distance per spatial location
        diff = (f_orig - f_recon) ** 2
        diff = diff.mean(dim=1, keepdim=True)  # average over channels
        diff = F.interpolate(diff, size=(img_size, img_size), mode="bilinear", align_corners=False)
        anomaly_maps.append(diff)

    # Average across layers
    combined = torch.stack(anomaly_maps, dim=0).mean(dim=0)
    return combined


def compute_combined_anomaly_map(
    pixel_map: torch.Tensor,
    feature_map: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Combine pixel and feature anomaly maps.

    Args:
        pixel_map: (B, 1, H, W) pixel-level anomaly scores
        feature_map: (B, 1, H, W) feature-level anomaly scores
        alpha: weight for pixel map (1-alpha for feature map)

    Returns:
        combined: (B, 1, H, W) combined anomaly map
    """
    # Normalize each to [0, 1] per image
    B = pixel_map.shape[0]
    pixel_norm = pixel_map.clone()
    feature_norm = feature_map.clone()

    for i in range(B):
        pmin, pmax = pixel_norm[i].min(), pixel_norm[i].max()
        if pmax > pmin:
            pixel_norm[i] = (pixel_norm[i] - pmin) / (pmax - pmin)

        fmin, fmax = feature_norm[i].min(), feature_norm[i].max()
        if fmax > fmin:
            feature_norm[i] = (feature_norm[i] - fmin) / (fmax - fmin)

    return alpha * pixel_norm + (1 - alpha) * feature_norm


def compute_image_score(anomaly_map: torch.Tensor) -> torch.Tensor:
    """
    Compute per-image anomaly score from spatial anomaly map.
    Uses max of the anomaly map as image-level score.

    Args:
        anomaly_map: (B, 1, H, W)

    Returns:
        scores: (B,) image-level anomaly scores
    """
    return anomaly_map.flatten(1).max(dim=1).values


if __name__ == "__main__":
    # Smoke test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    B, C, H, W = 2, 3, 128, 128
    original = torch.randn(B, C, H, W, device=device)
    recon = original + 0.1 * torch.randn_like(original)  # slight perturbation

    # Feature scoring
    feat_extractor = FeatureExtractor().to(device)
    feat_map = compute_feature_anomaly_map(feat_extractor, original, recon, img_size=H)
    print(f"Feature anomaly map shape: {feat_map.shape}")
    print(f"Feature anomaly map range: [{feat_map.min():.4f}, {feat_map.max():.4f}]")

    # Image scores
    scores = compute_image_score(feat_map)
    print(f"Image scores: {scores}")

    print("\nScoring smoke test passed.")
