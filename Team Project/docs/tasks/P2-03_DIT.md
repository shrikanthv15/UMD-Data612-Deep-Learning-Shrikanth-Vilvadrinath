# Task P2-03: Implement DiT Backbone

## Context
Diffusion Transformer (DiT) adapted from Peebles & Xie 2023 (facebookresearch/DiT).
This replaces the traditional UNet backbone. It takes a noised image + timestep
and predicts the noise to be removed.

## Reference
- Paper: "Scalable Diffusion Models with Transformers" (Peebles & Xie, ICCV 2023)
- Code: https://github.com/facebookresearch/DiT (CC-BY-NC 4.0, cite in report)

## Output
`Team Project/code/src/dit.py`

## Specification

```python
class DiT(nn.Module):
    """
    Diffusion Transformer for noise prediction.
    
    Args:
        img_size: int = 128          # input image resolution
        patch_size: int = 4          # patch size for patchification
        in_channels: int = 3         # RGB
        hidden_size: int = 384       # transformer hidden dimension (DiT-S)
        depth: int = 12              # number of transformer blocks
        num_heads: int = 6           # attention heads (hidden_size / 64)
        mlp_ratio: float = 4.0       # MLP hidden dim = hidden_size * mlp_ratio
    
    Architecture:
        1. PatchEmbed: Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
           Input (B, 3, 128, 128) -> (B, num_patches, hidden_size)
           num_patches = (128/4)^2 = 1024
        
        2. Positional encoding: learnable 2D positional embeddings
           pos_embed: Parameter(1, num_patches, hidden_size)
        
        3. Time embedding: 
           sinusoidal_embedding(t) -> MLP(256 -> hidden_size -> hidden_size)
        
        4. N x DiTBlock:
           - AdaLN: scale, shift, gate = MLP(time_embed) -> 6 * hidden_size
           - x = gate_msa * Attention(AdaLN(x, scale_1, shift_1)) + x
           - x = gate_mlp * MLP(AdaLN(x, scale_2, shift_2)) + x
        
        5. Final layer: AdaLN -> Linear(hidden_size, patch_size^2 * in_channels)
        
        6. Unpatchify: reshape back to (B, 3, 128, 128)
    
    Forward:
        def forward(self, x: Tensor, t: Tensor) -> Tensor:
            '''
            Args:
                x: noised images (B, 3, 128, 128)
                t: timesteps (B,) integers in [0, T-1]
            Returns:
                predicted noise (B, 3, 128, 128)
            '''
    """
```

### Key components to implement:

```python
class PatchEmbed(nn.Module):
    """Image to patch embedding via Conv2d."""

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep -> MLP -> hidden_size vector."""

class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning."""

def modulate(x, shift, scale):
    """Apply adaptive layer norm: scale * x + shift."""

def unpatchify(x, patch_size, img_size, channels):
    """Reshape (B, num_patches, patch_dim) -> (B, C, H, W)."""
```

### DiT-S configuration (default):
- hidden_size: 384
- depth: 12
- num_heads: 6
- patch_size: 4
- Total params: ~33M (fits in Colab T4 16GB easily)

### If Colab T4 VRAM is tight, fall back to DiT-Tiny:
- hidden_size: 192
- depth: 6
- num_heads: 3
- Total params: ~5M

## Success Criteria
- `DiT()(torch.randn(2, 3, 128, 128), torch.tensor([0, 500]))` returns shape (2, 3, 128, 128)
- No NaN in output
- Parameter count matches expected (~33M for DiT-S)
- Forward pass takes < 100ms on CPU for batch_size=2
