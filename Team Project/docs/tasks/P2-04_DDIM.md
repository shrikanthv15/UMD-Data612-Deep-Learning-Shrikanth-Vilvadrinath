# Task P2-04: Implement DDIM Reverse Sampling

## Context
DDIM (Song et al. 2021) enables fast deterministic sampling using 50 steps
instead of the 1000 steps required by DDPM. This is critical for practical inference.

## Output
Add to `Team Project/code/src/diffusion.py` (extends GaussianDiffusion class)

## Specification

```python
# Add these methods to GaussianDiffusion class:

def ddim_sample(
    self,
    model: nn.Module,
    x_t: Tensor,
    t_start: int,
    num_steps: int = 50,
    eta: float = 0.0,   # eta=0 -> deterministic DDIM; eta=1 -> DDPM
) -> Tensor:
    """
    DDIM reverse sampling from x_{t_start} back to x_0.
    
    Args:
        model: trained DiT (or UNet) that predicts noise
        x_t: noised image at timestep t_start, shape (B, C, H, W)
        t_start: starting timestep (e.g., 250 for partial reconstruction)
        num_steps: number of DDIM steps (default 50)
        eta: stochasticity parameter (0 = deterministic)
    
    Returns:
        x_0_hat: reconstructed image, shape (B, C, H, W)
    
    Algorithm:
        1. Create subsequence of timesteps: [t_start, ..., 0] with num_steps entries
           E.g., for t_start=250 and num_steps=50: step size = 5
        2. For each consecutive pair (t, t_prev) in the subsequence:
           a. predicted_noise = model(x_t, t)
           b. predicted_x0 = (x_t - sqrt(1-alpha_bar_t) * predicted_noise) / sqrt(alpha_bar_t)
           c. direction = sqrt(1-alpha_bar_{t_prev} - sigma^2) * predicted_noise
           d. sigma = eta * sqrt((1-alpha_bar_{t_prev})/(1-alpha_bar_t)) * sqrt(1-alpha_bar_t/alpha_bar_{t_prev})
           e. noise = torch.randn_like(x_t) if t_prev > 0 else 0
           f. x_{t_prev} = sqrt(alpha_bar_{t_prev}) * predicted_x0 + direction + sigma * noise
        3. Return final x_0_hat
    """

def reconstruct(
    self,
    model: nn.Module,
    x_0: Tensor,
    t_partial: int = 250,
    num_ddim_steps: int = 50,
) -> Tensor:
    """
    Full anomaly detection reconstruction pipeline.
    
    1. Add t_partial steps of noise to x_0
    2. Run DDIM reverse to get x_0_hat
    
    Args:
        model: trained noise prediction model
        x_0: clean test image, shape (B, C, H, W)
        t_partial: number of noise steps to add (default 250, tuned via ablation)
        num_ddim_steps: DDIM sampling steps (default 50)
    
    Returns:
        x_0_hat: reconstruction, shape (B, C, H, W)
    """
```

## Success Criteria
- `ddim_sample` with eta=0 is deterministic (same input -> same output)
- `reconstruct(model, x_0, t_partial=0)` returns x_0 unchanged (no noise added)
- `reconstruct(model, x_0, t_partial=250)` returns a plausible reconstruction
- 50 DDIM steps produce similar quality to 1000 DDPM steps (visual check)
- Inference time for 50 steps << inference time for 1000 steps
