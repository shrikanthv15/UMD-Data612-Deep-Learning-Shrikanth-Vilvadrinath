# Task P2-01: Implement Cosine Noise Schedule
# Task P2-02: Implement Forward Process

## Context
These two tasks produce the noise schedule and forward diffusion process.
Both go into the same file: `code/src/diffusion.py`.

## Output
`Team Project/code/src/diffusion.py`

## Specification

### Cosine Noise Schedule (P2-01)
Reference: Nichol & Dhariwal, "Improved DDPM", 2021.

```python
def cosine_beta_schedule(timesteps: int = 1000, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule from Nichol & Dhariwal 2021.
    
    Args:
        timesteps: total diffusion steps T (default 1000)
        s: small offset to prevent beta_t from being too small at t=0
    
    Returns:
        betas: Tensor of shape (timesteps,) with values in (0, 0.999)
    
    Formula:
        f(t) = cos((t/T + s) / (1 + s) * pi/2)^2
        alpha_bar_t = f(t) / f(0)
        beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        beta_t = clip(beta_t, 0, 0.999)
    """
```

### Forward Process (P2-02)
```python
class GaussianDiffusion:
    """
    Manages the diffusion schedule and forward/reverse processes.
    
    Args:
        betas: Tensor of noise schedule (from cosine_beta_schedule)
    
    Precomputed on init (register as buffers or store as tensors):
        alphas = 1 - betas
        alphas_cumprod = cumprod(alphas)
        sqrt_alphas_cumprod = sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = sqrt(1 - alphas_cumprod)
    """
    
    def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor = None) -> Tensor:
        """
        Forward process: sample x_t given x_0 and timestep t.
        
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        
        Args:
            x_0: clean images, shape (B, C, H, W)
            t: timesteps, shape (B,) with values in [0, T-1]
            noise: optional pre-sampled noise, shape (B, C, H, W)
        
        Returns:
            x_t: noised images, shape (B, C, H, W)
        """
    
    def _extract(self, tensor, t, x_shape):
        """Extract values from tensor at timestep t, reshape for broadcasting."""
```

## Success Criteria
- `cosine_beta_schedule(1000)` returns tensor of shape (1000,), all values in (0, 0.999)
- `betas` are monotonically increasing (mostly)
- `alphas_cumprod[-1]` is close to 0 (signal nearly destroyed at T=1000)
- `alphas_cumprod[0]` is close to 1 (almost no noise at t=0)
- `q_sample(x_0, t=0)` returns something very close to x_0
- `q_sample(x_0, t=999)` returns near-pure noise
- Visual test: plot x_0, x_100, x_250, x_500, x_999 side by side
