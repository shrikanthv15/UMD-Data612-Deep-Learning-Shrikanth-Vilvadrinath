# Task P2-06: Implement Training Loop with AMP

## Context
Training loop for the DiT diffusion model. Uses PyTorch AMP (Automatic Mixed
Precision) for 2x speedup and 50% memory reduction on GPU.

## Inputs
- `code/src/diffusion.py` (cosine schedule, forward process)
- `code/src/dit.py` (DiT backbone)
- `code/src/dataset.py` (MVTec DataLoader)

## Output
`Team Project/code/src/train.py`

## Specification

```python
def train(
    category: str,
    data_root: str = "data/mvtec",
    img_size: int = 128,
    batch_size: int = 16,
    epochs: int = 100,
    lr: float = 1e-4,
    timesteps: int = 1000,
    checkpoint_dir: str = "output/checkpoints",
    device: str = "cuda",
    seed: int = 42,
):
    """
    Train DiT on normal images from one MVTec category.
    
    Training algorithm:
        For each epoch:
            For each batch of normal images x_0:
                1. Sample random timesteps t ~ Uniform(0, T-1)
                2. Sample noise epsilon ~ N(0, I)
                3. Compute x_t = q_sample(x_0, t, epsilon)
                4. Predict epsilon_hat = model(x_t, t)
                5. Loss = MSE(epsilon, epsilon_hat)
                6. Backprop with AMP scaler
        
    Saves:
        - Checkpoint every 10 epochs: {checkpoint_dir}/{category}_epoch{N}.pt
        - Final checkpoint: {checkpoint_dir}/{category}_final.pt
        - Training loss log: {checkpoint_dir}/{category}_loss.csv
    
    AMP usage:
        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            loss = ...
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    Optimizer: AdamW(lr=1e-4, weight_decay=0.01)
    Scheduler: CosineAnnealingLR(T_max=epochs)
    
    Prints per epoch (clean, no verbose progress bars):
        "Epoch [001/100] | Loss: 0.0423 | LR: 1.00e-04"
    """

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_root", type=str, default="data/mvtec")
    args = parser.parse_args()
    train(**vars(args))
```

## Success Criteria
- `python train.py --category hazelnut --epochs 5` runs without error
- Loss decreases over 5 epochs
- Checkpoint file saved to output/checkpoints/
- Loss CSV written with columns: epoch, loss, lr
- AMP enabled (verify with `torch.cuda.is_available()` check, graceful CPU fallback)
- Memory usage < 10GB on T4 (16GB) for batch_size=16, img_size=128
- Reproducible: same seed -> same loss curve
