# Task P0-03: Write MVTec Dataset Class

## Context
PyTorch Dataset class that loads MVTec AD images for training and evaluation.
Must handle the MVTec folder structure, per-category loading, and augmentations.

## Inputs
- MVTec AD dataset in `data/mvtec/` with structure:
  ```
  data/mvtec/{category}/train/good/*.png      (normal training images)
  data/mvtec/{category}/test/good/*.png        (normal test images)
  data/mvtec/{category}/test/{defect}/*.png    (anomalous test images)
  data/mvtec/{category}/ground_truth/{defect}/*.png  (pixel-level masks)
  ```
- Categories: bottle, cable, capsule, carpet, grid, hazelnut, leather,
  metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

## Output
`Team Project/code/src/dataset.py`

## Specification

```python
class MVTecDataset(torch.utils.data.Dataset):
    """
    Args:
        root: str -- path to mvtec/ folder (e.g., "data/mvtec")
        category: str -- one of the 15 category names
        split: str -- "train" or "test"
        img_size: int -- resize to (img_size, img_size), default 128
        transform: optional torchvision transform override
    
    Returns (for each __getitem__):
        If split == "train":
            image: Tensor (3, img_size, img_size) normalized to [-1, 1]
        If split == "test":
            image: Tensor (3, img_size, img_size) normalized to [-1, 1]
            mask: Tensor (1, img_size, img_size) binary ground truth (0 or 1)
            label: int (0 = normal, 1 = anomalous)
    """
```

### Training augmentations (applied when split=="train"):
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=10)`
- `ColorJitter(brightness=0.1, contrast=0.1)`
- Resize to img_size x img_size
- ToTensor
- Normalize to [-1, 1]: `Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])`

### Test transforms (no augmentation):
- Resize to img_size x img_size
- ToTensor
- Normalize to [-1, 1]

### Ground truth mask handling:
- For "good" test images: return all-zeros mask
- For defect test images: load from `ground_truth/{defect_type}/{filename}_mask.png`
- Resize mask to img_size x img_size using NEAREST interpolation (no antialiasing)
- Binarize: mask > 0.5 -> 1, else 0

### Helper function:
```python
def get_mvtec_categories() -> list[str]:
    """Return sorted list of all 15 MVTec category names."""

def get_dataloaders(root, category, img_size=128, batch_size=16, num_workers=4):
    """Return (train_loader, test_loader) for a category."""
```

## Success Criteria
- `MVTecDataset("data/mvtec", "hazelnut", "train")` loads without error
- `len(dataset)` returns correct count
- `dataset[0]` returns tensor with shape (3, 128, 128) in range [-1, 1]
- Test dataset returns (image, mask, label) tuple
- Mask shape is (1, 128, 128), label is 0 or 1
