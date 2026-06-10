import os
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def show_masks(image, masks, elapsed_s, ax):
    ax.imshow(image)
    for m in masks:
        seg = m["segmentation"].astype(float)
        rgba = np.zeros((*seg.shape, 4))
        rgba[..., :3] = np.random.rand(3)
        rgba[..., 3] = seg * 0.45
        ax.imshow(rgba)
    ax.set_title(f"{len(masks)} masks | {elapsed_s:.1f} s")
    ax.axis("off")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = Path(os.path.expanduser("~/data"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # wczytuje obrazek z Flowers102 i zamieniam na uint8 HWC, bo tego oczekuje SAM
    flowers = torchvision.datasets.Flowers102(
        DATA_DIR, split="test", download=True,
        transform=T.Compose([T.Resize((400, 400)), T.ToTensor()]),
    )
    img_tensor, label = flowers[0]
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    print(f"Image shape (HWC): {img_np.shape}  label: {label}")

    # wczytanie checkpointu modelu
    SAM_CHECKPOINT = DATA_DIR / "sam_checkpoints" / "sam_vit_b.pth"
    assert SAM_CHECKPOINT.exists(), f"SAM checkpoint not found at {SAM_CHECKPOINT}"
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT)).to(device)
    print("SAM loaded on:", device)

    # automatyczne generowanie masek, mierze ile to trwa
    generator = SamAutomaticMaskGenerator(sam, points_per_side=16)
    start = time.time()
    masks = generator.generate(img_np)
    elapsed_s = time.time() - start
    print(f"Generated {len(masks)} masks in {elapsed_s:.2f} s")

    # oryginal obok nalozonych masek
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
    show_masks(img_np, masks, elapsed_s, axes[1])
    plt.tight_layout(); plt.show()

    # sprawdzenie
    assert masks is not None and len(masks) > 0, "No masks generated"
    assert elapsed_s is not None and elapsed_s < 60, "Mask generation took too long or was not timed"
    assert "segmentation" in masks[0], "Each mask must have a 'segmentation' key"
    assert masks[0]["segmentation"].shape == (400, 400), \
        f"Mask shape should be (400,400), got {masks[0]['segmentation'].shape}"
    print("\n✓ Exercise 1 passed!")


if __name__ == '__main__':
    main()
