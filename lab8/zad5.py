import os
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path


class LinearNoiseScheduler:
    """Liniowy harmonogram szumu DDPM, beta rosnie liniowo, T=1000 krokow."""
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def forward_diffuse(self, x0: torch.Tensor, t: int) -> torch.Tensor:
        # wzor zamkniety - jednym krokiem dostajemy x_t zamiast t-krotnego szumienia
        alpha_bar_t = self.alpha_bar[t]
        eps = torch.randn_like(x0)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps


def main():
    DATA_DIR = Path(os.path.expanduser("~/data"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # normalizacja do [-1, 1], tak jak w DL08
    img_transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    flowers_diff = torchvision.datasets.Flowers102(DATA_DIR, split="test", download=True, transform=img_transform)
    x0, _ = flowers_diff[5]
    print("x0 shape:", x0.shape, "  range:", x0.min().item(), "–", x0.max().item())

    scheduler = LinearNoiseScheduler(T=1000, device="cpu")
    timesteps = [0, 100, 250, 500, 750, 999]

    # dla t=0 zostawiam czysty obraz, dalej szumie wzorem
    noisy_images = []
    for t in timesteps:
        if t == 0:
            noisy_images.append(x0.clone())
        else:
            noisy_images.append(scheduler.forward_diffuse(x0, t))

    # podglad jak obraz stopniowo zamienia sie w szum
    fig, axes = plt.subplots(1, len(timesteps), figsize=(14, 3))
    for ax, t, img in zip(axes, timesteps, noisy_images):
        display = ((img.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        ax.imshow(display)
        ax.set_title(f"t={t}", fontsize=9)
        ax.axis("off")
    plt.suptitle("DDPM forward diffusion (t=0 → t=999)")
    plt.tight_layout(); plt.show()

    # histogram pikseli na poczatku i na koncu
    pix_t0 = noisy_images[0].flatten().numpy()
    pix_t999 = noisy_images[-1].flatten().numpy()

    plt.figure(figsize=(8, 3))
    plt.hist(pix_t0, bins=60, alpha=0.6, label="t=0 (original)")
    plt.hist(pix_t999, bins=60, alpha=0.6, label="t=999 (noisy)")
    plt.xlabel("Pixel value"); plt.ylabel("Count")
    plt.title("Pixel histogram: original vs fully-diffused")
    plt.legend(); plt.tight_layout(); plt.show()

    # na koncu rozklad powinien wygladac jak szum gaussowski (std ~ 1)
    std_t0 = noisy_images[0].std().item()
    std_t999 = noisy_images[-1].std().item()
    print(f"std at t=0  : {std_t0:.3f}  (structured image)")
    print(f"std at t=999: {std_t999:.3f}  (should approach 1.0 — pure Gaussian)")

    stds = [img.std().item() for img in noisy_images]
    print("Noise level across timesteps:",
          [f"t={t}: {s:.2f}" for t, s in zip(timesteps, stds)])

    # sprawdzenie
    assert all(img.shape == x0.shape for img in noisy_images), \
        "All noisy images must have the same shape as x0"
    assert abs(std_t999 - 1.0) < 0.15, \
        f"At t=999 std should be ~1.0 (Gaussian), got {std_t999:.3f}"
    assert stds[-1] > stds[0], \
        "Noise level must increase monotonically: std at t=999 should exceed std at t=0"
    assert scheduler.alpha_bar[-1].item() < 0.01, \
        f"alpha_bar at t=999 should be near 0, got {scheduler.alpha_bar[-1].item():.4f}"
    print("\n✓ Exercise 5 passed!")


if __name__ == '__main__':
    main()
