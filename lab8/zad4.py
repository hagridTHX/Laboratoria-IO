import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

LATENT_DIM = 16


def elbo_loss_ref(x_recon, x_orig, mu, logvar):
    bce = F.binary_cross_entropy(x_recon, x_orig, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kl


# model i trening z zadania 3, przeklejam zeby skrypt dzialal sam z siebie
class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(784, 512), nn.ReLU())
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 784),
        )

    def encode(self, x):
        h = self.encoder(x.view(-1, 784))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = torch.sigmoid(self.decode(z))
        return recon, mu, logvar


def get_first_digit(dataset, target_label):
    for img, label in dataset:
        if label == target_label:
            return img.unsqueeze(0)
    raise ValueError(f"No digit {target_label} found")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = Path(os.path.expanduser("~/data"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    mnist_transform = T.Compose([T.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=mnist_transform)
    mnist_loader = DataLoader(mnist_train, batch_size=256, shuffle=True, num_workers=2)

    model_vae = VAE(LATENT_DIM).to(device)
    opt_vae = torch.optim.Adam(model_vae.parameters(), lr=1e-3)

    model_vae.train()
    for epoch in range(1, 6):
        for x, _ in mnist_loader:
            x = x.to(device)
            opt_vae.zero_grad()
            recon, mu, logvar = model_vae(x)
            loss = elbo_loss_ref(recon, x.view(-1, 784), mu, logvar)
            loss.backward()
            opt_vae.step()
        print(f"Epoch {epoch}/5 done")

    # wybieram po jednym przykladzie cyfry 0 i 7
    img_0 = get_first_digit(mnist_train, 0).to(device)
    img_7 = get_first_digit(mnist_train, 7).to(device)

    # koduje oba do srednich latentu
    model_vae.eval()
    with torch.no_grad():
        mu_A, _ = model_vae.encode(img_0)
        mu_B, _ = model_vae.encode(img_7)

    print("mu_A shape:", mu_A.shape)
    print("mu_B shape:", mu_B.shape)

    # interpolacja po prostej miedzy mu_A a mu_B, dekoduje kazdy krok
    N_STEPS = 8
    alphas = torch.linspace(0, 1, N_STEPS)
    interp_images = []
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1 - alpha) * mu_A + alpha * mu_B
            img = torch.sigmoid(model_vae.decode(z_interp)).view(1, 1, 28, 28)
            interp_images.append(img)

    fig, axes = plt.subplots(1, N_STEPS, figsize=(N_STEPS * 1.5, 2))
    for i, ax in enumerate(axes):
        ax.imshow(interp_images[i][0, 0].cpu().numpy(), cmap="gray")
        ax.set_title(f"α={alphas[i]:.2f}", fontsize=7)
        ax.axis("off")
    plt.suptitle('Latent interpolation: "0"  →  "7"')
    plt.tight_layout(); plt.show()

    # sprawdzenie
    assert alphas is not None, "Define alphas using torch.linspace"
    assert len(interp_images) == N_STEPS, \
        f"Expected {N_STEPS} interpolation steps, got {len(interp_images)}"
    assert interp_images[0].shape == (1, 1, 28, 28), \
        f"Each image must be (1,1,28,28), got {interp_images[0].shape}"
    with torch.no_grad():
        expected_first = torch.sigmoid(model_vae.decode(mu_A))
    diff = (interp_images[0].to(device) - expected_first.view(1, 1, 28, 28)).abs().mean().item()
    assert diff < 0.01, f"First interpolation step (α=0) should equal decoding of mu_A, diff={diff:.4f}"
    print("\n✓ Exercise 4 passed!")


if __name__ == '__main__':
    main()
