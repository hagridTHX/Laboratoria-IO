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
        # trik reparametryzacji - losowosc siedzi w eps, gradient leci przez mu i std
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

    def sample(self, n, device):
        # nowe cyfry biore prosto z rozkladu N(0, I), enkoder nie jest potrzebny
        z = torch.randn(n, self.latent_dim, device=device)
        out = torch.sigmoid(self.decode(z))
        return out.view(n, 1, 28, 28)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = Path(os.path.expanduser("~/data"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    mnist_transform = T.Compose([T.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=mnist_transform)
    mnist_loader = DataLoader(mnist_train, batch_size=256, shuffle=True, num_workers=2)

    # trening VAE przez 5 epok
    EPOCHS_VAE = 5
    model_vae = VAE(LATENT_DIM).to(device)
    opt_vae = torch.optim.Adam(model_vae.parameters(), lr=1e-3)

    vae_losses = []
    model_vae.train()
    for epoch in range(1, EPOCHS_VAE + 1):
        total = 0.0
        for x, _ in mnist_loader:
            x = x.to(device)
            opt_vae.zero_grad()
            recon, mu, logvar = model_vae(x)
            loss = elbo_loss_ref(recon, x.view(-1, 784), mu, logvar)
            loss.backward()
            opt_vae.step()
            total += loss.item()
        avg = total / len(mnist_train)
        vae_losses.append(avg)
        print(f"Epoch {epoch}/{EPOCHS_VAE}  ELBO/sample={avg:.2f}")

    # generuje 16 nowych cyfr i je pokazuje
    model_vae.eval()
    with torch.no_grad():
        samples = model_vae.sample(16, device).cpu()

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i, 0], cmap="gray")
        ax.axis("off")
    plt.suptitle("VAE samples after training")
    plt.tight_layout(); plt.show()

    # sprawdzenie
    assert samples.shape == (16, 1, 28, 28), f"Expected (16,1,28,28), got {samples.shape}"
    assert samples.min() >= 0.0 and samples.max() <= 1.0, "Samples must be in [0,1]"
    assert vae_losses[-1] < vae_losses[0], "ELBO loss should decrease"
    print("\n✓ Exercise 3 passed!")


if __name__ == '__main__':
    main()
