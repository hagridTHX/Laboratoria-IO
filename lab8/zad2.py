import torch
import torch.nn.functional as F


def elbo_loss_ref(x_recon, x_orig, mu, logvar):
    bce = F.binary_cross_entropy(x_recon, x_orig, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kl


def main():
    # wyjscia enkodera - batch 4 probek, wymiar latentu 16
    torch.manual_seed(42)
    mu = torch.randn(4, 16) * 0.5
    logvar = torch.randn(4, 16) * 0.3 - 1.0

    x_orig = torch.rand(4, 784).clamp(0.01, 0.99)
    x_recon = torch.sigmoid(torch.randn(4, 784))

    # czlon rekonstrukcji - kara za zle odtworzenie wejscia (sumujemy, nie srednia)
    bce = F.binary_cross_entropy(x_recon, x_orig, reduction="sum").item()

    # KL liczymy z wzoru zamknietego, logvar trzyma log wariancji wiec exp(logvar)=sigma^2
    kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item()

    elbo_loss = bce + kl

    print(f"Reconstruction (BCE) : {bce:.2f}")
    print(f"KL divergence        : {kl:.2f}")
    print(f"ELBO loss            : {elbo_loss:.2f}")

    ref = elbo_loss_ref(x_recon, x_orig, mu, logvar).item()
    print(f"Reference ELBO       : {ref:.2f}")
    print(f"Difference           : {abs(elbo_loss - ref):.6f}")

    # sprawdzenie
    assert bce is not None, "Compute bce"
    assert kl is not None, "Compute kl"
    assert elbo_loss is not None, "Compute elbo_loss"
    assert kl >= 0, f"KL divergence must be non-negative, got {kl:.4f}"
    assert abs(elbo_loss - ref) < 0.1, \
        f"ELBO does not match reference: your={elbo_loss:.2f} ref={ref:.2f}"
    print("\n✓ Exercise 2 passed!")


if __name__ == '__main__':
    main()
