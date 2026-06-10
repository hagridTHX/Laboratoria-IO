import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# model z zadania 3, przeklejam zeby train_loop mial z czego korzystac
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
    def forward(self, x):
        residual = x
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        if self.downsample is not None: residual = self.downsample(x)
        out += residual
        return self.relu(out)

class TinyResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(256, num_classes)
    def forward(self, x):
        return self.fc(self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))).flatten(1))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = Path(os.path.expanduser("~/data"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # dwie wersje przetwarzania - bez i z augmentacja
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

    transform_A = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    # augmentacje musza byc przed ToTensor, bo dzialaja na obrazie PIL
    transform_B = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

    # zbiory i loadery dla obu wersji
    SUBSET_SIZE = 4_000
    EPOCHS_EXP  = 4
    BATCH_SIZE  = 128

    ds_A = Subset(torchvision.datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=transform_A), range(SUBSET_SIZE))
    ds_B = Subset(torchvision.datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=transform_B), range(SUBSET_SIZE))

    loader_A = DataLoader(ds_A, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    loader_B = DataLoader(ds_B, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # jedna petla treningowa zeby nie powtarzac kodu dwa razy
    def train_loop(loader, epochs, subset_size):
        m   = TinyResNet().to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        curve = []
        m.train()
        for ep in range(1, epochs + 1):
            total = 0.0
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                loss = crit(m(imgs), labels)
                loss.backward()
                opt.step()
                total += loss.item() * len(imgs)
            avg = total / subset_size
            curve.append(avg)
            print(f"  epoch {ep}/{epochs}  loss={avg:.4f}")
        return curve

    print("=== Pipeline A (minimal) ===")
    losses_A = train_loop(loader_A, EPOCHS_EXP, SUBSET_SIZE)

    print("\n=== Pipeline B (augmented) ===")
    losses_B = train_loop(loader_B, EPOCHS_EXP, SUBSET_SIZE)

    # wykres obu krzywych
    ep_range = range(1, EPOCHS_EXP + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(ep_range, losses_A, marker="o", label="A — minimal")
    plt.plot(ep_range, losses_B, marker="s", label="B — augmented")
    plt.xlabel("Epoch"); plt.ylabel("Avg loss")
    plt.title("Effect of data augmentation on training loss")
    plt.legend(); plt.tight_layout(); plt.show()

    # wniosek z eksperymentu
    reflection = "Pipeline B (augmented) had higher training loss because random augmentations effectively change the data in every epoch, making the images harder to memorize, effectively preventing overfitting."
    print("\nReflection:", reflection)

    # sprawdzenie
    assert transform_B is not None, "transform_B must be defined"
    assert len(losses_A) == EPOCHS_EXP, "Run all epochs for pipeline A"
    assert len(losses_B) == EPOCHS_EXP, "Run all epochs for pipeline B"
    assert len(reflection) > 20, "Write a proper reflection sentence"
    assert losses_A[-1] < losses_B[-1] + 0.5, "Unexpected result: augmented loss far below minimal — check your implementation"
    print("\n✓ Exercise 4 passed!")

if __name__ == '__main__':
    main()