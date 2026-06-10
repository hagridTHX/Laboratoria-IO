import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset


# blok rezydualny - dwie konwolucje plus skrot
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # jak zmienia sie liczba kanalow/rozmiar to skrot trzeba przepuscic przez 1x1
        if self.downsample is not None:
            residual = self.downsample(x)

        # dodanie wejscia i relu dopiero po dodaniu
        out += residual
        out = self.relu(out)

        return out


class TinyResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # test wymiarow
    block_same = ResidualBlock(64, 64)
    block_proj = ResidualBlock(64, 128, stride=2)

    x_in = torch.zeros(2, 64, 8, 8)
    out_same = block_same(x_in)
    out_proj = block_proj(x_in)
    print("same-channel block output:", tuple(out_same.shape))
    print("projected block output:   ", tuple(out_proj.shape))

    # trening na malym podzbiorze cifar
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)
    basic_transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    DATA_DIR = Path(os.path.expanduser("~/data"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    full_train = torchvision.datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=basic_transform)

    SUBSET_SIZE = 3_000
    EPOCHS = 3
    BATCH_SIZE = 128

    train_sub = Subset(full_train, range(SUBSET_SIZE))
    sub_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = TinyResNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    losses = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for imgs, labels in sub_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = crit(model(imgs), labels)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(imgs)
        avg = epoch_loss / SUBSET_SIZE
        losses.append(avg)
        print(f"Epoch {epoch}/{EPOCHS}  loss={avg:.4f}")

    plt.plot(range(1, EPOCHS + 1), losses, marker="o")
    plt.xlabel("Epoch");
    plt.ylabel("Loss");
    plt.title("TinyResNet training loss")
    plt.tight_layout();
    plt.show()

    # sprawdzenie
    assert tuple(out_same.shape) == (2, 64, 8, 8), f"same-channel shape wrong: {out_same.shape}"
    assert tuple(out_proj.shape) == (2, 128, 4, 4), f"projected shape wrong: {out_proj.shape}"
    assert losses[-1] < losses[0], "Loss should decrease across epochs"
    print("\n✓ Exercise 3 passed!")


if __name__ == '__main__':
    main()