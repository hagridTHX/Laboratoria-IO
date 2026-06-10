import os
import torch
import torchvision
import torchvision.transforms as T
from pathlib import Path
from torch.utils.data import DataLoader, Subset

def main():
    # gpu jesli jest, w razie czego cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)
    if str(device) == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))

    # srednia i odchylenie dla cifar-100
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

    basic_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    DATA_DIR = Path(os.path.expanduser("~/data"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    full_train = torchvision.datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=basic_transform)
    test_ds    = torchvision.datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=basic_transform)

    # pierwsze 45k na trening, ostatnie 5k na walidacje
    train_ds = Subset(full_train, range(0, 45000))
    val_ds   = Subset(full_train, range(45000, 50000))

    print(f"train_ds size : {len(train_ds)}")
    print(f"val_ds   size : {len(val_ds)}")
    print(f"test_ds  size : {len(test_ds)}")

    # podglad jednego batcha
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    imgs, labels = next(iter(loader))
    print(f"Batch shape : {imgs.shape}   dtype: {imgs.dtype}")
    print(f"Label range : {labels.min().item()} – {labels.max().item()}")

    # sprawdzenie wymiarow
    assert str(device) in ("cuda", "cpu"), "device must be 'cuda' or 'cpu'"
    assert len(train_ds) == 45_000, f"Expected 45 000 training samples, got {len(train_ds)}"
    assert len(val_ds)   ==  5_000, f"Expected 5 000 validation samples, got {len(val_ds)}"
    assert len(test_ds)  == 10_000, f"Expected 10 000 test samples, got {len(test_ds)}"
    assert imgs.shape == (64, 3, 32, 32), f"Unexpected batch shape: {imgs.shape}"
    print("\n✓ Exercise 1 passed!")

if __name__ == '__main__':
    main()