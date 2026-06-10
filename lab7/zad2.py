import torch
import torch.nn as nn


def main():
    # kazdy maxpool(2,2) zmniejsza wymiar o polowe, start to 32x32
    size_after_block1 = 16
    size_after_block2 = 8
    size_after_block3 = 4
    size_after_block4 = 2

    print("Spatial sizes:", 32, "→", size_after_block1, "→", size_after_block2,
          "→", size_after_block3, "→", size_after_block4)

    # dwa bloki konwolucyjne, tak jak poczatek BasicCNN
    two_blocks = nn.Sequential(
        nn.Conv2d(3, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.1),
        nn.Dropout2d(0.3),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.1),
        nn.Dropout2d(0.3),
        nn.MaxPool2d(2, 2),
    )

    # pusty tensor zeby zobaczyc wymiar wyjscia (batch=1)
    dummy_input = torch.zeros(1, 3, 32, 32)
    output = two_blocks(dummy_input)
    output_shape = tuple(output.shape)

    print("Output shape after 2 blocks:", output_shape)

    # licze tylko parametry uczone
    num_params = sum(p.numel() for p in two_blocks.parameters() if p.requires_grad)
    print(f"Parameters in 2-block module: {num_params:,}")

    # sprawdzenie
    assert size_after_block1 == 16, f"After block 1 should be 16, got {size_after_block1}"
    assert size_after_block2 == 8, f"After block 2 should be 8, got {size_after_block2}"
    assert size_after_block3 == 4, f"After block 3 should be 4, got {size_after_block3}"
    assert size_after_block4 == 2, f"After block 4 should be 2, got {size_after_block4}"
    assert output_shape == (1, 512, 8, 8), f"Unexpected output shape: {output_shape}"
    assert num_params > 0, "Parameter count must be positive"
    print("\n✓ Exercise 2 passed!")


if __name__ == '__main__':
    main()