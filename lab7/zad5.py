import torch


def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Liczy srednie IoU po klasach."""
    # macierz pomylek: wiersze = prawdziwa klasa, kolumny = przewidziana
    indices = targets * num_classes + preds
    conf = torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)

    ious = []

    for c in range(num_classes):
        tp = conf[c, c].item()

        # cala kolumna minus diagonala to falszywe pozytywy
        fp = conf[:, c].sum().item() - tp

        # caly wiersz minus diagonala to falszywe negatywy
        fn = conf[c, :].sum().item() - tp

        # pomijam klasy ktorych nie ma w ground truth, zeby nie dzielic przez zero
        if conf[c, :].sum().item() > 0:
            total = tp + fp + fn
            iou_c = tp / total if total > 0 else 0.0
            ious.append(iou_c)

    return sum(ious) / len(ious) if len(ious) > 0 else 0.0


def main():
    # maly test na recznie policzonym przykladzie
    t = torch.tensor([0, 0, 1, 1])
    p = torch.tensor([0, 0, 0, 1])
    miou_unit = compute_miou(p, t, num_classes=2)
    print(f"Unit test mIoU: {miou_unit:.4f}  (expected ~0.5833)")

    # pokazanie problemu niezbalansowanych klas
    NUM_PIXELS = 1_000
    targets_imb = torch.cat([torch.zeros(950, dtype=torch.long), torch.ones(50, dtype=torch.long)])
    preds_lazy = torch.zeros(NUM_PIXELS, dtype=torch.long)  # leniwy model - zawsze tlo

    pixel_acc = (preds_lazy == targets_imb).float().mean().item()
    miou_lazy = compute_miou(preds_lazy, targets_imb, num_classes=2)

    print(f"\nImbalanced scene (950 bg / 50 fg):")
    print(f"  Pixel accuracy (lazy all-background): {pixel_acc:.2%}")
    print(f"  mIoU          (lazy all-background): {miou_lazy:.4f}")
    print("\nNotice: pixel accuracy is misleadingly high (~95%) while mIoU exposes")
    print("        the failure to detect any foreground object (IoU_fg = 0).")

    # sprawdzenie
    assert abs(miou_unit - (
                2 / 3 + 1 / 2) / 2) < 1e-4, f"Unit test failed: expected {(2 / 3 + 1 / 2) / 2:.4f}, got {miou_unit:.4f}"
    assert pixel_acc > 0.90, "Pixel accuracy of lazy model should be ~95%"
    assert miou_lazy < 0.55, f"mIoU of lazy all-background model should be below 0.55, got {miou_lazy:.4f}"
    print("\n✓ Exercise 5 passed!")


if __name__ == '__main__':
    main()