import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    plt.rcParams["figure.figsize"] = (24, 8)

    image_from_file = cv2.imread('../gaclaw-na-kuchni.jpg')
    if image_from_file is None:
        print("Nie znaleziono obrazu. Sprawdź ścieżkę.")
        return

    image_color = cv2.cvtColor(image_from_file, cv2.COLOR_BGR2RGB)

    k = 2

    simple_reduction = image_color.astype(np.float32)
    simple_reduction = np.round((k - 1) * simple_reduction / 255.0) * 255.0 / (k - 1)
    simple_reduction = np.clip(simple_reduction, 0, 255).astype(np.uint8)

    output = image_color.astype(np.float32)
    h, w, channels = output.shape

    for y in range(h):
        for x in range(w):
            old_pixel = output[y, x].copy()

            new_pixel = np.round((k - 1) * old_pixel / 255.0) * 255.0 / (k - 1)
            output[y, x] = new_pixel

            quant_error = old_pixel - new_pixel

            if x + 1 < w:
                output[y, x + 1] += quant_error * 7 / 16
            if x - 1 >= 0 and y + 1 < h:
                output[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < h:
                output[y + 1, x] += quant_error * 5 / 16
            if x + 1 < w and y + 1 < h:
                output[y + 1, x + 1] += quant_error * 1 / 16

    output = np.clip(output, 0, 255).astype(np.uint8)

    plt.subplot(1, 3, 1)
    plt.imshow(simple_reduction)
    plt.title(f'Sama redukcja barw (k={k})')

    plt.subplot(1, 3, 2)
    plt.imshow(output)
    plt.title(f'Dithering Floyda-Steinberga (k={k})')

    plt.subplot(1, 3, 3)
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        histr = cv2.calcHist([output], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)

    plt.xlim([-1, 256])
    plt.xlabel('Wartość składowej koloru')
    plt.ylabel('Liczba pikseli obrazu')
    plt.title('Histogram - Floyd-Steinberg')

    plt.show()


if __name__ == '__main__':
    main()