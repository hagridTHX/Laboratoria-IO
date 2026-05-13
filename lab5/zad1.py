import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    plt.rcParams["figure.figsize"] = (18, 10)

    image_from_file = cv2.imread('../gaclaw-na-kuchni.jpg')

    image_gray = cv2.cvtColor(image_from_file, cv2.COLOR_BGR2GRAY)

    output = image_gray.astype(np.float32)
    h, w = output.shape

    for y in range(h):
        for x in range(w):
            old_pixel = output[y, x]
            new_pixel = np.round(old_pixel / 255.0) * 255.0
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

    plt.subplot(1, 2, 1)
    plt.imshow(output, cmap='gray')
    plt.title('Dithering - Floyd-Steinberg skala szarości')

    plt.subplot(1, 2, 2)
    histr = cv2.calcHist([output], [0], None, [256], [0, 256])
    plt.plot(histr)
    plt.xlim([-1, 256])
    plt.xlabel('Wartość składowej koloru')
    plt.ylabel('Liczba pikseli obrazu')
    plt.title('Histogram')

    plt.show()


if __name__ == '__main__':
    main()