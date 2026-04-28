import cv2
import numpy as np

image_path = '../example.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Brak obrazka: {image_path}")
else:
    height, width = image.shape[:2]

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    Cr_down = cv2.resize(Cr, (width // 2, height // 2))
    Cb_down = cv2.resize(Cb, (width // 2, height // 2))

    size_orig = Y.nbytes + Cr.nbytes + Cb.nbytes
    size_down = Y.nbytes + Cr_down.nbytes + Cb_down.nbytes
    print(f"Rozmiar danych przed kompresja: {size_orig} bajtow")
    print(f"Rozmiar danych po zmniejszeniu chrominancji (4:2:0): {size_down} bajtow")

    Cr_up = cv2.resize(Cr_down, (width, height))
    Cb_up = cv2.resize(Cb_down, (width, height))

    reconstructed_ycrcb = cv2.merge([Y, Cr_up, Cb_up])
    final_image = cv2.cvtColor(reconstructed_ycrcb, cv2.COLOR_YCrCb2BGR)

    cv2.imshow('Oryginal', image)
    cv2.imshow('Po symulacji kompresji kolorow', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()