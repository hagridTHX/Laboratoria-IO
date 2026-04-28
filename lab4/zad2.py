import cv2
import numpy as np

width, height = 256, 256
image_rainbow = np.zeros((height, width, 3), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        image_rainbow[y, x] = [int((x / 255) * 255), int((y / 255) * 255), 128]

ppm_binary_header = f'P6\n{width} {height}\n255\n'
with open('lab4_zad2_rainbow.ppm', 'wb') as fh:
    fh.write(bytearray(ppm_binary_header, 'ascii'))
    image_rainbow.tofile(fh)

print("Pomyślnie zapisano plik lab4_zad2_rainbow.ppm")

loaded_image = cv2.imread('lab4_zad2_rainbow.ppm')
if loaded_image is not None:
    cv2.imshow('Wygenerowany PPM (Tecza)', loaded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()