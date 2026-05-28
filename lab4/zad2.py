import cv2
import numpy as np

width, height = 256, 256
image_rainbow = np.zeros((height, width, 3), dtype=np.uint8)

for x in range(width):
    t = (x / (width - 1)) * 7
    segment = int(t)

    if segment >= 7:
        segment = 6

    f = t - segment

    v_up = int(255 * f)
    v_down = int(255 * (1 - f))

    if segment == 0:
        r, g, b = 0, 0, v_up
    elif segment == 1:
        r, g, b = 0, v_up, 255
    elif segment == 2:
        r, g, b = 0, 255, v_down
    elif segment == 3:
        r, g, b = v_up, 255, 0
    elif segment == 4:
        r, g, b = 255, v_down, 0
    elif segment == 5:
        r, g, b = 255, 0, v_up
    elif segment == 6:
        r, g, b = 255, v_up, 255

    for y in range(height):
        image_rainbow[y, x] = [r, g, b]

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