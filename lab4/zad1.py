import os
import numpy as np

width, height = 100, 100
kolor_czerwony = [255, 0, 0]

image = np.full((height, width, 3), kolor_czerwony, dtype=np.uint8)

ppm_ascii_header = f'P3\n{width} {height}\n255\n'
with open('lab4_zad1_ascii.ppm', 'w') as fh:
    fh.write(ppm_ascii_header)
    image.tofile(fh, sep=' ')
    fh.write('\n')

ppm_binary_header = f'P6\n{width} {height}\n255\n'
with open('lab4_zad1_binary.ppm', 'wb') as fh:
    fh.write(bytearray(ppm_binary_header, 'ascii'))
    image.tofile(fh)

rozmiar_p3 = os.path.getsize("lab4_zad1_ascii.ppm")
rozmiar_p6 = os.path.getsize("lab4_zad1_binary.ppm")

print(f"Rozmiar pliku P3 (tekstowy): {rozmiar_p3} bajtów")
print(f"Rozmiar pliku P6 (binarny): {rozmiar_p6} bajtów")