import cv2
import numpy as np
import struct
import zlib

width, height = 256, 256
image = np.zeros((height, width, 3), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        image[y, x] = [int((x / 255) * 255), int((y / 255) * 255), 128]

png_file_signature = b'\x89PNG\r\n\x1a\n'

header_id = b'IHDR'
header_content = struct.pack('!IIBBBBB', width, height, 8, 2, 0, 0, 0)
header_size = struct.pack('!I', len(header_content))
header_crc = struct.pack('!I', zlib.crc32(header_id + header_content) & 0xFFFFFFFF)
png_file_header = header_size + header_id + header_content + header_crc

data_id = b'IDAT'
data_content = zlib.compress(b''.join([b'\x00' + bytes(row) for row in image]))
data_size = struct.pack('!I', len(data_content))
data_crc = struct.pack('!I', zlib.crc32(data_id + data_content) & 0xFFFFFFFF)
png_file_data = data_size + data_id + data_content + data_crc

end_id = b'IEND'
end_content = b''
end_size = struct.pack('!I', len(end_content))
end_crc = struct.pack('!I', zlib.crc32(end_id + end_content) & 0xFFFFFFFF)
png_file_end = end_size + end_id + end_content + end_crc

with open('lab4_zad3.png', 'wb') as fh:
    fh.write(png_file_signature)
    fh.write(png_file_header)
    fh.write(png_file_data)
    fh.write(png_file_end)


loaded_png = cv2.imread('lab4_zad3.png')
cv2.imshow('Recznie stworzony plik PNG', loaded_png)
cv2.waitKey(0)
cv2.destroyAllWindows()