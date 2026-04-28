import cv2
import numpy as np

QY_base = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 48, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)


def process_jpeg_blocks(channel, q_matrix):
    h, w = channel.shape
    processed_channel = np.zeros((h, w), dtype=np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i + 8, j:j + 8].astype(np.float32) - 128.0

            block_dct = cv2.dct(block)
            block_quant = np.round(block_dct / q_matrix)

            block_dequant = block_quant * q_matrix
            block_idct = cv2.idct(block_dequant)

            processed_channel[i:i + 8, j:j + 8] = block_idct + 128.0

    return np.clip(processed_channel, 0, 255).astype(np.uint8)

image = cv2.imread('../example.jpg')

if image is None:
    print("Brak obrazka: ../example.jpg")
else:
    h, w = image.shape[:2]
    h, w = h - (h % 8), w - (w % 8)
    image_cropped = image[:h, :w]

    ycrcb = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    Quality_Factor = 30
    scale = (5000 / Quality_Factor) / 100.0 if Quality_Factor < 50 else (200 - 2 * Quality_Factor) / 100.0
    Q_Matrix = QY_base * scale
    Q_Matrix[Q_Matrix < 1] = 1

    Y_processed = process_jpeg_blocks(Y, Q_Matrix)

    reconstructed_ycrcb = cv2.merge([Y_processed, Cr, Cb])
    final_image = cv2.cvtColor(reconstructed_ycrcb, cv2.COLOR_YCrCb2BGR)

    cv2.imshow('Oryginal', image_cropped)
    cv2.imshow(f'Pelna kompresja JPEG (QF={Quality_Factor})', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()