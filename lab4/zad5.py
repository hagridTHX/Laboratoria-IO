import cv2
import numpy as np
import zlib
from scipy.fftpack import dct, idct

ZIGZAG_INDICES = np.array([
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
])

def dct2(array):
    return dct(dct(array, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(array):
    return idct(idct(array, axis=0, norm='ortho'), axis=1, norm='ortho')

_QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 48, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])

_QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])

def _scale(QF):
    if 1 <= QF < 50:
        scale = np.floor(5000 / QF)
    elif QF < 100:
        scale = 200 - 2 * QF
    else:
        raise ValueError('Quality Factor must be in the range [1..99]')
    return scale / 100.0

def QY(QF=85):
    return _QY * _scale(QF)

def QC(QF=85):
    return _QC * _scale(QF)

def encode_channel(channel, q_matrix):
    h, w = channel.shape
    blocks = []
    zz_data = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i+8, j:j+8].astype(np.float32) - 128.0
            block_dct = dct2(block)
            block_quant = np.round(block_dct / q_matrix)
            blocks.append(block_quant)
            zz_data.extend(block_quant.flatten()[ZIGZAG_INDICES])
    return blocks, np.array(zz_data, dtype=np.int16)

def decode_channel(blocks, h, w, q_matrix):
    channel = np.zeros((h, w), dtype=np.float32)
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block_dequant = blocks[idx] * q_matrix
            block_idct = idct2(block_dequant)
            channel[i:i+8, j:j+8] = block_idct + 128.0
            idx += 1
    return np.clip(channel, 0, 255).astype(np.uint8)

QUALITY_FACTOR = 50

image = cv2.imread('../example.jpg')
h_img, w_img = image.shape[:2]
h_img -= h_img % 16
w_img -= w_img % 16
image = image[:h_img, :w_img]

ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(ycrcb)

h_chroma, w_chroma = h_img // 2, w_img // 2
Cr_down = cv2.resize(Cr, (w_chroma, h_chroma))
Cb_down = cv2.resize(Cb, (w_chroma, h_chroma))

blocks_Y, zz_Y = encode_channel(Y, QY(QUALITY_FACTOR))
blocks_Cr, zz_Cr = encode_channel(Cr_down, QC(QUALITY_FACTOR))
blocks_Cb, zz_Cb = encode_channel(Cb_down, QC(QUALITY_FACTOR))

all_data = np.concatenate([zz_Y, zz_Cr, zz_Cb])
print(len(zlib.compress(all_data.tobytes())))

Y_rec = decode_channel(blocks_Y, h_img, w_img, QY(QUALITY_FACTOR))
Cr_rec = decode_channel(blocks_Cr, h_chroma, w_chroma, QC(QUALITY_FACTOR))
Cb_rec = decode_channel(blocks_Cb, h_chroma, w_chroma, QC(QUALITY_FACTOR))

Cr_up = cv2.resize(Cr_rec, (w_img, h_img))
Cb_up = cv2.resize(Cb_rec, (w_img, h_img))

final_image = cv2.cvtColor(cv2.merge([Y_rec, Cr_up, Cb_up]), cv2.COLOR_YCrCb2BGR)
cv2.imwrite('lab4-po-kompresji-zad5.png', final_image)