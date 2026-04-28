import cv2
import numpy as np
import zlib

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

def process_channel_zad4(channel):
    h, w = channel.shape
    blocks = []
    zz_data = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i+8, j:j+8].astype(np.float32) - 128.0
            blocks.append(block)
            zz_data.extend(block.flatten()[ZIGZAG_INDICES])
    return blocks, np.array(zz_data, dtype=np.int16)

blocks_Y, zz_Y = process_channel_zad4(Y)
blocks_Cr, zz_Cr = process_channel_zad4(Cr_down)
blocks_Cb, zz_Cb = process_channel_zad4(Cb_down)

all_data = np.concatenate([zz_Y, zz_Cr, zz_Cb])
print(len(zlib.compress(all_data.tobytes())))

def reconstruct_channel_zad4(blocks, h, w):
    channel = np.zeros((h, w), dtype=np.float32)
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            channel[i:i+8, j:j+8] = blocks[idx] + 128.0
            idx += 1
    return np.clip(channel, 0, 255).astype(np.uint8)

Y_rec = reconstruct_channel_zad4(blocks_Y, h_img, w_img)
Cr_rec = reconstruct_channel_zad4(blocks_Cr, h_chroma, w_chroma)
Cb_rec = reconstruct_channel_zad4(blocks_Cb, h_chroma, w_chroma)

Cr_up = cv2.resize(Cr_rec, (w_img, h_img))
Cb_up = cv2.resize(Cb_rec, (w_img, h_img))

final_image = cv2.cvtColor(cv2.merge([Y_rec, Cr_up, Cb_up]), cv2.COLOR_YCrCb2BGR)
cv2.imwrite('lab4-po-kompresji-zad4.png', final_image)