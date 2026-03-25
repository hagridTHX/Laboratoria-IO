import cv2
import numpy as np

image_bgr = cv2.imread('../example.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
height, width, _ = image_bgr.shape

image_float = image_rgb.astype(np.float32)
matrix = np.array([
    [ 0.229,  0.587,  0.114],
    [ 0.500, -0.418, -0.082],
    [-0.168, -0.331,  0.500]
])
offset = np.array([0, 128, 128])

transformed_image = np.dot(image_float, matrix.T) + offset
image_ycrcb = np.clip(transformed_image, 0, 255).astype(np.uint8)

Y = image_ycrcb[:, :, 0]
Cr = image_ycrcb[:, :, 1]
Cb = image_ycrcb[:, :, 2]

Cr_down = cv2.resize(Cr, (width // 2, height // 2))
Cb_down = cv2.resize(Cb, (width // 2, height // 2))

Cr_up = cv2.resize(Cr_down, (width, height))
Cb_up = cv2.resize(Cb_down, (width, height))

merged_ycrcb = cv2.merge([Y, Cr_up, Cb_up])

image_rgb_reconstructed = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2RGB)
final_bgr = cv2.cvtColor(image_rgb_reconstructed, cv2.COLOR_RGB2BGR)

cv2.imshow('Oryginalne zdjecie', image_bgr)
cv2.imshow('Obraz po transmisji', final_bgr)
cv2.imshow('Skladowa Y (Oryginalna)', Y)
cv2.imshow('Skladowa Cb (Po upsamplingu)', Cb_up)
cv2.imshow('Skladowa Cr (Po upsamplingu)', Cr_up)

cv2.waitKey(0)
cv2.destroyAllWindows()