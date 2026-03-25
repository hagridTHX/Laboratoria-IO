import cv2
import numpy as np

def calculate_mse(imageA, imageB):
    imgA_float = imageA.astype(np.float32)
    imgB_float = imageB.astype(np.float32)

    err = np.mean((imgA_float - imgB_float) ** 2)
    return err

original_bgr = cv2.imread('../example.jpg')

ycrcb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(ycrcb)
height, width = Y.shape

Cr_down = cv2.resize(Cr, (width // 2, height // 2))
Cb_down = cv2.resize(Cb, (width // 2, height // 2))
Cr_up = cv2.resize(Cr_down, (width, height))
Cb_up = cv2.resize(Cb_down, (width, height))

reconstructed_ycrcb = cv2.merge([Y, Cr_up, Cb_up])
reconstructed_bgr = cv2.cvtColor(reconstructed_ycrcb, cv2.COLOR_YCrCb2BGR)

mse_value = calculate_mse(original_bgr, reconstructed_bgr)
print(f"MSE: {mse_value:.4f}")