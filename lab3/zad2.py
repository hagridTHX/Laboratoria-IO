import cv2
import numpy as np

image = cv2.imread('../example.jpg')

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_float = image_rgb.astype(np.float32) / 255.0

matrix = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.689, 0.168],
    [0.272, 0.534, 0.131]
])

transformed_image = np.dot(image_float, matrix.T)
clipped_image = np.clip(transformed_image, 0.0, 1.0)

final_image_bgr = cv2.cvtColor((clipped_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

cv2.imshow('Oryginalne zdjecie', image)
cv2.imshow('przerobione zdjecie', final_image_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()