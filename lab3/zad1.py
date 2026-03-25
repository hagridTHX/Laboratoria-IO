import cv2
import numpy as np

image = cv2.imread('../example.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Brak obrazka.")
else:
    matrix = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])

    filtered_image = cv2.filter2D(image, -1, matrix)

    cv2.imshow('zdjecie w skali szarosci', image)
    cv2.imshow('Detekcja krawedzi', filtered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()