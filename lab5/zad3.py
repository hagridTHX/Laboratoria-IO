import numpy as np
from matplotlib import pyplot as plt


def draw_point(image, x, y, color=(255, 255, 255)):
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        image[image.shape[0] - 1 - y, x, :] = color


def draw_line(image, x1, y1, x2, y2, color=(255, 255, 255)):
    # Algorytm Bresenhama
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    xi = 1 if x1 < x2 else -1
    yi = 1 if y1 < y2 else -1

    x, y = x1, y1
    draw_point(image, x, y, color)

    if dx > dy:
        d = 2 * dy - dx
        for _ in range(dx):
            if d >= 0:
                y += yi
                d -= 2 * dx
            x += xi
            d += 2 * dy
            draw_point(image, x, y, color)
    else:
        d = 2 * dx - dy
        for _ in range(dy):
            if d >= 0:
                x += xi
                d -= 2 * dy
            y += yi
            d += 2 * dx
            draw_point(image, x, y, color)


def edge_func(v0, v1, p):
    return (v1[0] - v0[0]) * (p[1] - v0[1]) - (v1[1] - v0[1]) * (p[0] - v0[0])


def draw_triangle(image, a, b, c, color=(255, 255, 255)):
    xmin = max(0, min(a[0], b[0], c[0]))
    xmax = min(image.shape[1] - 1, max(a[0], b[0], c[0]))
    ymin = max(0, min(a[1], b[1], c[1]))
    ymax = min(image.shape[0] - 1, max(a[1], b[1], c[1]))

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            p = (x, y)
            w0 = edge_func(a, b, p)
            w1 = edge_func(b, c, p)
            w2 = edge_func(c, a, p)

            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                draw_point(image, x, y, color)


def main():
    plt.rcParams["figure.figsize"] = (18, 10)

    width = 80
    height = 60
    image = np.zeros((height, width, 3), dtype=np.uint8)

    draw_line(image, 5, 5, 25, 15, color=(255, 255, 255))
    draw_triangle(image, (30, 5), (70, 20), (50, 55), color=(255, 0, 0))

    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    main()