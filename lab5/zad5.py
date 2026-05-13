import numpy as np
import cv2
from matplotlib import pyplot as plt


def draw_point(image, x, y, color=(255, 255, 255)):
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        image[image.shape[0] - 1 - y, x, :] = color


def draw_line(image, x1, y1, x2, y2, color1=(255, 255, 255), color2=(255, 255, 255)):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    xi = 1 if x1 < x2 else -1
    yi = 1 if y1 < y2 else -1

    total_steps = max(dx, dy)
    if total_steps == 0:
        draw_point(image, x1, y1, color1)
        return

    def get_color(step):
        t = step / total_steps
        return (
            int(color1[0] + t * (color2[0] - color1[0])),
            int(color1[1] + t * (color2[1] - color1[1])),
            int(color1[2] + t * (color2[2] - color1[2]))
        )

    x, y = x1, y1
    step = 0
    draw_point(image, x, y, get_color(step))

    if dx > dy:
        d = 2 * dy - dx
        for _ in range(dx):
            if d >= 0:
                y += yi
                d -= 2 * dx
            x += xi
            d += 2 * dy
            step += 1
            draw_point(image, x, y, get_color(step))
    else:
        d = 2 * dx - dy
        for _ in range(dy):
            if d >= 0:
                x += xi
                d -= 2 * dy
            y += yi
            d += 2 * dx
            step += 1
            draw_point(image, x, y, get_color(step))


def edge_func(v0, v1, p):
    return (v1[0] - v0[0]) * (p[1] - v0[1]) - (v1[1] - v0[1]) * (p[0] - v0[0])


def draw_triangle(image, a, b, c, cA, cB, cC):
    xmin = max(0, min(a[0], b[0], c[0]))
    xmax = min(image.shape[1] - 1, max(a[0], b[0], c[0]))
    ymin = max(0, min(a[1], b[1], c[1]))
    ymax = min(image.shape[0] - 1, max(a[1], b[1], c[1]))

    area_total = abs(edge_func(a, b, c))
    if area_total == 0: return

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            p = (x, y)
            w0 = edge_func(b, c, p)
            w1 = edge_func(c, a, p)
            w2 = edge_func(a, b, p)

            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                l0 = abs(w0) / area_total
                l1 = abs(w1) / area_total
                l2 = abs(w2) / area_total

                color = (
                    int(l0 * cA[0] + l1 * cB[0] + l2 * cC[0]),
                    int(l0 * cA[1] + l1 * cB[1] + l2 * cC[1]),
                    int(l0 * cA[2] + l1 * cB[2] + l2 * cC[2])
                )
                draw_point(image, x, y, color)


def main():
    plt.rcParams["figure.figsize"] = (18, 10)

    width = 80
    height = 60
    scale = 2

    image_hr = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)

    draw_line(
        image_hr,
        5 * scale, 5 * scale,
        5 * scale, 55 * scale,
        color1=(0, 255, 0), color2=(0, 0, 255)
    )

    draw_triangle(
        image_hr,
        (20 * scale, 10 * scale),
        (70 * scale, 10 * scale),
        (45 * scale, 55 * scale),
        cA=(255, 0, 0), cB=(0, 255, 0), cC=(0, 0, 255)
    )

    image = cv2.resize(image_hr, (width, height), interpolation=cv2.INTER_AREA)

    plt.imshow(image)
    plt.title("SSAA x2 - Wygładzanie krawędzi")
    plt.show()


if __name__ == '__main__':
    main()