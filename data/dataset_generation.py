from typing import NamedTuple, Optional, Tuple, Generator

import numpy as np
import csv
from matplotlib import pyplot as plt
from skimage.draw import circle_perimeter_aa


def draw_circle(img: np.ndarray, row: int, col: int, radius: int) -> np.ndarray:
    """
    Draw a circle in a numpy array, inplace.
    The center of the circle is at (row, col) and the radius is given by radius.
    The array is assumed to be square.
    Any pixels outside the array are ignored.
    Circle is white (1) on black (0) background, and is anti-aliased.
    """
    rr, cc, val = circle_perimeter_aa(row, col, radius)
    valid = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
    img[rr[valid], cc[valid]] = val[valid]
    return img


class CircleParams(NamedTuple):
    row: int
    col: int
    radius: int

def noisy_circle(
        img_size: int, min_radius: float, max_radius: float, noise_level: float
) -> Tuple[np.ndarray, CircleParams]:
    """
    Draw a circle in a numpy array, with normal noise.
    """

    # Create an empty image
    img = np.zeros((img_size, img_size))

    radius = np.random.randint(min_radius, max_radius)

    # x,y coordinates of the center of the circle
    row, col = np.random.randint(img_size, size=2)

    # Draw the circle inplace
    draw_circle(img, row, col, radius)

    added_noise = np.random.normal(0.5, noise_level, img.shape)
    img += added_noise

    return img, CircleParams(row, col, radius)


def generate_dataset(max_noise=0.75, num_images=1000):
    """Generates train, test and validation datasets based on the input parameters. Saves each image to a .npy file, and creates
       a .csv for each set with ground truths and file paths."""
    
    # Open training .csv and load in ground truths and filepaths as images are created
    with open("train_set.csv", 'w', newline='') as outFile:
        header = ['PATH', 'ROW', 'COL', 'RAD']
        write(outFile, header)
        for i in range(int(num_images*0.6)):
            img, params = noisy_circle(200, 10, 50, np.random.rand() * max_noise)

            filepath = "datasets/trainset/" + str(i) + ".npy"
            np.save(filepath, img)
            write(outFile, [filepath, params[0], params[1], params[2]])
    
    # Open test .csv and load in ground truths and filepaths as images are created
    with open("test_set.csv", 'w', newline='') as outFile:
        header = ['PATH', 'ROW', 'COL', 'RAD']
        write(outFile, header)
        for i in range(int(num_images*0.2)):
            img, params = noisy_circle(200, 10, 50, np.random.rand() * max_noise)

            filepath = "datasets/testset/" + str(i) + ".npy"
            np.save(filepath, img)
            write(outFile, [filepath, params[0], params[1], params[2]])

    # Open validation .csv and load in ground truths and filepaths as images are created
    with open("validation_set.csv", 'w', newline='') as outFile:
        header = ['PATH', 'ROW', 'COL', 'RAD']
        write(outFile, header)
        for i in range(int(num_images*0.2)):
            img, params = noisy_circle(200, 10, 50, np.random.rand() * max_noise)

            filepath = "datasets/validationset/" + str(i) + ".npy"
            np.save(filepath, img)
            write(outFile, [filepath, params[0], params[1], params[2]])

# Original dataset generation function, deprecated in favour of the above function.
# def generate_dataset(max_noise=0.75, num_images=1000):
#     with open("train_set.csv", 'w', newline='') as outFile:
#         header = ['ARRAY', 'ROW', 'COL', 'RAD']
#         write(outFile, header)
#         for i in range(num_images):
#             img, params = noisy_circle(200, 10, 100, np.random.rand() * max_noise)
#             write(outFile, [img, params[0], params[1], params[2]])


def write(csvFile, row):
    writer = csv.writer(csvFile)
    writer.writerows([row])


if __name__ == '__main__':
    generate_dataset(.8, 2000)

