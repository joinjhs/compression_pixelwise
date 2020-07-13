import os
import sys
import cv2
import numpy as np

from config import *


def split_image(image):
    width = int(image.shape[0]/2)
    height = int(image.shape[1]/2)

    x1 = np.zeros((width, height))
    x2 = np.zeros((width, height))
    x3 = np.zeros((width, height))
    x4 = np.zeros((width, height))

    for i in range(width):
        for j in range(height):
            if i % 2 == 0 and j % 2 == 0:
                x1[int(i / 2), int(j / 2)] = image[i, j]

            elif i % 2 == 1 and j % 2 == 0:
                x2[int((i - 1) / 2), int(j / 2)] = image[i, j]

            elif i % 2 == 0 and j % 2 == 1:
                x3[int(i / 2), int((j - 1) / 2)] = image[i, j]

            else:
                x4[int((i - 1) / 2), int((j - 1) / 2)] = image[i, j]

    return x1, x2, x3, x4

def create_dataset(args, data_type):

    path = args.data_dir + data_type + "/"

    # Read in files
    filelist = os.listdir(path)
    filelist.sort()

    print('num_file = ' + str(len(filelist)))

    # Make dataset
    n = 0

    train_data_1 = np.zeros((len(filelist), 128, 128, 1))
    train_data_2 = np.zeros((len(filelist), 128, 128, 1))
    train_data_3 = np.zeros((len(filelist), 128, 128, 1))
    train_data_4 = np.zeros((len(filelist), 128, 128, 1))

    for idx in range(len(filelist)):
        filename = path + filelist[idx]
        print("Reading " + filename)
        sys.stdout.flush()

        img = cv2.resize(cv2.imread(filename), (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x1, x2, x3, x4 = split_image(img)

        train_data_1[n, :, :, 0] = x1
        train_data_2[n, :, :, 0] = x2
        train_data_3[n, :, :, 0] = x3
        train_data_4[n, :, :, 0] = x4

        n = n + 1

    filename_x1 = args.data_dir + 'npy/image_' + data_type + '1.npy'
    filename_x2 = args.data_dir + 'npy/image_' + data_type + '2.npy'
    filename_x3 = args.data_dir + 'npy/image_' + data_type + '3.npy'
    filename_x4 = args.data_dir + 'npy/image_' + data_type + '4.npy'

    print("total file number: {}".format(n))

    np.save(filename_x1, train_data_1)
    np.save(filename_x2, train_data_2)
    np.save(filename_x3, train_data_3)
    np.save(filename_x4, train_data_4)

if __name__ == "__main__":

    args = parse_args()
    create_dataset(args, "valid")