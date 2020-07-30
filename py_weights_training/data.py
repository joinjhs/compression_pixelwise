import os
import sys
import cv2
import numpy as np
import tqdm
import random

from config import *

def read_dir(data_dir):

    dataset = []

    for dirpath, _, filenames in os.walk(data_dir):
        filenames = sorted(filenames)

        for img_filename in tqdm.tqdm(filenames):
            img_name = (data_dir + img_filename)
            
            dataset.append(img_name)
        
    return dataset

def create_data(args, dataset, channel=1):

    BATCH_SIZE = args.batch_size

    img_num = len(dataset)

    # Select 4 random images from dataset
    img_idx = random.sample(range(0,img_num),4)

    for i in range(4):
        cur_samples = patch_from_img(args, dataset[img_idx[i]])

        if i == 0:
            data_batch = cur_samples
        else:
            data_batch = np.concatenate([data_batch, cur_samples],axis=0)

    data_batch = data_batch.astype(np.float32)

    if channel==1:
        return np.expand_dims(data_batch[:,:,:,0],axis=3)
    elif channel==3:
        return data_batch
    else:
        print('Wrong number of channel')
        sys.exit(1)

def create_test_data(dataset, img_idx, channel=1):

    cur_img = cv2.cvtColor(cv2.imread(dataset[img_idx]), cv2.COLOR_BGR2RGB)
    cur_img_yuv = rgb2yuv(cur_img)

    if channel==1:
        return np.expand_dims(np.expand_dims(cur_img_yuv[:,:,0],axis=2),axis=0)
    elif channel==3:
        return np.expand_dims(cur_img_yuv,axis=0)
    else:
        print('Wrong number of channel')
        sys.exit(1)

def patch_from_img(args, img):
    
    BATCH_SIZE = args.batch_size
    PATCH_SIZE = args.patch_size

    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    height, width, _ = np.shape(img)

    # Random crop patches from images
    height_idx = [random.choice(range(0, height - PATCH_SIZE-1)) for i in range(int(BATCH_SIZE/4))]
    width_idx = [random.choice(range(0, width-PATCH_SIZE-1)) for i in range(int(BATCH_SIZE/4))]

    data_batch = []

    for i in range(int(BATCH_SIZE/4)):
        cur_patch_rgb = img[height_idx[i]:height_idx[i] + PATCH_SIZE, width_idx[i]:width_idx[i] + PATCH_SIZE, :]
        cur_patch_yuv = rgb2yuv(cur_patch_rgb)

        data_batch.append(cur_patch_yuv)

    data_batch = np.array(data_batch)

    return data_batch

def rgb2yuv(rgb_data):

    r, g, b = cv2.split(rgb_data)

    r = np.asarray(r, float)
    g = np.asarray(g, float)
    b = np.asarray(b, float)

    u = b - np.round((87 * r + 169 * g) / 256.0)
    v = r - g
    y = g + np.round((86 * v + 29 * u) / 256.0)

    y = np.expand_dims(y, axis=2)
    u = np.expand_dims(u, axis=2)
    v = np.expand_dims(v, axis=2)


    yuv_data = np.concatenate([y,u,v], axis=2)

    return yuv_data


if __name__ == "__main__":

    args = parse_args()
    data_dir = read_dir(args.data_dir + 'train/')
    input_data = create_data(args, data_dir)

    test_data_dir = read_dir(args.data_dir + 'test/')
    input_test_data = create_test_data(test_data_dir, 1, channel=1)

    a = 1