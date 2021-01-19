import cv2
import numpy as np
from config import parse_args
channel_num = parse_args().channel_num

def save_image_data(image_num, partition, image):
    if channel_num == 1:
        np.savetxt('data/' + str(image_num) + '_' + str(partition) + '.txt', image[0, :, :, 0],
               fmt='%d')
        cv2.imwrite('data/img/' + str(image_num) + '_' + str(partition) + '.png',
                np.squeeze(image[0, :, :, 0]))
    else:
        np.savetxt('../c_compression/ICIP_Compression/data/' + str(image_num) + '_' + str(partition) + '_y.txt',
                   image[0, :, :, 0], fmt='%d')
        np.savetxt('../c_compression/ICIP_Compression/data/' + str(image_num) + '_' + str(partition) + '_u.txt',
                   image[0, :, :, 1], fmt='%d')
        np.savetxt('../c_compression/ICIP_Compression/data/' + str(image_num) + '_' + str(partition) + '_v.txt',
                   image[0, :, :, 2], fmt='%d')
        cv2.imwrite('../c_compression/ICIP_Compression/data/img/' + str(image_num) + '_' + str(partition) + '_y.png',
                    np.squeeze(image[0, :, :, 0]))
        cv2.imwrite('../c_compression/ICIP_Compression/data/img/' + str(image_num) + '_' + str(partition) + '_u.png',
                    np.squeeze(image[0, :, :, 1]))
        cv2.imwrite('../c_compression/ICIP_Compression/data/img/' + str(image_num) + '_' + str(partition) + '_v.png',
                    np.squeeze(image[0, :, :, 2]))
    print('saved raw image(partition {}) of  {}'.format(str(partition), str(image_num)))

def save_pred_data(image_num, partition, image):
    np.savetxt('../c_compression/ICIP_Compression/data/' + str(image_num) + '_' + str(partition) + '_pred.txt', image[0, :, :, 0], fmt='%f')
    cv2.imwrite('../c_compression/ICIP_Compression/data/img/' + str(image_num) + '_' + str(partition) + '_pred.png',
            np.squeeze(image[0, :, :, 0]))
    print('saved prediction(partition {}) of  {}'.format(str(partition), str(image_num)))

def save_ctx_data(image_num, partition, image):

    np.savetxt('../c_compression/ICIP_Compression/data/' + str(image_num) + '_' + str(partition) + '_ctx.txt', image[0, :, :, 0], fmt='%f')
    cv2.imwrite('../c_compression/ICIP_Compression/data/img/' + str(image_num) + '_' + str(partition) + '_ctx.png',
            np.squeeze(image[0, :, :, 0]))

    print('saved context(partition {}) of  {}'.format(str(partition), str(image_num)))