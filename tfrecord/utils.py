import cv2
import numpy as np

def save_image_data(image_num, partition, image):
    np.savetxt('../c_compression/ICIP_Compression/data/' + str(image_num) + '_' + str(partition) + '.txt', image[0, :, :, 0],
               fmt='%d')
    cv2.imwrite('../c_compression/ICIP_Compression/data/img/' + str(image_num) + '_' + str(partition) + '.png',
                np.squeeze(image[0, :, :, 0]))
    print('saved raw image(partition {}/4) of  {}'.format(str(partition), str(image_num)))

def save_pred_data(image_num, partition, image):
    np.savetxt('../c_compression/ICIP_Compression/data/' + str(image_num) + '_' + str(partition) + '_pred.txt', image[0, :, :, 0],
               fmt='%f')
    cv2.imwrite('../c_compression/data/img/' + str(image_num) + '_' + str(partition) + '_pred.png',
                np.squeeze(image[0, :, :, 0]))
    print('saved prediction(partition {}/4) of  {}'.format(str(partition), str(image_num)))

def save_ctx_data(image_num, partition, image):
    np.savetxt('../c_compression/ICIP_Compression/data/' + str(image_num) + '_' + str(partition) + '_ctx.txt', image[0, :, :, 0],
               fmt='%f')
    cv2.imwrite('../c_compression/ICIP_Compression/data/img/' + str(image_num) + '_' + str(partition) + '_ctx.png',
                np.squeeze(image[0, :, :, 0]))
    print('saved context(partition {}/4) of  {}'.format(str(partition), str(image_num)))