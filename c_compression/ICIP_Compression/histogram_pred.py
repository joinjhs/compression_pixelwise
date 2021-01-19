import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread('./data/img_div/0_2.png', 0)
img2 = cv2.imread('./data/img_div/0_2_pred.png', 0)

error = img-img2

_ = plt.hist(error, bins='auto')

plt.show()
