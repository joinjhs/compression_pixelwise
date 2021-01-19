import cv2
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter

'''
for i in range(1):
    img = cv2.imread('./data/img/0_2_pred.png', 0)
    plt.hist(img.ravel(), 256, [0,256])
    plt.show()

'''
img = cv2.imread('./data/img_div/0_2.png', 0)
w = img.shape[0]
h = img.shape[1]

#mask = np.zeros(img.shape[:2], np.uint8)
#mask[5:(w-5), 5:(h-5)] = 255

mask = np.ones(img.shape[:2], np.uint8)*255
mask[5:(w-5), 5:(h-5)] = 0

hist_1 = cv2.calcHist([img],[0],mask,[256],[0,256])
#hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

img2 = cv2.imread('./data/img_div/0_2_pred.png', 0)
pred_error = img2-img
hist_2 = cv2.calcHist([img2],[0],mask,[256],[0,256])

cvt_img = np.zeros(img.shape[:2], np.int)
cvt_img2 = np.zeros([w*h], np.int)

for i in range(w):
    for j in range(h):
        cvt_img[i,j] = int(img2[i,j])-int(img[i,j])
        cvt_img2[i*w+j] = cvt_img[i,j]

counter = Counter(cvt_img2)
print(counter.most_common(n=5))
sns.distplot(cvt_img, hist=False)
#plt.hist(cvt_img, bins=512)


#plt.plot(hist_1), plt.plot(hist_2)
#plt.plot(hist_pred)
plt.show()

