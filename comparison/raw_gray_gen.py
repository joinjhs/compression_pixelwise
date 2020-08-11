import cv2
import os

all_filenames = os.listdir('Comparison/data/div2k/')

count = 0
for filename in all_filenames:

    img = cv2.cvtColor(cv2.imread('Comparison/data/div2k/'+filename), cv2.COLOR_BGR2GRAY)
    cv2.imwrite('Comparison/data/div_gray/'+str(count)+'.png', img)
    count += 1