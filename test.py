import cv2
import glob
import os
import cv2
import random
import numpy as np
from data_augmentation import DataAugmentation
import constant

img = cv2.imread("../training_folder/A/A_2.jpg", 0)

ret, img = cv2.threshold(img, img.mean(), constant.MAX_VALUE, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img = cv2.resize(img, (32, 32))
img = np.reshape(img, (32, 32, 1))
cv2.imshow("origin", img)
cv2.imwrite("origin.jpg", img)

da = DataAugmentation()

r = da.rotate("../training_folder/A/A_2.jpg", 8)
cv2.imshow("right", r)
cv2.imwrite("right.jpg", r)

l = da.rotate("../training_folder/A/A_2.jpg", -8)
cv2.imshow("left", l)
cv2.imwrite("left.jpg", l)

b = da.blur_img("../training_folder/A/A_2.jpg")
cv2.imshow("blur", b)
cv2.imwrite("blur.jpg", b)
cv2.waitKey(0)
