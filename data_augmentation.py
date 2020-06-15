import cv2
import math
import imutils
import numpy as np
import constant


class DataAugmentation:

    def rotate(self, img_path, angle):
        old_image = cv2.imread(img_path, 0)
        ret, old_image = cv2.threshold(old_image, old_image.mean(), constant.MAX_VALUE,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # kích thước ảnh cũ
        old_width = old_image.shape[1]
        old_height = old_image.shape[0]

        # Xoay ảnh
        new_image = imutils.rotate_bound(old_image, angle)

        # Kích thước ảnh mới
        new_width = new_image.shape[1]
        new_height = new_image.shape[0]

        # Tính toán cạnh bù do phép biến đổi
        # Cảnh nhỏ bên trên
        a = math.ceil(np.sin(math.fabs(angle) * np.pi / 180) * old_height)

        # Cạnh nhỏ bên dưới
        b = math.ceil(np.sin(math.fabs(angle) * np.pi / 180) * old_width)

        # Xử lí bên trái
        for i in range(a, -1, -1):
            for j in range(0, new_height):
                if new_image[j, i] < new_image[j, i + 1]:
                    new_image[j, i] = new_image[j, i + 1]

        # Xử lí bên trên
        for i in range(0, new_width):
            for j in range(b, -1, -1):
                if new_image[j, i] < new_image[j + 1, i]:
                    new_image[j, i] = new_image[j + 1, i]

        # Xử lí bên phải
        for i in range(new_width - a, new_width):
            for j in range(0, new_height):
                if new_image[j, i] < new_image[j, i - 1]:
                    new_image[j, i] = new_image[j, i - 1]

        # Xử lí bên dưới
        for i in range(0, new_width):
            for j in range(new_height - b, new_height):
                if new_image[j, i] < new_image[j - 1, i]:
                    new_image[j, i] = new_image[j - 1, i]

        rotate_img = cv2.resize(new_image, (constant.IMAGE_WIDTH_FOR_TRAIN, constant.IMAGE_HEIGHT_FOR_TRAIN))
        rotate_img = np.reshape(rotate_img, (constant.IMAGE_HEIGHT_FOR_TRAIN, constant.IMAGE_WIDTH_FOR_TRAIN, 1))
        return rotate_img

    def blur_img(self, img_path):
        img = cv2.imread(img_path, 0)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        ret, img = cv2.threshold(img, img.mean(), constant.MAX_VALUE,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.resize(img, (constant.IMAGE_WIDTH_FOR_TRAIN, constant.IMAGE_HEIGHT_FOR_TRAIN))
        img = np.reshape(img, (constant.IMAGE_HEIGHT_FOR_TRAIN, constant.IMAGE_WIDTH_FOR_TRAIN, 1))
        return img
