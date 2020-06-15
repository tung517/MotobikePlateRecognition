import glob
import os
import cv2
import random
import numpy as np
from data_augmentation import DataAugmentation
import constant


class ReadData:
    def __init__(self, folder_path):
        # Đường dẫn tới folder dữ liệu
        self.folder_path = folder_path
        # Mảng đường dẫn của ảnh đầu vào
        self.file_paths = []
        # Mảng đầu vào
        self.X = []
        # Mảng kết quả
        self.y = []

    # Đọc vào dữ liệu ảnh để training
    def read_data(self):
        # Danh sách các thư mục chứa ảnh
        list_dir = os.listdir(self.folder_path)
        for folder in list_dir:
            # Với từng folder đọc các ảnh trong folder đó
            files = self.folder_path + "/" + folder + "/*.*"
            # print(files)
            list_file = glob.glob(files)
            # print(list_file)
            for file in list_file:
                # Danh sách địa chỉ của ảnh
                self.file_paths.append(file)

        # Xáo trộn mảng file path
        # random.shuffle(self.file_paths)

        # data_augment = DataAugmentation()

        # Duyệt mảng để tạo dữ liệu
        for file_path in self.file_paths:
            name = file_path.split("/")[-1]
            if name[0] == "7":
                # Đọc dữ liệu cho mảng X
                img = cv2.imread(file_path, 0)
                ret, img = cv2.threshold(img, img.mean(), constant.MAX_VALUE, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img = cv2.resize(img, (constant.IMAGE_WIDTH_FOR_TRAIN, constant.IMAGE_HEIGHT_FOR_TRAIN))
                img = np.reshape(img, (constant.IMAGE_HEIGHT_FOR_TRAIN, constant.IMAGE_WIDTH_FOR_TRAIN, 1))
                cv2.imshow("img_origin - " + str(file_path), img)
                cv2.waitKey(0)
                self.X.append(img)
                # Đọc vào dữ liệu cho y
                split = str(file_path).split("/")
                self.y.append(split[-2])

        # Đưa 2 mảng thành dạng np.array
        self.X = np.array(self.X, dtype='float') / 255.0
        self.y = np.array(self.y)


read_data = ReadData("./training_folder")
read_data.read_data()
