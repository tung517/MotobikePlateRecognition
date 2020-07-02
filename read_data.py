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

        file_path_copy = self.file_paths.copy()
        file_path_size = len(file_path_copy)

        # Tăng thêm dữ liệu sủ dụng data augmentation
        for i in range(0, 3):
            self.file_paths.extend(file_path_copy)

        # Đánh dấu các ảnh cần xử lí
        # Từ 0 -> 1199 là ảnh gốc
        # Từ 1200 -> 2399 là ảnh xoay trái
        # Từ 2400 -> 3599 là ảnh xoay phải
        # Từ 3600 -> 4799 là ảnh làm mờ

        for i in range(file_path_size, len(self.file_paths)):
            if i >= 3 * file_path_size:
                self.file_paths[i] += 'b'
            elif i >= 2 * file_path_size:
                self.file_paths[i] += 'r'
            else:
                self.file_paths[i] += 'l'

        # Xáo trộn mảng file path
        random.shuffle(self.file_paths)

        data_augment = DataAugmentation()

        # Duyệt mảng để tạo dữ liệu
        for file_path in self.file_paths:
            # Nếu là ảnh gốc
            if file_path[-1] == 'g':
                # Đọc dữ liệu cho mảng X
                img = cv2.imread(file_path, 0)
                ret, img = cv2.threshold(img, img.mean(), constant.MAX_VALUE, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img = cv2.resize(img, (32, 32))
                img = np.reshape(img, (32, 32, 1))
                self.X.append(img)
                # Đọc vào dữ liệu cho y
                split = str(file_path).split("/")
                self.y.append(split[-2])
            # Nếu là ảnh muốn chỉnh sửa
            else:
                real_file_path = file_path[:len(file_path) - 1]
                img = None
                # Ảnh xoay trái
                if file_path[-1] == 'l':
                    img = data_augment.rotate(real_file_path, -5)
                # Ảnh xoay phải
                if file_path[-1] == 'r':
                    img = data_augment.rotate(real_file_path, 5)
                # Ảnh làm mờ
                if file_path[-1] == 'b':
                    img = data_augment.blur_img(real_file_path)
                # Đọc vào dữ liệu X
                self.X.append(img)
                # Đọc vào dữ liệu y
                split = str(real_file_path).split("/")
                self.y.append(split[-2])

        # Đưa 2 mảng thành dạng np.array
        self.X = np.array(self.X, dtype='float') / 255.0
        self.y = np.array(self.y)

