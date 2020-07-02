import cv2
import numpy as np
import constant
import math
import matplotlib.pyplot as plt


class CharacterSegmentation:
    def __init__(self, img):
        # Ảnh biển số
        self.img_copy = None
        self.img = img
        # Ảnh nhị phân
        self.thresh = None
        # Mảng các biên đóng
        self.contours = None
        # Mảng các ảnh chữ cái
        self.character = []

    # Thay đổi kích thước ảnh
    def resize_image(self):
        self.img = cv2.resize(self.img, (constant.IMAGE_SIZE_FOR_DETECT, constant.IMAGE_SIZE_FOR_DETECT))
        # self.img_copy = self.img.copy()
        # cv2.imshow("resize", self.img)
        # cv2.waitKey(0)

    # Làm rõ ảnh cục bộ
    def clahe_image(self, clip_limit, tile_grid_size):
        cl = cv2.createCLAHE(clip_limit, tile_grid_size)
        self.img = cl.apply(self.img)
        # cv2.imshow("clahe", self.img)
        # cv2.waitKey(0)

    # Làm rõ ảnh toàn cục
    def equal_hist(self):
        self.img = cv2.equalizeHist(self.img)

    # Làm mờ ảnh
    def blur(self):
        self.img = cv2.GaussianBlur(self.img, (5, 5), 0)
        # cv2.imshow("blur", self.img)
        # cv2.waitKey(0)

    # Làm mờ bảo toàn cạnh
    def bilateral_blur(self):
        self.img = cv2.bilateralFilter(self.img, 5, 11, 11)

    # Làm mờ bụi, bẩn
    def median_blur(self):
        self.img = cv2.medianBlur(self.img, 5)

    # Lấy ngưỡng
    def thresh_image(self):
        ret, self.thresh = cv2.threshold(self.img, self.img.mean(), constant.MAX_VALUE,
                                         cv2.THRESH_BINARY
                                         + cv2.THRESH_OTSU)
        # print(self.img.mean())
        # thresh = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)
        # cv2.imshow("image", self.img)
        # cv2.imshow("thresh", self.thresh)
        # cv2.imshow("adaptive thresh", thresh)
        # cv2.waitKey(0)

    # Tìm biên đóng
    def find_contour(self):
        self.contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = sorted(self.contours, key=cv2.contourArea, reverse=True)[:20]

    # Lấy vùng chứa chữ cái
    def get_character_area(self):

        # new_img_1 = self.img.copy()
        # new_img_2 = self.img.copy()

        character_area = []
        for con in self.contours:
            minRect = cv2.minAreaRect(con)
            check_size = self.remove_wrond_position_area(minRect)
            if check_size:
                # x, y = minRect[0][0], minRect[0][1]
                w, h = 0, 0
                angle = minRect[-1]

                if minRect[1][0] < minRect[1][1]:
                    w = minRect[1][0]
                    h = minRect[1][1]
                else:
                    h = minRect[1][0]
                    w = minRect[1][1]

                box = cv2.boxPoints(minRect)
                box = np.int0(box)

                # cv2.drawContours(new_img_2, [box], -1, (0, 0, 0), 1, cv2.LINE_AA)

                # Điều kiện về diện tích và tỉ lệ cạnh
                if constant.MIN_CHARACTER_AREA_1 <= w * h <= constant.MAX_CHARACTER_AREA \
                        and constant.CHARACTER_MIN_RATIO <= h / w <= constant.CHARACTER_MAX_RATIO:
                    # Điều kiện về góc
                    # cv2.drawContours(new_img_1, [box], -1, (0, 0, 0), 1, cv2.LINE_AA)
                    if (-angle) <= 15 or 90 + angle <= 15:
                        character_area.append(con)
            else:
                continue
        # cv2.imshow("new_image_1", new_img_1)
        # cv2.imshow("new_image_2", new_img_2)
        # cv2.waitKey(0)
        return character_area

    # Điều kiện lọc vùng chồng nhau
    def remove_overlap_area(self, character_area):
        remove_area = []
        new_image_3 = self.img.copy()
        for i in range(len(character_area)):
            x, y, w, h = cv2.boundingRect(character_area[i])
            temp = character_area.copy()
            del temp[i]
            for t in temp:
                xt, yt, wt, ht = cv2.boundingRect(t)
                if (xt > x and yt > y) and ((xt + wt) < (x + w) and (yt + ht) < (y + h)):
                    remove_area.append(t)

        character_area = [x for x in character_area if x not in remove_area]

        # Lấy mảng các boundingRect
        character_bounding_rect = []
        for c in character_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(new_image_3, (x, y), (x + w, y + h), (0, 0, 0), 1)
            character_bounding_rect.append((x, y, w, h))
        # cv2.imshow("new_image_3", new_image_3)
        # cv2.waitKey(0)
        return character_bounding_rect

    # Điều kiện lọc vùng có kích thước không thỏa mãn
    def remove_wrond_position_area(self, min_rect):
        # 4 đỉnh của rotate rectangle
        box = cv2.boxPoints(min_rect)
        # Sắp xếp theo tạo độ x
        box_x = sorted(box, key=lambda x: x[0])
        # print("Box_x = " + str(box_x))
        # Sắp xếp theo tọa độ y
        box_y = sorted(box, key=lambda y: y[1])
        # print("Box_y = " + str(box_y))
        if box_x[0][0] < 0 or box_x[3][0] > 600:
            return False
        if box_y[0][1] < 0 or box_y[3][1] > 600:
            return False
        width = math.fabs(box_y[0][0] - box_y[1][0])
        height = math.fabs(box_x[0][1] - box_x[1][1])

        if width > height:
            return False
        return True

    # Sắp xếp các chữ cái theo thứ tự
    def sort_character_area(self, character_area):
        # Sắp xếp mảng các bounding rect theo thứ tự trái - phải, trên - dưới
        # Sắp xếp thep y
        crt_sorted_Y = sorted(character_area, key=lambda y: y[1])

        # Chỉ số đê tách mảng
        index = 0

        # Tách mảng theo điều kiện của y
        for i in range(len(crt_sorted_Y) - 1):
            yi = crt_sorted_Y[i][1]
            yj = crt_sorted_Y[i + 1][1]
            if math.fabs(yj - yi) > 100:
                index = i
                break

        row_1 = crt_sorted_Y[:index + 1]
        row_2 = crt_sorted_Y[index + 1:]

        # Sắp xếp 2 mảng theo x
        row_1 = sorted(row_1, key=lambda x: x[0])
        row_2 = sorted(row_2, key=lambda x: x[0])

        # Mảng số mới
        character_bounding_rect = []
        character_bounding_rect.extend(row_1)
        character_bounding_rect.extend(row_2)

        return character_bounding_rect

    # Lấy các hình ảnh chữ cái
    def get_character_image(self, character_bounding_rect):
        q = 0
        for crt in character_bounding_rect:
            x = crt[0]
            y = crt[1]
            w = crt[2]
            h = crt[3]
            character_img = self.thresh[y:y + h, x:x + w]

            character_img = cv2.resize(character_img, (constant.IMAGE_WIDTH_FOR_TRAIN, constant.IMAGE_HEIGHT_FOR_TRAIN))
            character_img = np.reshape(character_img,
                                       (constant.IMAGE_HEIGHT_FOR_TRAIN, constant.IMAGE_WIDTH_FOR_TRAIN, 1))
            # print("test " + str(q) + " " + str(w * h))
            # cv2.imshow("test" + str(q), character_img)
            # cv2.waitKey(0)
            self.character.append(character_img)
            q += 1

    # Tìm các kí tự
    def get_character(self, plate_property, num_c):
        self.resize_image()
        # Biển rõ
        if plate_property == 1:
            self.bilateral_blur()
            self.clahe_image(2, (8, 8))
            self.blur()
            self.thresh_image()
        # Biển mờ
        elif plate_property == 2:
            # cv2.imshow("plate", self.img)
            # self.equal_hist()
            # cv2.imshow("e_h", self.img)
            self.clahe_image(10, (5, 5))
            # cv2.imshow("cl", self.img)
            self.bilateral_blur()
            # cv2.imshow("b_b", self.img)
            self.thresh_image()
            cv2.waitKey(0)
        # Biển bụi, bẩn
        else:
            self.median_blur()
            # cv2.imshow("m_b", self.img)
            self.clahe_image(15, (8, 8))
            # cv2.imshow("e_h", self.img)
            # self.bilateral_blur()
            # cv2.imshow("e_h", self.img)
            self.thresh_image()

        # self.blur()
        # self.thresh_image(self.img.mean())
        self.find_contour()
        character_area = self.get_character_area()
        character_bounding_rect = self.remove_overlap_area(character_area)
        character_bounding_rect = self.sort_character_area(character_bounding_rect)
        self.get_character_image(character_bounding_rect)
