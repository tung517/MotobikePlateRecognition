import cv2
import numpy as np
from character_segmentation import CharacterSegmentation
import constant


class Detector:
    def __init__(self, img):
        self.img = img
        self.gray = None
        self.blur = None
        self.hist = None
        self.binary = None
        self.plate = None

    # Resize
    def resize_image(self):
        self.img = cv2.resize(self.img, (constant.IMAGE_SIZE_FOR_DETECT, constant.IMAGE_SIZE_FOR_DETECT))

    # Chuyển ảnh sang xám
    def grayscale(self):
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    # Làm mờ ảnh
    def blur_image(self, plate_property):
        if plate_property == constant.PLATE_NORMAL:
            self.blur = cv2.bilateralFilter(self.gray, 5, 11, 11)
        elif plate_property == constant.PLATE_DIRTY:
            self.blur = cv2.bilateralFilter(self.gray, 11, 17, 17)
        else:
            self.blur = cv2.bilateralFilter(self.gray, 5, 11, 11)

    # Tăng độ tương phản
    def increase_contrast(self):
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        clahe_1 = cv2.createCLAHE(clipLimit=3, tileGridSize=(5, 5))
        clahe_2 = cv2.createCLAHE(clipLimit=1, tileGridSize=(3, 3))
        clahe_3 = cv2.createCLAHE(clipLimit=10, tileGridSize=(21, 21))
        # self.hist = cv2.equalizeHist(self.blur)
        self.hist = clahe_3.apply(self.blur)
        hist_1 = clahe_1.apply(self.blur)
        hist_2 = clahe_2.apply(self.blur)
        hist_3 = clahe_3.apply(self.blur)
        # hist_4 = cv2.equalizeHist(self.blur)
        cv2.imshow("hist_0", self.hist)
        cv2.imshow("hist_1", hist_1)
        cv2.imshow("hist_2", hist_2)
        cv2.imshow("hist_3", hist_3)
        # cv2.imshow("hist_4", hist_4)
        cv2.waitKey(0)

    def get_binary_image(self, thresh):
        ret, self.binary = cv2.threshold(self.hist, thresh, constant.MAX_VALUE, cv2.THRESH_BINARY)
        th = cv2.adaptiveThreshold(self.hist, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
        canny = cv2.Canny(self.hist, 100, 220)
        cv2.imshow("Adaptive", th)
        cv2.imshow("Binary", self.binary)
        cv2.waitKey(0)

    def find_image_contours(self):
        contours, hierarchy = cv2.findContours(self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        return contours

    def get_mask_list(self, contours):
        mask_list = []
        rotate_rect = []
        img_copy = self.hist.copy()
        img_copy_1 = self.hist.copy()
        img_copy_2 = self.hist.copy()

        for con in contours:
            # Mặt nạ lọc miền
            mask = np.zeros(self.hist.shape[:2], dtype=np.uint8)

            # Vùng chữ nhật bao quanh vùng nghi ngờ
            rect = cv2.minAreaRect(con)
            # print(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_copy, [box], -1, (0, 255, 0), 1, cv2.LINE_AA)
            # Loại vùng theo góc
            # Góc nghiêng của vùng được chọn
            angle = - rect[-1]
            if angle < constant.ANGLE_OFFSET or 90 - angle < constant.ANGLE_OFFSET:
                w = rect[1][0]
                h = rect[1][1]

                cv2.drawContours(img_copy_1, [box], constant.CONTOUR_IDX, constant.COLOR_GREEN,
                                 constant.EDGES_NORMAL_THICKNESS,
                                 constant.LINE_TYPE)

                # Loại theo diện tích và cạnh
                if constant.MIN_PLATE_AREA < w * h < constant.MAX_PLATE_AREA \
                        and (constant.EDGE_PLATE_MIN_RATIO < w / h < constant.EDGE_PLATE_MAX_RATIO
                             or constant.EDGE_PLATE_MIN_RATIO < h / w < constant.EDGE_PLATE_MAX_RATIO):
                    # Vẽ các vùng thỏa mãn lên mặt nạ
                    cv2.drawContours(mask, [box], constant.CONTOUR_IDX, constant.COLOR_WHITE,
                                     constant.EDGES_FULL_THICKNESS,
                                     constant.LINE_TYPE)
                    cv2.drawContours(img_copy_2, [box], constant.CONTOUR_IDX, constant.COLOR_GREEN,
                                     1, constant.LINE_TYPE)
                    mask_list.append(mask)
                    rotate_rect.append(rect)
        # cv2.imshow("image_1", img_copy)
        # cv2.imshow("image_2", img_copy_1)
        # cv2.imshow("image_3", img_copy_2)
        cv2.waitKey(0)
        return mask_list, rotate_rect

    def get_mask_image(self, mask_list, rotate_rect):

        mask_image = []

        for i in range(len(mask_list)):
            # Ảnh gray có backgrond đen và vùng ảnh gốc được lấy ra bởi mask
            gray = cv2.bitwise_and(self.hist, self.hist, mask=mask_list[i])

            # Thực hiện xoay ảnh
            # Lấy góc tương ứng với mask
            angle = rotate_rect[i][-1]
            if angle < -45:
                angle = - (90 + angle)
            else:
                angle = -angle

            # Nếu góc tương ứng với mask != 0 hoặc 90 thì xoay
            if angle != 0 and angle != 90:
                gray = self.rotate_image(gray, angle)

            # Nhị phân hóa ảnh
            ret, th = cv2.threshold(gray, constant.THRESH_FOR_ROTATE_IMAGE, constant.THRESH_MAX_VALUE,
                                    cv2.THRESH_BINARY)

            # Lấy viền
            contour, hierarchy = cv2.findContours(th, constant.FIND_CONTOUR_MODE, constant.FIND_CONTOUR_METHOD)

            # Lấy ảnh vùng ROI
            x, y, w, h = cv2.boundingRect(contour[0])
            img_plate = gray[y:y + h, x:x + w]
            mask_image.append(img_plate)
        return mask_image

    def check_plate_image(self, mask_image):
        # Duyệt mảng gồm ảnh các vùng lấy được từ mask
        for img in mask_image:
            character_segmentation = CharacterSegmentation(img)
            character_segmentation.get_character()

            if 7 <= len(character_segmentation.character) < 10:
                return img, character_segmentation.character, len(character_segmentation.character)
        return None, None, 0

    def rotate_image(self, img, angle):
        (h, w) = img.shape[:2]

        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        new_img = cv2.warpAffine(img, M, (w, h))
        return new_img

    # Tìm vùng biển số xe
    def find_plate_location(self, threshold):
        self.get_binary_image(threshold)
        contours = self.find_image_contours()
        mask_list, rotate_rect = self.get_mask_list(contours)
        mask_image = self.get_mask_image(mask_list, rotate_rect)
        img, character, count = self.check_plate_image(mask_image)
        if img is not None:
            return True, img, character, count
        else:
            return False, img, character, count

    def get_plate_image(self, plate_property, num_c, light):
        self.resize_image()
        self.grayscale()

        self.blur_image(plate_property)

        self.increase_contrast()

        # Mảng các vùng thỏa mãn
        plate_list = []

        for i in range(constant.THRESH_MIN_VALUE, constant.THRESH_MAX_VALUE, 5):
            result, img, character, count = self.find_plate_location(i)
            if result:
                plate_list.append((img, count, character))

        if len(plate_list) > 0:
            # Sắp xếp theo diện tích ảnh và số vùng
            plate_list = sorted(plate_list, key=lambda x: (x[1], - x[0].shape[0] * x[0].shape[1]), reverse=True)
            self.plate = plate_list[0]
        else:
            self.plate = None
