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
        self.hist = clahe.apply(self.blur)
        cv2.imwrite("image_pre.jpg", self.hist)

    # Phân ngưỡng nhị phân
    def get_binary_image(self, thresh):
        ret, self.binary = cv2.threshold(self.hist, thresh, constant.MAX_VALUE, cv2.THRESH_BINARY)
        # if thresh == 210:
        #     cv2.imwrite("thresh.jpg", self.binary)

    # Tìm viền
    def find_image_contours(self, threshold):
        contours, hierarchy = cv2.findContours(self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # if threshold == 210:
        #     img_copy = self.img.copy()
        #     image = cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 2)
        #     cv2.imwrite("contours_0_1.jpg", image)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        # if threshold == 210:
        #     img_copy = self.img.copy()
        #     image = cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 2)
        #     cv2.imwrite("contours_0_2.jpg", image)
        # if threshold == 210:
        #     image = np.zeros((600, 600), np.uint8)
        #     image = cv2.drawContours(image, contours, -1, (255, 255, 255), -1)
        #     cv2.imwrite("contours.jpg", image)
        return contours

    def get_mask_list(self, contours, threshold):
        mask_list = []
        rotate_rect = []
        image_1 = np.zeros((600, 600), np.uint8)
        image_2 = np.zeros((600, 600), np.uint8)
        image_3 = np.zeros((600, 600), np.uint8)
        img_copy = self.img.copy()
        # img_copy_1 = self.hist.copy()
        # img_copy_2 = self.hist.copy()

        for con in contours:
            # Mặt nạ lọc miền
            mask = np.zeros(self.hist.shape[:2], dtype=np.uint8)

            # Vùng chữ nhật bao quanh vùng nghi ngờ
            rect = cv2.minAreaRect(con)
            # print(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # if threshold == 210:
            #     cv2.drawContours(img_copy, [box], -1, (0, 255, 0), 2, cv2.LINE_AA)
            # Loại vùng theo góc
            # Góc nghiêng của vùng được chọn
            angle = - rect[-1]
            if angle < constant.ANGLE_OFFSET or 90 - angle < constant.ANGLE_OFFSET:
                w = rect[1][0]
                h = rect[1][1]
                # if threshold == 210:
                #     image_1 = cv2.drawContours(image_1, [con], -1, (255, 255, 255), -1, cv2.LINE_AA)
                # Loại theo diện tích và cạnh
                if constant.MIN_PLATE_AREA < w * h < constant.MAX_PLATE_AREA:
                    # if threshold == 210:
                    #     image_2 = cv2.drawContours(image_2, [con], -1, (255, 255, 255), -1, cv2.LINE_AA)
                    if (constant.EDGE_PLATE_MIN_RATIO < w / h < constant.EDGE_PLATE_MAX_RATIO
                            or constant.EDGE_PLATE_MIN_RATIO < h / w < constant.EDGE_PLATE_MAX_RATIO):
                        # if threshold == 210:
                        #     image_3 = cv2.drawContours(image_3, [con], -1, (255, 255, 255), -1, cv2.LINE_AA)
                        # Vẽ các vùng thỏa mãn lên mặt nạ
                        cv2.drawContours(mask, [box], constant.CONTOUR_IDX, constant.COLOR_WHITE,
                                         constant.EDGES_FULL_THICKNESS,
                                         constant.LINE_TYPE)
                        if threshold == 210:
                            image = cv2.drawContours(img_copy, [box], -1, (0, 255, 0), 2, cv2.LINE_AA)

                        mask_list.append(mask)
                        rotate_rect.append(rect)
        # if threshold == 210:
        # cv2.imwrite("contours_1.jpg", image_1)
        # cv2.imwrite("contours_2.jpg", image_2)
        # cv2.imwrite("contours_3.jpg", image_3)
        # cv2.imwrite("box_0.jpg", img_copy)
        # cv2.imwrite("box_1.jpg", img_copy)
        # cv2.imshow("image_2", img_copy_1)
        # cv2.imshow("image_3", img_copy_2)
        # cv2.waitKey(0)
        return mask_list, rotate_rect

    def get_mask_image(self, mask_list, rotate_rect, threshold):

        mask_image = []

        for i in range(len(mask_list)):
            # Ảnh gray có backgrond đen và vùng ảnh gốc được lấy ra bởi mask
            gray = cv2.bitwise_and(self.gray, self.gray, mask=mask_list[i])
            if threshold == 210:
                cv2.imwrite("plate_" + str(i) + ".jpg", gray)

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
            ret, th = cv2.threshold(gray, constant.THRESH_FOR_ROTATE_IMAGE, constant.MAX_VALUE,
                                    cv2.THRESH_BINARY)

            # Lấy viền
            contour, hierarchy = cv2.findContours(th, constant.FIND_CONTOUR_MODE, constant.FIND_CONTOUR_METHOD)

            if len(contour) == 0:
                continue
            else:
                # Lấy ảnh vùng ROI
                x, y, w, h = cv2.boundingRect(contour[0])
                img_plate = gray[y:y + h, x:x + w]
                mask_image.append(img_plate)
        return mask_image

    def check_plate_image(self, mask_image, plate_property, num_c, threshold):
        # Duyệt mảng gồm ảnh các vùng lấy được từ mask
        i = 0
        print("Mask = " + str(len(mask_image)))
        for img in mask_image:
            character_segmentation = CharacterSegmentation(img, threshold, i)
            character_segmentation.get_character(plate_property, num_c)
            if num_c != 4:
                if len(character_segmentation.character) == num_c + 6:
                    return img, character_segmentation.character, len(character_segmentation.character)
            else:
                if 7 <= len(character_segmentation.character) < 10:
                    return img, character_segmentation.character, len(character_segmentation.character)
            i += 1
        return None, None, 0

    # Xoay ảnh
    def rotate_image(self, img, angle):
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        new_img = cv2.warpAffine(img, M, (w, h))
        return new_img

    # Tìm vùng biển số xe
    def find_plate_location(self, threshold, plate_property, num_c):
        self.get_binary_image(threshold)
        contours = self.find_image_contours(threshold)
        mask_list, rotate_rect = self.get_mask_list(contours, threshold)
        mask_image = self.get_mask_image(mask_list, rotate_rect, threshold)
        img, character, count = self.check_plate_image(mask_image, plate_property, num_c, threshold)
        if img is not None:
            return threshold, img, character, count
        else:
            return 0, img, character, count

    def get_plate_image(self, plate_property, num_c, light):
        self.resize_image()
        self.grayscale()
        self.blur_image(plate_property)
        self.increase_contrast()
        # Mảng các vùng thỏa mãn
        plate_list = []
        if light == 1 or (light == 3 and self.img.mean() > 80):
            for i in range(constant.THRESH_MIN_VALUE_LIGHT, constant.THRESH_MAX_VALUE_LIGHT, 5):
                result, img, character, count = self.find_plate_location(i, plate_property, num_c)
                if result != 0:
                    plate_list.append((img, count, character, result))
        else:
            for i in range(constant.THRESH_MIN_VALUE_DARK, constant.THRESH_MAX_VALUE_DARK, 5):
                result, img, character, count = self.find_plate_location(i, plate_property, num_c)
                if result != 0:
                    plate_list.append((img, count, character, result))

        if len(plate_list) > 0:
            # Sắp xếp theo diện tích ảnh và số vùng
            plate_list = sorted(plate_list, key=lambda x: (x[1], - x[0].shape[0] * x[0].shape[1]), reverse=True)
            self.plate = plate_list[0]
            print(self.plate[3])
        else:
            self.plate = None
