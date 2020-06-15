import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from detector import Detector
from character_segmentation import CharacterSegmentation
from read_data import ReadData
import constant

# Tải ảnh lên
img = cv2.imread("./test/img_23.jpg")

# Khởi tạo đối tượng nhận diện biển số
detector = Detector(img)

# Tìm vị trí của biển số
detector.get_plate_image()

if detector.plate is None:
    print("Không nhận dạng được biển số")
else:
    character_segmentation = CharacterSegmentation(detector.plate)

    # Thực hiện các bước phân vùng chữ
    character_segmentation.get_character()

    # Đưa dữ liệu vào trong model để dự đoán
    model = tf.keras.models.load_model('./my_model/')

    # model.summary()

    # read_data = ReadData("./training_folder")
    # read_data.read_data()

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(constant.LABEL)

    y_pred = model.predict(np.array(character_segmentation.character))
    # print(y_pred)
    result = label_binarizer.inverse_transform(y_pred)
    lisence = ""
    for j in range(len(result)):
        lisence += str(result[j])
        if j == 1:
            lisence += "-"
    print(lisence)
