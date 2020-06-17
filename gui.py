from tkinter import *
from tkinter import ttk, _setit
from tkinter import filedialog
import tkinter.font as tkfont
from PIL import ImageTk, Image
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tkinter import messagebox

from tensorflow.python.keras.backend import set_floatx

from detector import Detector
import constant

PLATE_COLOR_VALUE = 1
PLATE_PROPERTIES_VALUE = 2
NUM_CHARACTER_VALUE = 3
LIGHT_CONDITION = 4


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Motorbike Recognition App")
        self.minsize(800, 600)

        self.img_origin = None
        self.img_plate = None
        self.plate_type = None

        # Radio value vùng đặc điểm biển sô
        self.plate_properties_value = IntVar()
        self.plate_properties_value.set(1)

        # Radio value vùng số kí tự
        self.num_character_value = IntVar()
        self.num_character_value.set(2)

        # Radio value vùng điều kiện ánh sáng
        self.light_condition_value = IntVar()
        self.light_condition_value.set(1)

        # Ảnh gốc
        self.origin_image_area()

        # Ảnh biển số
        self.plate_image_area()

        # Kết quả biển số
        font_style = tkfont.Font(family='Arial', size=15)
        self.label_result = self.create_label(650, 175, text="               ", bg="#b4eb8a", font=font_style)

        # Option
        self.label_option = self.create_label(50, 400, "Tùy chọn", bg='#ed34f5')

        # Vùng option đặc điểm biển số
        self.plate_properties()

        # Vùng số chữ số của biển
        self.number_character_of_plate()

        # Vùng điều kiện ánh sáng
        self.light_condition()

    # Vùng ảnh gốc
    def origin_image_area(self):
        self.label_img_ori = self.create_label(150, 20, text='Ảnh gốc', bg='#ed34f5')
        self.canvas_img_ori = self.create_canvas(50, 50, 250, 250, img=self.img_origin)
        self.btn_load_image = self.create_button(50, 330, "Tải ảnh lên", self.click_load_image)

    # Vùng ảnh biển số
    def plate_image_area(self):
        self.label_img_plate = self.create_label(440, 20, text='Ảnh biển số', bg='#ed34f5')
        self.canvas_img_plate = self.create_canvas(350, 50, 250, 250)
        self.btn_recognize_plate = self.create_button(350, 330, "Nhận diện biển số", self.click_recognize_plate)

    # Vùng option đặc điểm biển số
    def plate_properties(self):
        self.label_plate_properties = self.create_label(150, 400, "Đặc điểm biển số", bg='#ed34f5')
        self.rb_normal_plate = self.create_radio_button(150, 440, "Biển rõ", 1, PLATE_PROPERTIES_VALUE,
                                                        lambda: self.radio_event(PLATE_PROPERTIES_VALUE))
        self.rb_dirty_plate = self.create_radio_button(150, 470, "Biển mờ", 2, PLATE_PROPERTIES_VALUE,
                                                       lambda: self.radio_event(PLATE_PROPERTIES_VALUE))
        self.rb_dirty_plate = self.create_radio_button(150, 500, "Biển bụi, bẩn", 3, PLATE_PROPERTIES_VALUE,
                                                       lambda: self.radio_event(PLATE_PROPERTIES_VALUE))

    # Vùng option só chữ số
    def number_character_of_plate(self):
        self.label_num_character = self.create_label(350, 400, "Số kí tự", bg='#ed34f5')
        self.rb_num_three = self.create_radio_button(350, 440, "3 số", 1, NUM_CHARACTER_VALUE, lambda: self.radio_event(
            NUM_CHARACTER_VALUE))
        self.rb_num_four = self.create_radio_button(350, 470, "4 số", 2, NUM_CHARACTER_VALUE, lambda: self.radio_event(
            NUM_CHARACTER_VALUE))
        self.rb_num_five = self.create_radio_button(350, 500, "5 số", 3, NUM_CHARACTER_VALUE, lambda: self.radio_event(
            NUM_CHARACTER_VALUE))

    # Vùng option điều kiện ánh sáng
    def light_condition(self):
        self.label_light_condition = self.create_label(500, 400, "Điều kiện ánh sáng", bg='#ed34f5')
        self.rb_day = self.create_radio_button(500, 440, "Ngày", 1, LIGHT_CONDITION,
                                               lambda: self.radio_event(LIGHT_CONDITION))
        self.rb_night = self.create_radio_button(500, 470, "Đêm", 2, LIGHT_CONDITION,
                                                 lambda: self.radio_event(LIGHT_CONDITION))

    # Tạo label
    def create_label(self, x, y, text=None, bg=None, font=None):
        label = Label(self, text=text, bg=bg)
        label.configure(anchor='center')
        label.configure(font=font)
        label.place(x=x, y=y)
        return label

    # Tạo khung ảnh
    def create_canvas(self, x, y, width, height, img=None):
        canvas = Canvas(self, width=width, height=height, highlightthickness=1, highlightbackground="black")
        canvas.create_image(0, 0, anchor=NW, image=img)
        canvas.place(x=x, y=y)
        return canvas

    # Load image ảnh gốc
    def load_image_origin(self, path, size=None):
        self.img_origin = Image.open(path)
        self.img_origin = self.img_origin.resize(size, Image.ANTIALIAS)
        self.img_origin = ImageTk.PhotoImage(self.img_origin)

    # Tạo button
    def create_button(self, x, y, text, click_event):
        button = ttk.Button(self, text=text, command=click_event)
        button.place(x=x, y=y)
        return button

    # Tạo radio button
    def create_radio_button(self, x, y, text, value, variable, event):
        if variable == PLATE_PROPERTIES_VALUE:
            radio_button = ttk.Radiobutton(self, text=text, value=value, variable=self.plate_properties_value,
                                           command=event)
            radio_button.place(x=x, y=y)
            return radio_button
        elif variable == NUM_CHARACTER_VALUE:
            radio_button = ttk.Radiobutton(self, text=text, value=value, variable=self.num_character_value,
                                           command=event)
            radio_button.place(x=x, y=y)
            return radio_button
        else:
            radio_button = ttk.Radiobutton(self, text=text, value=value, variable=self.light_condition_value,
                                           command=event)
            radio_button.place(x=x, y=y)
            return radio_button

    # Xử lí sự kiện radio button
    def radio_event(self, value):
        if value == PLATE_PROPERTIES_VALUE:
            print(self.plate_properties_value.get())
        elif value == NUM_CHARACTER_VALUE:
            print(self.num_character_value.get())
        else:
            print(self.light_condition_value.get())

    # Tạo load image dialog
    def file_dialog(self):
        self.filename = filedialog.askopenfilename(initialdir="./", title='Chose image',
                                                   filetypes=(('jpeg', '*.jpg'), ('jpeg', '*.jpeg'), ('png', '*.png'),
                                                              ('All file', '*.*')))

    # Xử lí sự kiện load image button
    def click_load_image(self):
        self.file_dialog()
        self.load_image_origin(self.filename, size=(250, 250))
        self.canvas_img_ori.create_image(0, 0, anchor=NW, image=self.img_origin)

    # xử lí sự kiện click recognize image button
    def click_recognize_plate(self):
        # Tải ảnh lên
        img = cv2.imread(self.filename)

        # Khởi tạo đối tượng nhận diện biển số
        detector = Detector(img)

        # Các thông số về biển số
        plate_property = self.plate_properties_value.get()
        num_c = self.num_character_value.get()
        light = self.light_condition_value.get()

        # Tìm vị trí của biển số
        detector.get_plate_image(plate_property, num_c, light)

        if detector.plate is None:
            messagebox.showinfo(title="Thông báo", message="Không nhận dạng được biển số")
        else:

            self.convert_image(detector.plate[0], (250, 250))
            self.canvas_img_plate.create_image(0, 0, anchor=NW, image=self.img_plate)

            # character_segmentation = CharacterSegmentation(detector.plate)
            #
            # # Thực hiện các bước phân vùng chữ
            # character_segmentation.get_character()

            # Đưa dữ liệu vào trong model để dự đoán
            model = tf.keras.models.load_model('./my_model/')

            label_binarizer = LabelBinarizer()
            label_binarizer.fit(constant.LABEL)

            y_pred = model.predict(np.array(detector.plate[2]))

            result = label_binarizer.inverse_transform(y_pred)
            lisence = ""
            for j in range(len(result)):
                lisence += str(result[j])
                if j == 1:
                    lisence += "-"

            self.label_result.configure(text=lisence)

    # Chuyển ảnh Mat -> Image
    def convert_image(self, img, size):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_plate = Image.fromarray(image)
        self.img_plate = self.img_plate.resize(size, Image.ANTIALIAS)
        self.img_plate = ImageTk.PhotoImage(self.img_plate)


root = Root()
root.mainloop()
