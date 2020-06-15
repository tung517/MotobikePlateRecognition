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
from detector import Detector
from character_segmentation import CharacterSegmentation
import constant


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("My App")
        self.minsize(800, 600)

        self.img_origin = None
        self.img_plate = None
        self.plate_type = None

        self.image_type_value = IntVar()

        self.label_img_ori = self.create_label(150, 20, text='Ảnh gốc', bg='#ed34f5')

        self.canvas_img_ori = self.create_canvas(50, 50, 250, 250, img=self.img_origin)

        self.label_img_plate = self.create_label(450, 20, text='Ảnh biển số', bg='#ed34f5')

        self.canvas_img_plate = self.create_canvas(350, 50, 250, 250)

        fontStyle = tkfont.Font(family='Arial', size=15)
        self.label_result = self.create_label(650, 175, text="               ", bg="#b4eb8a", font=fontStyle)

        self.btn_load_image = self.create_button(50, 330, "Tải ảnh lên", self.click_load_image)

        self.btn_recognize_plate = self.create_button(350, 330, "Nhận diện biển số", self.click_recognize_plate)

        self.label_option = self.create_label(50, 400, "Tùy chọn", bg='#ed34f5')

        self.label_image_type = self.create_label(150, 400, "Loại biển", bg='#ed34f5')

        self.rb_white_plate = self.create_radio_button(150, 450, "Biển trắng", 1, self.radio_event)

        self.rb_color_plate = self.create_radio_button(150, 500, "Biển màu", 2, self.radio_event)

        self.label_plate_properties = self.create_label(300, 400, "Đặc điểm biển số", bg='#ed34f5')

        self.cb_dirty_plate = self.create_checkbox(300, 450, "Biển bị bụi, bẩn")

        self.cb_broken_plate = self.create_checkbox(300, 500, "Biển bị vỡ")

    def create_label(self, x, y, text=None, bg=None, font=None):
        label = Label(self, text=text, bg=bg)
        label.configure(anchor='center')
        label.configure(font=font)
        label.place(x=x, y=y)
        return label

    def create_canvas(self, x, y, width, height, img=None):
        canvas = Canvas(self, width=width, height=height, highlightthickness=1, highlightbackground="black")
        canvas.create_image(0, 0, anchor=NW, image=img)
        canvas.place(x=x, y=y)
        return canvas

    def load_image_origin(self, path, size=None):
        self.img_origin = Image.open(path)
        self.img_origin = self.img_origin.resize(size, Image.ANTIALIAS)
        self.img_origin = ImageTk.PhotoImage(self.img_origin)

    def create_button(self, x, y, text, click_event):
        button = ttk.Button(self, text=text, command=click_event)
        button.place(x=x, y=y)
        return button

    def create_radio_button(self, x, y, text, value, event):
        radio_button = ttk.Radiobutton(self, text=text, value=value, variable=self.image_type_value, command=event)
        radio_button.place(x=x, y=y)
        return radio_button

    def create_checkbox(self, x, y, text):
        checkbox = ttk.Checkbutton(self, text=text)
        checkbox.place(x=x, y=y)
        return checkbox

    def radio_event(self):
        print(self.image_type_value.get())

    def file_dialog(self):
        self.filename = filedialog.askopenfilename(initialdir="./", title='Chose image',
                                                   filetypes=(('jpeg', '*.jpg'), ('jpeg', '*.jpeg'), ('png', '*.png'),
                                                              ('All file', '*.*')))

    def click_load_image(self):
        self.file_dialog()
        self.load_image_origin(self.filename, size=(250, 250))
        self.canvas_img_ori.create_image(0, 0, anchor=NW, image=self.img_origin)

    def click_recognize_plate(self):
        # Tải ảnh lên
        img = cv2.imread(self.filename)

        # Khởi tạo đối tượng nhận diện biển số
        detector = Detector(img)

        # Tìm vị trí của biển số
        detector.get_plate_image()

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

    def convert_image(self, img, size):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_plate = Image.fromarray(image)
        self.img_plate = self.img_plate.resize(size, Image.ANTIALIAS)
        self.img_plate = ImageTk.PhotoImage(self.img_plate)


root = Root()
root.mainloop()
