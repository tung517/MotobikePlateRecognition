from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from read_data import ReadData

# Lấy dữ liệu đầu vào từ ảnh gốc
read_data = ReadData("./training_folder")
read_data.read_data()

# Chuyển dữ liệu sang dạng np.array
X = read_data.X
y = read_data.y

print(X.shape)
print(y.shape)

# Chuyển dữ liệu nhãn sang vector One Hot
label_binarizer = LabelBinarizer()
label_binarizer.fit(y)
y = label_binarizer.transform(y)
print(X.shape)
print(y.shape)

X_train, X_validate, X_test = np.split(X, [int(0.6 * len(X)), int(0.8 * len(X))])
y_train, y_validate, y_test = np.split(y, [int(0.6 * len(y)), int(0.8 * len(y))])

print(X_train.shape)
print(X_validate.shape)
print(X_test.shape)

print(y_train.shape)
print(y_validate.shape)
print(y_test.shape)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=[32, 32, 1], activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=30, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

epochs = 55

history = model.fit(X_train, y_train, batch_size=64, epochs=epochs, verbose=1, validation_data=(X_validate, y_validate))

model.save("my_model")

epoch_range = range(0, epochs)

# Biểu đồ accuracy
plt.plot(epoch_range, history.history['accuracy'])
plt.plot(epoch_range, history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Biểu đồ loss
plt.plot(epoch_range, history.history['loss'])
plt.plot(epoch_range, history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

y_pred = model.predict(np.array(X_test))

for i in range(len(y_pred)):
    for j in range(len(y_pred[i])):
        if y_pred[i][j] < y_pred[i].max():
            y_pred[i][j] = 0
        else:
            y_pred[i][j] = 1

score = accuracy_score(y_test, y_pred)
print(score)

print(y_pred)
print(label_binarizer.inverse_transform(y_pred))
print(label_binarizer.inverse_transform(y_test))
