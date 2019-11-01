from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np


def excludeZero(x, y): # 수도쿠에는 0이라는 숫자가 없기 때문에 mnist 데이터에서 0을 제외한다

    cnt = 0
    temp1 = []
    temp2 = []

    for i in range(0, len(x)):
        if np.argmax(y[i]) == 0:
            continue

        temp1.append(x[i])
        temp2.append(y[i])

        cnt += 1

    return np.array(temp1), np.array(temp2)


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
shape = 28
channel = 1
#   mnist 에서 데이터 받아오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#   one hot 인코딩
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#   학습시킬 데이터에 0 제외
x_train, y_train = excludeZero(x_train, y_train)
x_test, y_test = excludeZero(x_test, y_test)

print('x_train: ', x_train.shape)
print('x_test: ', x_test.shape)

#input shape 맞추기
x_train = x_train.reshape(x_train.shape[0], shape, shape, channel)
x_test = x_test.reshape(x_test.shape[0], shape, shape, channel)

#전처리
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train: ', x_train.shape)
print('x_test: ', x_test.shape)


#모델
#   모델 레이어 추가
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(shape, shape, channel)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


#   학습 방법 설정
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#   학습 시작
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("models/model.h5")
