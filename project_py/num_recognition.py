from keras.models import load_model
model = load_model("./models/mnist_0.h5")

import numpy as np


def set_image():

    import sys
    import cv2
    import glob

    path = './cell_data\\'
    filenames = glob.glob(path + '*.png')

    num_coords = []

    if len(filenames) == 0:
        print("no such directory")
        sys.exit()

    results = []
    pixel_size = 28
    cnt = 0

    for fname in filenames:
        #이미지 경로에서 수도쿠 판에서의 좌표가 될 숫자들만 추출
        temp = fname[-7:-4]
        temp = str(temp).replace('_', '')
        num_coords.append(temp)

        #이미지 읽기
        image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(255 - image, (28, 28))
        image = image.reshape(28, 28, 1)
        image = image.astype('float32')
        image /= 255

        results.append(image)

    return results, num_coords


def dict_reshape(coord_pred):

    sudoku_board = [] #9개의 0이 채워질 리스트
    y_pred = [] #딕셔너리에서 정답들만 뽑아올 자리

    # 9 x 9 의 스도쿠 판을 만든다
    for i in range(0,9):
        sudoku_board.append(list('000000000'))

    # 딕셔너리에서 예측값들을 뽑아온다
    for answer in coord_pred.values():
        y_pred.append(answer)

    cnt = 0
    for coords in coord_pred:
        sudoku_board[int(coords[0])][int(coords[1])] = str(y_pred[cnt]) #'200006100' 이런식으로 예측된 숫자들을 한줄에 넣어준다
        cnt += 1

    return sudoku_board




def predict():
    #   이미지를 불러오고 인식을 위한 처리
    images, num_coords = set_image()
    images = np.array(images)
    #   딥러닝 모델로 숫자 예측
    y_pred = model.predict(images)
    coord_pred = {}
    print('predictions: ', end='')
    for i in range(0, len(y_pred)):
        print(np.argmax(y_pred[i]), end='  ')
        coord_pred[num_coords[i]] = np.argmax(y_pred[i])

    print()

    sudoku_board = dict_reshape(coord_pred)
    return sudoku_board

#predict()