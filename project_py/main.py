import crop_numbers as cn
import num_recognition as rec
import sudoku_solver as sv
import show_result as res
import cv2

import matplotlib.pyplot as plt

path = "./images/sudoku4.jpg"
############스도쿠 이미지 처리
coords = cn.sudoku_v1(path)

print()

############숫자 이미지 예측
pred_9x9 = rec.predict()
print(pred_9x9)

print()

############스도쿠 알고리즘
answers = sv.sudoku_pro(pred_9x9)
print("answer:",answers)

print()

############이미지 정답 출력 알고리즘
result_img = res.print_back(path, coords, answers)
print()

plt.title('Result')
plt.xticks([])
plt.yticks([])
plt.imshow(result_img, cmap='rgb')
plt.show()
# cv2.imshow('Result', result_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



