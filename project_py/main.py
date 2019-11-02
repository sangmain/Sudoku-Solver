import crop_numbers as cn
import num_recognition as rec
import sudoku_solver as sv
import show_result as res
import cv2

import matplotlib.pyplot as plt

Debug = False
# cap = cv2.VideoCapture(0)

# if (cap.isOpened() == False): 
#   print("Unable to read camera feed")
 
# # Default resolutions of the frame are obtained.The default resolutions are system dependent.
# # We convert the resolutions from float to integer.
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
 
# # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
 
# while(True):
#   ret, frame = cap.read()
 
#   if ret == True: 
     
#     # Write the frame into the file 'output.avi'
#     out.write(frame)
 
#     # Display the resulting frame    
#     cv2.imshow('frame',frame)
 
#     # Press Q on keyboard to stop recording
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#       break
 
#   # Break the loop
#   else:
#     break 
 
# # When everything done, release the video capture and video write objects
# cap.release()
# out.release()
 
# # Closes all the frames
# cv2.destroyAllWindows() 

path = "./images/sudoku4.jpg"
img= cv2.imread(path, 1)

if Debug:
    cv2.imshow("original", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

coords = cn.image_process(img, Debug) #스도쿠 이미지 처리
pred_9x9 = rec.predict(Debug) #mnist 숫자 이미지 예측
answers = sv.sudoku_pro(pred_9x9) #스도쿠 문제 해결 알고리즘
print("answer:",answers)

result_img = res.print_back(path, coords, answers) #이미지 정답 출력 알고리즘

fig = plt.figure(figsize=(10, 8))

plot = fig.add_subplot(1, 2, 1)
plot.set_title('Original image')
plt.imshow(img)
plt.xticks([])
plt.yticks([])

plot = fig.add_subplot(1, 2, 2)
plot.set_title('Result')
plt.xticks([])
plt.yticks([])
plt.imshow(result_img)

plt.show()