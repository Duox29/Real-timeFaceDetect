import cv2
import os
import time
import numpy as np

detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

print("Menu:")
print("1. NHập ảnh từ file data")
print("2. Nhập ảnh từ webcam")
print("0. Thoát")
id = input("Nhập lựa chọn của bạn: ").strip()

if id == '1':
    print(0)
    with open("config.txt", "w") as file:
        file.write(f"nModel = 5\nnImg = 20")

    for i in range(1,6):
        for j in range (1,21):
            filename = 'data/anh.'  + str(i) + '.' +str(j) + '.jpg'
            frame = cv2.imread(filename)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fa = detector.detectMultiScale(gray, 1.1, 5)
            for(x,y,w,h) in fa:
                cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
                if not os.path.exists('dataset'):
                    os.makedirs('dataset')
                cv2.imwrite('dataset/anh'  + str(i) + '.' +str(j) + '.jpg', gray[y:y+h,x:x+w])


# Nhập ảnh từ webcam
if id == '2':
    nModel = int(input("Số lượng model: "))
    nImg = int(input("Số lượng ảnh mỗi model:"))
    with open("config.txt", "w") as file:
        file.write(f"nModel = {nModel}\nnImg = {nImg}")

    for i in range(1,nModel+1):
        cap = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        sampleNum = 0
        while (True):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fa = detector.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in fa:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if not os.path.exists('dataset'):
                    os.makedirs('dataset')
                sampleNum += 1
                cv2.imwrite('dataset/anh' + str(5) + '.' + str(sampleNum) + '.jpg', gray[y:y + h, x:x + w])
            cv2.imshow('frame', frame)
            time.sleep(0.1)
            print("Đã chụp thành công ảnh số ", sampleNum)
            if sampleNum > nImg-1:
                break;
        # modelName = input("Nhập tên model số ",i,":")
        # with open("infoModel.txt","w") as file:
        #     file.write(f"modelName = {modelName} - i\n")
        prompt = input("Bạn có muốn tiếp tục(y/n): ")
        if prompt == 'n':
            break;
    cap.release()
    cv2.destroyAllWindows()