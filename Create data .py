import cv2
import os
import time
import numpy as np
detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
id=2
if id == 2:
    print(0)
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
if id == 2:
    cap = cv2.VideoCapture(0)
    detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    sampleNum = 0
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fa = detector.detectMultiScale(gray, 1.1, 5)
        for(x,y,w,h) in fa:
            cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
            if not os.path.exists('dataset'):
                os.makedirs('dataset')
            sampleNum+=1
            cv2.imwrite('dataset/anh'+str(5)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])
        cv2.imshow('frame',frame)
        #time.sleep(0.1)
        print("Đã chụp thành công ảnh số ",sampleNum)
        if sampleNum > 99:
            break;
    cap.release()
    cv2.destroyAllWindows()
