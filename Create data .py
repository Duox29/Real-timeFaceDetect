import cv2
import os
import time
import numpy as np
# Khởi tạo bộ phát hiện khuôn mặt
detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

print("Menu:")
print("1. NHập ảnh từ file data")
print("2. Nhập ảnh từ webcam")
print("0. Thoát")
id = input("Nhập lựa chọn của bạn: ").strip()
# Nhập hình ảnh từ file data có sẵn
if id == '1':
    print(0)
    # Ghi thông tin cấu hình vào file config.txt
    with open("config.txt", "w") as file:
        file.write(f"nModel = 6\nnImg = 20")

    # Xóa dữ liệu trong file infoModel.txt
    with open("infoModel.txt", "w") as file:
            pass
    # Nhập tên các mẫu khuôn mặt
    for i in range(1,7):
        print("Nhập tên mẫu ", i, ": ")
        modelName = input()
        with open("infoModel.txt", "a") as file:
            file.write(f"modelName = {modelName} - {i}\n")
    # Xử lý từng ảnh trong file data
    for i in range(1,7):

        for j in range (1,21):
            filename = 'data/anh.'  + str(i) + '.' +str(j) + '.jpg'
            # Đọc hình ảnh từ tệp tin được chỉ định bởi filename
            frame = cv2.imread(filename)
            # Chuyển đổi hình ảnh sang ảnh xám
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Phát hiện khuôn mặt trong ảnh
            fa = detector.detectMultiScale(gray, 1.1, 5)
            for(x,y,w,h) in fa:
                # Vẽ hình chữ nhật quanh khuôn mặt
                cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
                # Kiểm tra thư mục dataset có tồn tại hay không, nếu không thì tạo thư mục
                if not os.path.exists('dataset'):
                    os.makedirs('dataset')
                # Lưu khu vực khuôn mặt đã phát hiện vào thư mục dataset
                cv2.imwrite('dataset/anh'  + str(i) + '.' +str(j) + '.jpg', gray[y:y+h,x:x+w])


# Nhập ảnh từ webcam
if id == '2':
    nModel = int(input("Số lượng mẫu: "))
    nImg = int(input("Số lượng ảnh mỗi mẫu:"))
    # Ghi thông tin model vào file config.txt
    with open("config.txt", "w") as file:
        file.write(f"nModel = {nModel}\nnImg = {nImg}")

    for i in range(1,nModel+1):
        # Xóa toàn bộ dữ liệu trong file infoModel.txt
        with open("infoModel.txt", "w") as file:
            pass
        # Tạo đối tượng cap để truy cập webcan
        cap = cv2.VideoCapture(0)
        # Khởi tạo bộ phát hiện khuôn mặt
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Khởi tạo biến sampleNum để đếm số lượng ảnh đã chụp từ webcam
        sampleNum = 0
        while (True):
            # Đọc 1 khung hình từ webcam
            ret, frame = cap.read()
            # Chuyển đổi hình ảnh sang ảnh xám
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Phát hiện khuôn mặt trong ảnh
            fa = detector.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in fa:
                # Vẽ hình chữ nhật quanh khuôn mặt
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Kiểm tra thư mục dataset có tồn tại hay không, nếu không thì tạo thư mục
                if not os.path.exists('dataset'):
                    os.makedirs('dataset')
                sampleNum += 1
                # Lưu khu vực khuôn mặt đã phát hiện vào thư mục dataset
                cv2.imwrite('dataset/anh' + str(5) + '.' + str(sampleNum) + '.jpg', gray[y:y + h, x:x + w])
            cv2.imshow('frame', frame)
            time.sleep(0.1)
            print("Đã chụp thành công ảnh số ", sampleNum)
            if sampleNum > nImg-1:
                break;
        print("Nhập tên model ",i,": ")
        modelName = input()
        with open("infoModel.txt","a") as file:
            file.write(f"modelName = {modelName} - {i}\n")
        prompt = input("Bạn có muốn tiếp tục(y/n): ")
        if prompt == 'n':
            break;
    cap.release()
    cv2.destroyAllWindows()