import cv2
import tensorflow as tf
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
# Đọc file infoModel lấy thông tin về các mẫu và gán vào list
model_names = []
with open("infoModel.txt", "r") as file:
    for line in file:
        if line.startswith("modelName"):
            model_name = line.split("=")[1].strip().split(" - ")[0]
            model_names.append(model_name)

print("     Menu")
print(" 1.Nhận diện từ ảnh")
print(" 2.Nhận diện từ webcam")
print(" 3.Thoát")
mode = int(input("Nhập lựa chọn: "))
# Nhận diện trên hình ảnh
if mode == 1:
    # Mở cửa sổ File Explorer để chọn ảnh
    root = Tk()
    root.withdraw()
    parent_dir = askopenfilename()
    filepath = os.path.relpath(parent_dir, os.path.dirname(parent_dir))
    filename = 'test/' + filepath
    image = cv2.imread(filename)

    # Load pre-trained model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    save_model = tf.keras.models.load_model("khuonmat.keras")

    # Chuyển ảnh sang grayscale để nhận diện khuôn mặt
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong hình
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Vẽ hình chữ nhật và dự đoán khuôn mặt
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Tiền xử lý khuôn mặt
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(src=roi_gray, dsize=(100, 100))
        roi_gray = roi_gray.reshape((100, 100, 1))
        roi_gray = np.array(roi_gray)

        # Dự đoán khuôn mặt
        result = save_model.predict(np.array([roi_gray]))
        final = np.argmax(result)
        # Đặt tên cho khuôn mặt
        if final == final:
            cv2.putText(image, model_names[final], (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Hiển thị khung hình kết quả
    cv2.imshow('trainning', image)
    print(result)
    cv2.waitKey(0)
# Nhận diện trên video
elif mode == 2:

    # Load pre-trained model
    save_model = tf.keras.models.load_model("khuonmat.keras")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Khởi tạo webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Đọc từng khung hình từ webcam
        ret, frame = cap.read()

        # Chuyển ảnh sang grayscale để nhận diện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt trong khung hình
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # Vẽ hình chữ nhật và dự đoán khuôn mặt
        for (x, y, w, h) in faces:
            # Vẽ hình chữ nhật xung quanh khuôn mặt
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Tiền xử lý khuôn mặt
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(src=roi_gray, dsize=(100, 100))
            roi_gray = roi_gray.reshape((100, 100, 1))
            roi_gray = np.array(roi_gray)

            # Dự đoán khuôn mặt
            result = save_model.predict(np.array([roi_gray]))
            final = np.argmax(result)
            # Đặt tên cho khuôn mặt
            if final == final:
                cv2.putText(frame, model_names[final], (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị khung hình kết quả
        cv2.imshow('Real-time Face Detection', frame)

        # Thoát khỏi vòng lặp nếu nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
elif mode == 3:
    pass
