import cv2
import tensorflow as tf
import numpy as np

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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Tiền xử lý khuôn mặt
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(src=roi_gray, dsize=(100,100))
        roi_gray = roi_gray.reshape((100,100,1))
        roi_gray = np.array(roi_gray)

        # Dự đoán khuôn mặt
        result = save_model.predict(np.array([roi_gray]))
        final = np.argmax(result)

        # Hiển thị kết quả dự đoán lên khung hình
        if final == 0:
            cv2.putText(frame, "Duong", (x+10, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif final == 1:
            cv2.putText(frame, "Son Tung", (x+10, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif final == 2:
            cv2.putText(frame, "Lai Van Sam", (x+10, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif final == 3:
            cv2.putText(frame, "My Tam", (x+10, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif final == 4:
            cv2.putText(frame, "Xuan Hinh", (x+10, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị khung hình kết quả
    cv2.imshow('Real-time Face Detection', frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()