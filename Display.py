import cv2
import tensorflow as tf
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
mode=2
if mode == 1:
    def capture_and_save_image(output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(0) == 13:
            image_path = os.path.join(output_dir, 'captured_image.jpg')

            cv2.imwrite(image_path, frame)
            print("Đã lưu ảnh thành công!")

        cap.release()
        cv2.destroyAllWindows()


    capture_and_save_image('test')
    filename = 'test/captured_image.jpg'

if mode == 2:
    root = Tk()
    root.withdraw()
    parent_dir = askopenfilename()
    filepath = os.path.relpath(parent_dir, os.path.dirname(parent_dir))
    filename ='test/'+filepath


image = cv2.imread(filename)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
# save_model = tf.keras.models.load_model("khuonmat.h5")
save_model = tf.keras.models.load_model("khuonmat.keras")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fa = face_cascade.detectMultiScale(gray, 1.1, 5)
fontface = cv2.FONT_HERSHEY_SIMPLEX
for (x, y, w, h) in fa:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(src=roi_gray, dsize=(100,100))
    roi_gray = roi_gray.reshape((100,100,1))
    roi_gray = np.array(roi_gray)
    result = save_model.predict(np.array([roi_gray]))
    final = np.argmax(result)
    if final == 0:
        cv2.putText(image, "Duong",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
    if final == 1:
        cv2.putText(image, "Dien",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
    if final == 2:
        cv2.putText(image, "Tran Thanh",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
    if final == 3:
        cv2.putText(image, "Truong Giang",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
    if final == 4:
        cv2.putText(image, "Quang Hai",(x+10,y+h+ 30), fontface, 1, (0,255,0),2)
cv2.imshow('trainning',image)
print(result)
cv2.waitKey(0)
cv2.destroyAllWindows()
