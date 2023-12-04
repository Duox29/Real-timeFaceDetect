import cv2
import numpy as np
from PIL import Image
data = [] 
label = []
# Đọc file config.txt, lấy ra giá trị của nModel và nImg
with open("config.txt", "r") as file:
    content = file.readlines()
variables = {}
for line in content:
    parts = line.strip().split("=")
    if len(parts) == 2:
        variable_name = parts[0].strip()
        variable_value = parts[1].strip()
        variables[variable_name] = int(variable_value)

nModel = variables.get("nModel")
nImg = variables.get("nImg")


for j in range (1,nModel+1):
  for i in range (1,nImg+1):
    filename = './dataset/anh'+ str(j) + '.'  + str(i) + '.jpg'
    # Đọc hình ảnh từ tệp tin được chỉ định bởi filename
    Img = cv2.imread(filename)
    # Chuyển đổi hình ảnh Img sang ảnh xám
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    # Thay đổi kích thước hình ảnh Img thành kích thước (100, 100)
    Img = cv2.resize(src=Img, dsize=(100,100))
    # Chuyển đổi hình ảnh Img thành một mảng NumPy
    Img = np.array(Img)
    # Thêm hình ảnh Img vào danh sách data
    data.append(Img)
    label.append(j-1)
# Chuyển đổi danh sách data thành một mảng NumPy
data1 = np.array(data)
# Chuyển đổi danh sách label thành một mảng NumPy
label = np.array(label)
# Thay đổi hình dạng của mảng data1 thành (số lượng mô hình * số lượng hình ảnh, 100, 100, 1)
data1 = data1.reshape((nModel*nImg,100,100,1))
# Chuẩn hóa giá trị của mảng data1 bằng cách chia mỗi phần tử cho 255 và lưu kết quả vào biến X_train
X_train = data1/255
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
trainY =lb.fit_transform(label)
# Import lớp Model để xây dựng mô hình mạng neural
from tensorflow.keras.models import Model
# Import lớp Sequential để tạo một mô hình mạng neural tuần tự
from tensorflow.keras.models import Sequential

# Import các lớp AveragePooling2D, MaxPooling2D, Conv2D, Activation, Flatten, Input, Dense, concatenate để xy dựng kiến trúc mạng neural
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
# Khởi tạo một đối tượng Sequential để tạo một mô hình mạng neural tuần tự
Model = Sequential()
# Định nghĩa kích thước đầu vào của mô hình
shape = (100,100, 1)
# Khối B2 : thêm 1 lớp Conv2D và 1 lớp Activation với hàm kích hoạt relu
Model.add(Conv2D(32,(3,3),padding="same",input_shape=shape))
Model.add(Activation("relu"))
# Khối B3 : thêm 1 lớp Conv2D và 1 lớp Activation với hàm kích hoạt relu
Model.add(Conv2D(32,(3,3), padding="same"))
Model.add(Activation("relu"))
# Khối B4 : thêm 1 lớp maxpooling2D
Model.add(MaxPooling2D(pool_size=(2,2)))
# Khối B5 : thêm 1 lớp Conv2D và 1 lớp Activation với hàm kích hoạt relu
Model.add(Conv2D(64,(3,3), padding="same"))
Model.add(Activation("relu"))
# Khối B6 : thêm 1 lớp maxpooling2D
Model.add(MaxPooling2D(pool_size=(2,2)))
# Khối b7 : Thêm các lớp Flatten, Dense, Activation
Model.add(Flatten())
Model.add(Dense(512))
Model.add(Activation("relu"))
Model.add(Dense(nModel))
Model.add(Activation("softmax"))
# In ra tóm tắt của kiến trúc mô hình
Model.summary()

# Cấu hình quá trình huấn luyện của mô hình. Chúng ta chỉ định hàm mất mát (categorical_crossentropy), thuật toán tối ưu hóa (adam) và các độ đo để đánh giá hiệu suất của mô hình (accuracy).
Model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("start training")

# Huấn luyện mô hình trên dữ liệu huấn luyện
Model.fit(X_train,trainY,batch_size=nModel,epochs=10)
# Lưu mô hình đã huấn luyện
Model.save("khuonmat.keras")
