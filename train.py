import cv2
import numpy as np
from PIL import Image
data = [] 
label = []

with open("config.txt", "r") as file:
    content = file.readlines()

# Gán giá trị cho các biến
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
    Img = cv2.imread(filename)
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    Img = cv2.resize(src=Img, dsize=(100,100))
    Img = np.array(Img)
    data.append(Img)
    label.append(j-1)
data1 = np.array(data)
label = np.array(label)
data1 = data1.reshape((nModel*nImg,100,100,1))
X_train = data1/255
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
trainY =lb.fit_transform(label)
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
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
Model = Sequential()
shape = (100,100, 1)
Model.add(Conv2D(32,(3,3),padding="same",input_shape=shape))
Model.add(Activation("relu"))
Model.add(Conv2D(32,(3,3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Conv2D(64,(3,3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Flatten())
Model.add(Dense(512))
Model.add(Activation("relu"))
Model.add(Dense(5))
Model.add(Activation("softmax"))
Model.summary()
Model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("start training")
Model.fit(X_train,trainY,batch_size=5,epochs=15)
Model.save("khuonmat.keras")
