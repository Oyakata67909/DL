%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

data_y = list(open('._data_y_train.txt'))[:2]
data_x = np.load('./data/x/train.npy')[:2]

plt.figure(fignize=(12, 8))
for i, (x, y) in enumerate(zip(data_x, data_y)):
    plt.subplot(1, 2, i+1)
    plt.imshow(x)
    plt.title(y)
    plt.axis('off')

plt.show()

#ネットワーク構成
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input

x = Input(shape=(224, 224, 3))
model = VGG16(\
    weighs='imagenet'
    include_top=False
    input_tensor0=x)

model.summary()



