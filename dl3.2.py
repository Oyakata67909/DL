from tensorflow.python.keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from tensorflow.keras.layers import Dense, Activation, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.io import arff
import numpy as np

dataset, meta = arff.loadarff('data/ECG5000_TRAIN.arff')
test, t_mata = arff.loadarff('data/ECG5000_TEST.arff')

ds = np.asarray(dataset.tolist(), dtype=np.float32)
x_dataset = ds[:, :140]
y_dataset = np.asarray(ds[:, -1].tolist(), dtype=np.int8)-1

x_train, x_test, y_train, y_test = train_test_split(
    x_dataset[:, :, np.newaxis], to_categorical(y_dataset), test_size=0.2, random_state=42)


#モデル構築

hid_dim = 10

#SimpleRNNにDenseを接続して，分類
model = Sequential()

model.add(SimpleRNN(hid_dim, input_shape=x_train.shape[1:]))
#input_shape=(系列長T, x_tの次元), output_shape=(units(=hid_dim),)
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', \
              optimizer='adam', metrics=['accuracy'])


#モデルの学習
model.fit(x_train, y_train, epochs=50, batch_size=100, \
          verbose=2, validation_split=0.2)

#モデルによる分類制度の評価
score = model.evaluate(x_test, y_test, verbose=0)
print('test_loss:', score[0])
print('test_acc:', score[1])

#モデルの可視化

SVG(model_to_dot(model).create(prog='dot', format='svg'))
