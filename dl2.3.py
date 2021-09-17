# %matplotlib inline

import os

import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, add, Add, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot

random_state = 42

#data augmentation

#オリジナル
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255
y_train = np.eye(10)[y_train.astype('int32').flatten()]

x_test = x_test.astype('float32') / 255
y_test = np.eye(10)[y_test.astype('int32').flatten()]

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=10000)

from tensorflow.keras.preprocessing.image import ImageDaraGenerator

fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0,
                    top=0.5, hspace=0.05, wspace=0.05)

for i in range(5):
    ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])
    ax.imshow(x_train[i])


#左右にずらす
datagen = ImageDaraGenerator(width_shift_range=0.4)

datagen_fit(x_train)

fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0,
                    top=0.5, hspace=0.05, wspace=0.05)

for x_batch, y_batch, iin datagen.flow(x_train, y_train, batch_size=9, shuffle=False):
    for i in range(5):
        ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(x_batch[i])
    break

#上下にずらす
datagen = ImageDataGenerator(height_shift_range=0.4)

datagen.fit(x_train)

fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0,
                    top=0.5, hspace=0.05, wspace=0.05)

for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9, shuffle=False):
    for i in range(5):
        ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(x_batch[i])
    break


#左右反転
datagen = ImageDataGenerator(horizontal_flip=True)

datagen.fit(x_train)

fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0,
                    top=0.5, hspace=0.05, wspace=0.05)

for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9, shuffle=False):
    for i in range(5):
        ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(x_batch[i])
    break

#回転
datagen = ImageDataGenerator(rotation_range=30)

datagen.fit(x_train)

fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0,
                    top=0.5, hspace=0.05, wspace=0.05)

for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9, shuffle=False):
    for i in range(5):
        ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(x_batch[i])
    break

#画像データの正規化

#Global Contrast Normalization GCN
"""
画像ごとにピクセルの値を平均0、分散1に正規化します。
"""
#可視化用にrangeを[0, 1]に修正
def normalizie(x):
    max_x = np.max(x, axis(0, 1), keepdims=True)
    min_x = np,min(x, axis(0, 1), keepdims=True)
    return (x - min_x) / (max_x - min_x)

from Tensorflow.keras.preprocessing.image import ImageDataGenerator 
gcn_whitening = ImageDataGrnetator(samplewise_center=True, samplewise_std_normailization=True)
gcn_whitening.fit(x_train)

fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0,
                    top=0.5, hspace=0.05, wspace=0.05)

for x_batch, y_batch in gcn_whitening.flow(x_train, y_train, batch_size=9, shuffle=False):
    for i in range(5):
        ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])
        ax.imshow(normalize(x_batch[i]))
    break

#Zero-phase Component Analysis (ZCA) Whitening
"""
入力の各要素間の相関をゼロ(白色化)にします。

PCAを利用して共分散行列を単位行列化 (分散1、共分散0) したのち、元の空間に戻します。
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

zca_whitening = ImageDataGenerator(zca_whitening=True)
zca_whitening.fit(x_train)

fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0,
                    top=0.5, hspace=0.05, wspace=0.05)

for x_batch, y_batch in zca_whitening.flow(x_train, y_train, batch_size=9, shuffle=False):
    for i in range(5):
        ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(normalize(x_batch[i]))
    break

#Batch Normalization
"""
入力画像に対して正規化をおこなったように、各隠れ層の入力の分布も安定させたいと考えます。 
これがない場合、深層のネットワークにおいては層を重ねるほど分布が不安定になります。 
特に深層学習において、学習データと検証(評価)データ間で各層の分布が変わってしまう現象は 内部共変量シフト と呼ばれます.

Batch Normalization では各層の出力を正規化することでこの問題の解決を試みます。

"""
from keras.layers.normalization import BatchNormalization

model.add(BatchNormalization())

#Skip Connection (residual network)
"""
Skip Connection (Residual Network) は、層を飛び越えた結合をつくることで勾配消失問題を解消しようとする手法です。

重みを掛けずに値を渡し、逆伝播の際に出力層側から誤差をそのまま入力層側に伝えることで、
勾配が消失しにくいネットワーク構造をつくっていると考えることができます。
"""
def resblock(x, filters=64, kernel_size(3, 3)):
    x_ = Conv2D(filters, kernel_size, padding='same')(x) 
    x_ = BatchNormalizaiton()(x_)
    x_ = Conv2D(filters, kernel_size, padding='same')(x_)
    x = Add()([x_, x])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

from keras import backed as K 

inputs = Input(shape=(32, 32, 3))

x = Conv2D(64, kernel_size=(5, 5), papdding='same', activation='relu')(inputs)
x = resblock(x)
x = resblock(x)
y = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=y)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=3, verbose=1)

"""
verbose
：ログ出力の指定。「0」だとログが出ないの設定。
"""

#学習済みネットワークの利用
"""
CNNの入力層付近では局所的な特徴 (エッジなど) を抽出しています。これらの特徴は多くの画像データに共通しています。

このことを利用し、あらかじめ別の大規模なデータセットで十分に学習されたネットワークの出力層以外の重みを初期値として活用することを考えます。

Kerasでは事前に大規模なデータセット (ImageNet) に対して学習されたモデルがロードできるようになっています。
"""
model = VGG(weight='imagenet')
weights = [com.get_weights() for com in model.layers[1:]]

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
                 input_shape=(32, 32, 3)))  # 32x32x3 -> 30x30x64
# 30x30x64 -> 28x28x64
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # 28x28x64 -> 14x14x64

# 14x14x64 -> 12x12x128
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# 12x12x128 -> 10x10x128
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # 10x10x128 -> 5x5x128

model.add(Flatten())
model.add(Dense(10, activation='softmax'))

"""
上記のようにモデルを構築したのち、ロードした重みを各層に設定します。
"""

#weightの初期化
model.layers[0].set_weights(weights[0])
model.layers[1].set_weights(weights[1])
model.layers[3].set_weights(weights[3])
model.layars[4].set_weights(weights[4])

#学習させたモデルの保存，再利用
"""
学習させたモデルはhdf5形式で保存することができます。
"""
model.save('./mnist_cnn.h5')

#再利用
"""
保存されたモデルは以下のようにロードして再利用することができます。
"""
from tensorflow.keras.models import load_model
model = load_model('./mnist_cnn.h5')
y_pred = mode.pridict(x_valid)


