# 畳み込みニューラルーラルネットワーク CNN

#CNN基礎
import os

import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt 

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, \
    Input, Activation, add, Add, Dropout, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot

random_state = 42

#全結合層 「全ての入力値」から「全ての出力」への結合をもつ層の事
"""
CNNは Convolution(畳み込み)層 と Pooling(プーリング)層 と呼ばれる層を組合させて構築していきます。
(出力に近い層は全結合層を組み合わせることが多く、全結合層も使用します)

全結合層では画素同士の位置関係の情報を排除(ベクトルに落とし込む)していたのに対し、
畳み込み層では画素同士の位置情報を保持したまま扱うことで、結合を疎にすることを可能にしています。

全結合層では層間で1つのパラメータ（重み）は1度だけ使われますが、
畳み込み層では入力のすべての位置で同じパラメータ（重み）を使用します。(重み共有)

"""
#不変生
"""
さらに獲得した特徴の歪みやずれにたいしての頑強性をあげるため、小さな領域での統計量 (Max、Mean) などを取ります。
また画像サイズを小さくする役割もあります。プーリング層に対応します。

"""

#ネットワークの構成
"""
基本的には
畳み込み層->プーリング層->畳み込み層->プーリング層->...
と畳み込みとプーリングを繰り返していくのが基本になります。
全結合層は位置情報を失うため、ネットワークの最後でのみ使います。
"""

# Convolution(畳み込み)層
"""
畳み込み層における畳込みとは、
入力にたいしてフィルターを掛けた (畳み込んだ) ときに得られる値のことです。

畳み込み層ではある領域においてのフィルターに対する類似度のようなものを計算している

CNNでの各層の入力は実際には(縦のピクセル数)x(横のピクセル数)x(フィルター数)の3階テンソルとなります。 
それに合わせてフィルターも3階テンソル（縦*横*1(そのフィルター自体)）となりますが、
畳み込みの考え方自体は同じです。

入力画像が10x10x3(合計300ピクセル)の場合を考えてみます。

全結合層でユニット数300の隠れ層に繋ぐ場合、
パラメータ数は300x300+300（重みのはず？）=90300となります。

畳み込み層で考えてみると、5x5x3のフィルターを100枚用いた場合、
5x5x3x100+100(重み？のはず)=7600となり、全結合層の約12分の1のパラメータサイズとなります。

全結合層では入力画像のサイズに比例してパラメータ数が増えるのに対し畳み込み層では増えないので、
パラメータ削減の効果は入力画像が大きくなるにつれて大きくなります。→増えるのはプラスooのところだけ，つまりフィルターの枚数だけ

"""

#パディング
"""
特徴マップが縮小してしまうのを防ぐために、入力の両端に対して0などの値をくっつけることをします。 
これをパディングと言います。

慣例的に、何もくっつけないパディングを Valid、
入力と出力のサイズが変わらないようにするパディングを Same と呼びます。

"""

#kerasのおける実装
"""
Kerasにおいて畳み込み層を設定するには
keras.layers.Conv2Dを使用します。
"""
#サンプル画像 5*5
sample_image = np.array([[1, 1, 1, 0, 0],\
                        [0, 1, 1, 1, 0],\
                        [0, 0, 1, 1, 1], \
                        [0, 0, 1, 1, 0], \
                        [0, 1, 1, 0, 0]])\
                        .astype('float32').reshape(1, 5, 5, 1)

#フィルタ
W = np.array([1, 0, 1], [0, 1, 0], [1, 0, 1]).astype('float32').reshape(3, 3, 1, 1)

model= Sequential()

model.add(Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='valid', input_shape=(5, 5, 1), use_bias=False))

model.predict(sample_image).reshape(3, 3)

#Pooling層
"""
小さな領域に対して統計量 (Max、Mean) を取ることで、
位置のズレなどに対して頑強な特徴抽出を行います。
"""

#サンプル画像

sample_image = np.array([[1, 1, 2, 4],
                         [5, 6, 7, 8],
                         [3, 2, 1, 0],
                         [1, 2, 3, 4]]
                        ).astype("float32").reshape(1, 4, 4, 1)

model = Sequential()

model.add(MaxPooling2D(pool_size=(2, 2), strides=None,
                       padding='valid', input_shape=(4, 4, 1)))

model.predict(sample_image).reshape(2, 2)

