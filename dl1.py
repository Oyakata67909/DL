from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)

#各mnist画像の上に（タイトルとして）対応するラベルを表示
for i in range(9):
    ax = fig.add_subplot(1, 9, i + 1, xticks=[], yticks=[])
    ax.set_title(str(y_train[i]))
    ax.imshow(x_train[i], cmap='gray')

"""
分類タスクの時の出力データはラベルですが、ラベルは数字としての大小には意味がないということです。

というのも、グループの名前として数字を割り振っているだけであるためです。こうした数字を名義尺度と呼びます。

機械学習のアルゴリズムでは数字の大小に意味があるものとして扱ってしまうため、名義尺度をうまく変換しなければなりません。

この名義尺度を変換する表現として使用されるのが、one-hot表現と呼ばれるものです。
このone-hot表現への変換を行ってくれる関数がKerasにはあります。

keras.utils.to_categorical関数がその関数です。
"""
from tensorflow.keras.utils import to_categorical
#入力画像を行列（28*28）からベクトル（長さ784）に変換
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

#名義尺度の値をone-shot表現に変更
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

"""
学習に使用するMLPのモデルを構築します。具体的には、どんなlayer（層）をどこに配置するか、また各layerのユニット数はいくつかを指定していきます。

このモデルを構築するための「容器」として機能するのが、keras.models.Sequentialクラスです。

この「容器」の中に、Sequential.add関数によってkeras.layersに定義されているlayerクラス（後で詳述）を積み重ねていくことでモデルの構築を行います。

layerをSequentialクラスに積み終えたら、最後にSequential.compile関数でモデルの学習処理について指定し、モデル構築は完了です。

compile関数では

optimizer（最適化手法）
loss（損失関数）
metrics（評価関数（任意））
を指定することになります。
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

#モデルの容器を作成
model = Sequential()

#容器へ各layer(Dense, Activation)を積み重ねていく（追加された順に配置されていく）
#最初のlayerはinput_shapeを指定して，入力するデータの次元を与える必要がある
model.add(Dense(units=256, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(units=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

#モデルの学習方法について指定しておく
model.compile(loss='categorical_crossentrophy', optimizer='sgd', metrics=['accuracy'])

"""
1.2.1で構築したモデルで実際に学習を行うには、Sequential.fit関数を用います。この関数は固定長のバッチで学習を行います。
"""

print(model.fit(x_train, y_train, batch_size=1000, epochs=10, verbose=1, validation_data=(x_test, y_test)))

"""
モデルの評価を行うには、Sequential.evaluate関数を用います。この関数は固定長のバッチごとに損失関数値または評価関数値を出力します。
"""
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""
1.2.2で学習させたモデルによって予測を行ってみましょう。Sequential.predict関数によって予測が行えます。
"""
classes = model.predict(x_test, batch_size=128)

from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot

SVG(model_yo_dot(model, show_shapes=True).create(prog='dot', format='svg'))

