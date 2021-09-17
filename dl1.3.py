#前処理
"""
学習の高速化や性能向上などのために入力データを変換する前処理を行うことがあります。
そのひとつに、データを学習において取り扱いやすいよう、
特定の範囲にデータが収まるように変換するスケーリングがあります。

他にも平均や分散を特定の値にする正規化,
データの各要素間の相関を取り除く白色化,
またデータ全体を考慮したバッチ正規化といった前処理もあります。

"""

# Momentumの実装例
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False))

# Adagradの実装例
from keras.optimizers import Adagrad
model.compile(loss='categorical_crossentropy', optimizer=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0))

# RMSPropの実装例
from keras.optimizers import RMSProp
model.compile(loss='categorical_crossentropy', optimizer=RMSProp(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0))
# rhoはどれだけ過去の勾配を重視するかを表し、通常はrho=0.9程度とすることが多い

"""
RMSpropによって学習率が不可逆的に悪化することを防ぐことができましたが、AdaGradの全体の学習率に鋭敏であるという性質はそのままです。

この全体の学習率への鋭敏性、つまり問題設定毎に適切な学習率が変化してしまうという問題は、

実は更新量と勾配の次元の不一致を学習率で調整していることによるものです。（ここでの次元は物理的な次元のことで、いわゆる単位に相当するものです）

そこで、AdaDeltaではそうした次元の不一致を加味して自動的に適切な学習率が設定されるようにしています。
"""

# Adadeltaの実装例
from keras.optimizers import Adadelta
model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0))
#Kerasの実装では一応学習率lrを設定できるようになっていますが、AdaDeltaの提案論文では学習率は自動的に決定されるものとしている上、
#Kerasの公式HPでもlrはデフォルトのままとすることを推奨しているため、学習率の設定は基本的に不要です。

"""
AdaDeltaとは異なるRMSpropの改良法としてAdamが挙げられます。
Adamでは、各方向への勾配の2乗に加えて勾配自身も、指数移動平均による推定値に置き換えています。

これにより、ある種Momentumと似た効果が期待できます。

"""

# Adamの実装例
from keras.optimizers import Adam
model.compile(loss='categorical_crossentrpy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, dacay=0.0))

#3.2.2
#活性化関数 actiavation
"""
勾配消失問題
文字通り、勾配（＝損失関数のパラメータ微分）が数値計算上極めて小さくなってしまい、学習が進まなくなってしまうという問題です。
誤差逆伝播法では、（ある層の勾配）＝（1層前の勾配）×（2層前の勾配）×・・・×（出力層の勾配）と積の形で勾配を求めるため、
途中の勾配が小さいと入力層付近の勾配はどんどん0に近づいていってしまうわけです。
"""

#sigmoid, tanh, ReLU関数の微分はそれぞれ以下のようになります
import numpy as np
def deriv_sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))** 2

def deriv_tanh(x):
    return 1 - np.tanh(x)** 2

def deriv_rulu(x):
    return 1 * (x > 0)


#sigmoidよりもtanh、reluのほうがより大きな値をとり、勾配消失しにくいこと。

#最近の論文でもreluもしくはその派生形を用いているものが多い

#3.2.3
#初期化 initializer

"""
勾配に関するテクニックの3つめはパラメータの初期化についてです。

各層のパラメータは0を中心とした乱数で初期化しますが、大きすぎる値で初期化すれば学習の初期段階での勾配が過大になり、

逆に小さすぎる値だと勾配自体も過小になってしまい、いずれにしても学習はうまく進みません。

そこで、初期化にあたっては、その値のスケール（分散）を適切に設定する必要があります。

このパラメータの初期化にあたって比較的頻繁に用いられる手法として、LeCunによる手法、Glorotによる手法、Heによる手法が挙げられます。
"""

# LuCun's initializationの実装例
model.add(Dense(128, activation='relu', kernel_initializer='lucun_uniform'))
model.add(Dense(128, activation='relu', kernel_initializer='luncun_normal'))

#Glorotの初期化 = Xavierの初期化
#Glorot's initizlizationの初期化
model.add(Dense(128, activation='sigmoid', kernel_initializer='glorot_uniform'))
model.add(Dense(128, activation='sigmoid', kernel_initilizer='glorot_normal'))

#Heの初期化
#He's initialization
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))

#3.3
#過学習に対するテクニック

"""
過学習が発生する一つの理由は、MLPのモデルは特に多くのパラメータ・自由度を持つために、

訓練データに対して、その本質的な部分以上に統計的ばらつきまで含めて完全にフィットしようとしてしまうことにあります。

そこで過学習を回避するには、学習過程でいくつかのパラメータが自動的に機能しなくなると良いわけですが、これを実現するのが正則化です。

正則化はwをいじる，ドロップアウトは，中間層をランダム（特定の確率）で無視しながらフォワードプロパゲーションをする
"""

#L2正則化
from keras.layers import Dense
from keras import regularizers
model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))

#L1正則化
from keras.layers import Dense
from keras import regularizers
model.add(Dense(128, kernel_regularizer=regularizers.l1(0.01)))

#ElasticNet
from keras.layers import Dense
from keras import regularizers
model.add(Dense(128, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))

#3.32早期終了
model.fit(x=x_train, y=y_train, ..., callbacks=keras.callbacks.EarlyStopping(patience=0, verbose=1))

#3.3.3ドロップアウト
"""
これは近似的にアンサンブル法を実現するものになっています。

具体的には、ドロップアウトは入力の一部をランダムに0にして出力するlayerの一種です。要するに一部のユニットを取り除いた状況を再現します。

このユニットの除去を確率的に行い、一部のユニットが除去された部分ネットワークに対して学習することを繰り返すことで、

多数のモデルを同時に訓練することと同じ効果を再現しているわけです

"""

#ドロップアウトの実装
keras.layers.core.Dropout(rate, noise_shape=None, seed=None)

