"""
3.1 clippingによる勾配爆発の対処 
gradient clipping
勾配の大きさそのものを制限してしまうという手法が有効です
"""

from tensorflow.keras import optimizers
from tensorflow.python.keras.layers.recurrent import LSTM

sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
# clipnormは勾配の2乗ノルムの最大値を制限する
sgd2 = optimizers.SGD(lr=0.01, clipvalue=1.)
# clipvalueは勾配の"要素"の絶対値の大きさを制限する
# clipnormの方が勾配の方向を変えないという利点があるが、経験的にはどちらの振る舞いも大差ない

ada = optimizers.Adagrad(lr=0.01, clipnorm=1.)
# SGDに限らずすべてのoptimizerで指定可能
#model.compile(loss='mean_squared_error', optimizer=sgd)

""""
ショートカットとゲートによる勾配消失への対処
ショートカット
各層の出力にその層への入力だったものも加えてしまうという手法
（RNNに限らず、一般に第l層目の出力をo^{(l)}=f^{(l)}(o^{(l-1)})+o^{(l-1)}$とすることで可能）
一見するとこれが勾配消失に有用なのか疑わしいですが、このショートカットよりこの層の勾配が「1+元の勾配」と増加します。

そのため、勾配の積が積み重なる、入口に近い層でも勾配が消失することなく、パラメータ更新が可能になります。
"""
"""
ゲート
ショートカットの一般化として重み付き和を考えるもの
（つまり、f^{(l)}(o^{(l-1)})$と$o^{(l-1)}$に各々係数をかけたうえで足し合わせる）

ゲートの係数も学習することにより、前の層からの情報と現在の層による情報の重みを最適に調整できる、
つまり以前の層からの情報の忘却度合いを丁度よく決められます。
"""

"""
LSTM
RNNの中でも長い系列に強い（系列内の長期的な相互依存性をモデル化可能）モデルであるという特性
"""

keras.layers.LSTM(units, activaton='tanh', recurrent_ativation='hard_sigmoid', use_bias=True,
                  kernel_initializer='glorot_uniform',
                  recurrent_initializer='orthogonal', bias_initializer='zeros',
                  units_forget_bias=True,
                  kernel_regularizer=None, recurrent_regularizer=None,
                  bias_regularizer=None, activity_regularizer=None,
                  kernel_constraint=None, recurrent_constraint=None,
                  bias_constraint=None,
                  dropout=0.0, recurrent_dropout=0.0, implementation=1,
                  recurrent_sequenes=False, retrn_sstate=False,
                  go_backwards=False, stateful=False, unroll=False)

"""
引数は次の通りです。

units：ユニット数（系列長$T$）
activation：活性化関数
recurrent_activation：ゲート係数の計算で使用する活性化関数
use_bias：バイアスベクトル（$Wx_t+Rh_{t-1}$に付け加えるベクトル）を使用するか
{kernel,recurrent,bias}_initializer：各パラメータの初期化法（kernelは$W$, recurrentは$R$を指す）
unit_forget_bias：忘却ゲートを1に初期化
{kernel,recurrent,bias,activity}_regularizer：各パラメータの正則化（activityは出力=activationを指す）
{kernel,recurrent,bias}_constraint：各パラメータに課す制約
dropout：入力についてのdropoutの比率（$W$に対するdropout）
recurrent_dropout：再帰についてのdropoutの比率（$R$に対するdropout）
return_sequences: Falseなら出力としては系列の最後の出力のみ（$o_T$のみ）を返す、Trueなら出力として完全な系列（$o_1,o_2,\ldots,o_T$）を返す
return_state: Trueのときは出力とともに，最後の状態（$s_T$）を返す
go_backwards: Trueのときは入力系列を後ろから処理する（出力も逆順に）
stateful: Trueのときは、前バッチの各サンプルに対する最後の状態を、次のバッチのサンプルに対する初期状態として引き継ぐ
unroll: （高速化のためのオプション）Trueのときは再帰が展開され高速化されるが、よりメモリに負荷がかかる（短い系列にのみ適する）
"""

"""
GRU Gated Recurrent Unit
ゲートの考え方を利用しながら、隠れ状態ベクトル$h_t$のみに長期の情報も集約したモデル
"""
keras.layers.GRU(units, activation='tanh', recurrent_activaton='hard_sigmoid', \
                    use_bias=True, 
                    kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
                    bias_initializer='zeros', kernel_regularizer=None, 
                    recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                    dropout=0.0, recurrent_dropout=0.0, implementation=1, 
                    return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, 
                    reset_after=False)

"""
代表的な引数は次の通りです。

units：ユニット数（系列長$T$）
activation：活性化関数
recurrent_activation：内部で使用する活性化関数
use_bias：バイアスベクトル（$Ux_t+Wh_{t-1}$に付け加えるベクトル）を使用するか
{kernel,recurrent,bias}_initializer：各パラメータの初期化法（kernelは上図$U$, recurrentは上図$W$を指す）
{kernel,recurrent,bias,activity}_regularizer：各パラメータの正則化（activityは出力=activationを指す）
{kernel,recurrent,bias}_constraint：各パラメータに課す制約
dropout：入力についてのdropoutの比率
recurrent_dropout：再帰についてのdropoutの比率（上図横矢印に対して適用するdropout）
return_sequences: Falseなら出力としては系列の最後の出力のみ（$o_T$のみ）を返す、Trueなら出力として完全な系列（$o_1,o_2,\ldots,o_T$）を返す
return_state: Trueのときは出力とともに，最後の状態（$h_T$）を返す
go_backwards: Trueのときは入力系列を後ろから処理する（出力も逆順に）
stateful: Trueのときは、前バッチの各サンプルに対する最後の状態を、次のバッチのサンプルに対する初期状態として引き継ぐ
unroll: （高速化のためのオプション）Trueのときは再帰が展開され高速化されるが、よりメモリに負荷がかかる（短い系列にのみ適する）
"""
