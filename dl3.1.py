# RNN関数の中身
keras.layers.SimpleRNN(units, activaton='tanh', use_bias=True, kernel_initializer='glorot_unifirm', recurrent_initializer='orthogonal', \
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constriant=None, bias_constraint=None, \
        dropout=0.0, recurrent_dropout=0.0, return_sequence= False, return_state= False, \
            go_backward=False, stateful=False, unroll=False)

"""
引数は次の通りです。

units：出力次元（上図$o_t$の次元）
activation：活性化関数
use_bias：バイアスベクトル（$Ux_t+Ws_{t-1}$に付け加えるベクトル）を使用するか
{kernel,recurrent,bias}_initializer：各パラメータの初期化法（kernelは上図$U$, recurrentは上図$W$を指す）
{kernel,recurrent,bias,activity}_regularizer：各パラメータの正則化（activityは出力=activationを指す）
{kernel,recurrent,bias}_constraint：各パラメータに課す制約
dropout：入力についてのdropoutの比率
recurrent_dropout：再帰についてのdropoutの比率（上図横矢印に対して適用するdropout）
return_sequences: Falseなら出力としては系列の最後の出力のみ（$o_T$のみ）を返す、Trueなら出力として完全な系列（$o_1,o_2,\ldots,o_T$）を返す
return_state: Trueのときは出力とともに，最後の状態（$s_T$）を返す
go_backwards: Trueのときは入力系列を後ろから処理する（出力も逆順に）
stateful: Trueのときは、前バッチの各サンプルに対する最後の状態を、次のバッチのサンプルに対する初期状態として引き継ぐ
unroll: （高速化のためのオプション）Trueのときは再帰が展開され高速化されるが、よりメモリに負荷がかかる（短い系列にのみ適する）

"""