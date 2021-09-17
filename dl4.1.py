"""
1.2
単語のベクトル化と分散表現
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical  # データ読み込み用
from tensorflow.keras.datasets import mnist  # データ読み込み用
import keras
from tensorflow.python.ops.gen_array_ops import quantized_instance_norm
keras.preprocessing.text.Tokenizer(
    num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, split=" ", char_level=False)

"""
引数は次の通りです。

num_words：利用する単語の最大数（指定するとデータセット中の頻度上位num_wordsの単語のみ使用）．
filters：句読点などフィルタする文字のリスト
lower：テキストを小文字に強制するか
split：単語を分割するセパレータ
char_level：文字ごとに分割・数値化するか
主なメソッドは次の通りです。

fit_on_texts(texts)：入力＝学習に使う文章のリスト、出力＝なし
texts_to_sequences(texts)：入力＝数値化する文章のリスト、出力＝数値化された文章のリスト
"""

"""
分散表現
"""
keras.layers.embeddings.Embedding(input_dim, out_dim,
                                  embeddings_initializer='uniform',
                                  embeddings_regularizer=None, activity_regularizer=None,
                                  enbeddings_constraint=None,
                                  mask_zero=False, input_length=None)

"""
引数は次の通りです。

input_dim: 単語数（＝入力データの最大インデックス + 1）
output_dim: 出力次元（何次元に圧縮するか）
embeddings_{initializer, regularizer, constraint}: embeddings行列のInitializers, Regularizers, Constraints
mask_zero: 入力系列中の0をパディング（系列の長さを統一するために追加される無意味な要素）と解釈し、無視するか
input_length: 入力の系列長
他にも分散表現を実現する手法として、俗にword2vecと呼ばれるものがあります。
"""

"""
1.3 系列変換モデル
（Seq2Seqモデル）

符号化器（左3ユニット）：入力系列を受け取って抽象化します
復号化器（右5ユニット）：抽象化された入力系列を加味しつつ、真の出力系列を元に各々1つ先の単語を出力します
encoder-decoder model
"""

"""
seq2seq model
1. 符号化器Embeddingレイヤー：特徴量変換（入力系列のone_hot表現→埋め込み表現）
2. 符号化器再帰レイヤー：入力系列を"抽象化"（最終的な隠れ状態ベクトルの取得が目的、符号化器の途中の出力系列には興味がない）
3. 復号化器Embeddingレイヤー：特徴量変換（(5で生成された)直前の出力単語のone_hot表現→埋め込み表現）
4. 復号化器再帰レイヤー：抽象化した入力系列を加味しながら（状態ベクトルの初期値として使う）、現在の単語の1つ先の単語を出力
5. 復号化器出力レイヤー：復号化器再帰レイヤーの出力系列をもとにして目的の出力系列に変換する（隠れ状態ベクトル表現→one-hot表現）

RNN言語モデルで符号化器と復号化器の骨格を構成し、入力や出力との間をEmbeddingレイヤー（&Denseレイヤー）で取り持っている

再帰レイヤーにはRNNやLSTMのほかにもGRUなどがありますし、単方向か双方向か、何層積み重ねるかなど幅広い選択肢があり、工夫が求められる部分
こうしたSeq2Seqモデルの作成に当たっては、再帰レイヤーで①隠れ状態ベクトルが取得でき、②出力系列の取得ができる必要がありあます。

具体的には、

①隠れ状態ベクトル（LSTMの$c_t,h_t$に相当）を取得：引数にreturn_state=Trueを指定
②出力系列を取得：引数にreturn_sequences=Trueを指定
とすればよく、LSTMレイヤーを生成する際の引数として指定します。
"""

"""
1.4 functional API

Seq2SeqモデルはこれまでのSequentialクラスによるモデル構築では実現できません。

Sequentialクラスを用いる場合はadd関数を使用して簡単にモデルを構築可能である一方で、途中に分岐や合流があるような複雑なモデルは作成できません。

こうしたより複雑なモデルの構築方法がFunctional APIです。Seq2Seqモデルもこちらの方法によって実装可能になります。

このFunctional APIの実装上の特徴は、

Inputレイヤーから構築を始める
各レイヤーの返り値（テンソル）を次のレイヤーの入力として順々に構築していく
keras.models.Modelクラスに入力と出力を指定することでモデルを生成
といった点が挙げられます。
"""

# Input layer からスタート(返り値はテンソル)
inputs = Input(shape=(784, ))

# レイヤークラスのインスタンスはテンソルを引数に取れる（返り値はテンソル）
x = Dense(128, activation='relu')(inputs)  # InputレイヤーとDenseレイヤー(1層目)を接続
x = Dense(64, activation='relu')(x)  # Denseレイヤー(1層目)とDenseレイヤー(2層目)を接続
output_layer = Dense(10, activation='softmax')  # レイヤーのインスタンス化を切り分けることももちろん可能

# (別のモデル構成時にこのレイヤーを指定・再利用することも可能になる)
predictions = output_layer(x)  # Denseレイヤー(2層目)とDenseレイヤー(3層目)を接続

# Modelクラスを作成（入力テンソルと出力テンソルを指定すればよい）
# これで、「(784,)のInputを持つDense3層」構成のモデルが指定される
model = Model(inputs=inputs, outputs=predictions)

# 以降はSequentialと同じ
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train)

# check quiz
# 3
# 1
# 2
# 3

