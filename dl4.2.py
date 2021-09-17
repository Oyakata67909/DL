from numpy.lib.arraypad import pad
from numpy.testing._private.utils import break_cycles
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(file_path):
    tokenizer = Tokenizer(filters="")
    whole_texts = []
    for line in open(file_path, encoding='utf-8'):
        whole_texts.append("<s>" + line.strip() + "</s>")

    tokenizer.fit_on_texts(whole_texts)

    return tokenizer.texts_to_sequences(whole_texts), tokenizer

#読み込み＆Tokenizerによる数値化
x_train, tokenizer_en = load_data('data/train.en')
y_train, tokenizer_ja = load_data('data/train.ja')

en_vocab_size = len(tokenizer_en.word_index) + 1
ja_vocab_size = len(tokenizer_ja.word_index) + 1

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.02, random_state=42)

#パディング
x_train = pad_sequences(x_train, padding='post')
y_train = pad_sequences(y_train, padding='post')

seqX_len = len(x_train[0])
seqY_len = len(y_train[0])
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM

emb_dim = 256
hid_dim = 256

##符号器
#Input layer (返り血としてテンソルを受け取る)
encoder_inputs = Input(shape=(seqX_len,))

# モデルの層構成（手前の層の返り値テンソルを、次の接続したい層に別途引数として与える）
# InputレイヤーとEmbeddingレイヤーを接続（+Embeddingレイヤーのインスタンス化）
encoder_embedded = Embedding(en_vocab_size, emb_dim, mask_zero=True)(encoder_inputs)
# shape: (seqX_len,)->(seqX_len, emb_dim)
# EmbeddingレイヤーとLSTMレイヤーを接続（+LSTMレイヤーのインスタンス化）
_, *encoder_states = LSTM(hid_dim, return_state=True)(encoder_embedded)
# shape: (seqX_len, emb_dim)->(hid_dim, )
# このLSTMレイヤーの出力に関しては下記に補足あり

##複合化器
#input layer
decoder_inputs = Input(shape=(seqY_len, ))

# モデルの層構成（手前の層の返り値テンソルを、次の接続したい層に別途引数として与える）
# InputレイヤーとEmbeddingレイヤーを接続
decoder_embedding = Embedding(ja_vocab_size, emb_dim)
# 後で参照したいので、レイヤー自体を変数化
decoder_embedded = decoder_embedding(decoder_inputs)
# shape: (seqY_len,)->(seqY_len, emb_dim)
# EmbeddingレイヤーとLSTMレイヤーを接続（encoder_statesを初期状態として指定）
decoder_lstm = LSTM(hid_dim, return_sequences=True, return_state=True)
# 後で参照したいので、レイヤー自体を変数化
decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
# shape: (seqY_len, emb_dim)->(seqY_len, hid_dim)
# LSTMレイヤーとDenseレイヤーを接続
decoder_dense = Dense(ja_vocab_size, activation='softmax')
# 後で参照したいので、レイヤー自体を変数化
decoder_outputs = decoder_dense(decoder_outputs)
# shape: (seqY_len, hid_dim)->(seqY_len, ja_vocab_size)

#モデル構築
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
# 今回は、sparse_categorical_crossentropy（正解ラベルとしてone_hot表現のベクトルでなく数値を受け取るcategorical_crossentropy）を使用

#モデルの学習
import numpy as np
train_target = np.hstack((y_train[:, 1:], np.zeros((len(y_train), 1), dtype=np.int32)))

model.fit([x_train, y_train], np.expand_dims(train_target, -1), batch_size=128, epochs=15, verbose=2, validation_split=0.2)

#サンプリング用(生成用)のモデルを作成
#符号化器(学習時と同じ構成，学習したレイヤーを利用)
encoder_model = Model(encoder_inputs, encoder_states)

#復号化器
decoder_states_inputs = [Input(shape=(hid_dim)), Input(shape=(hid_dim, ))] #decoder_lstmの初期状態指定用(h_t, c_t)
decoder_inputs = Input(shape=(1, ))
decoder_embedded = decoder_embedding(decoder_inputs) #学習済みembeddingレイヤー利用
decoder_outputs, *decoder_states = decoder_lstm(decoder_embedded, initial_state=decoder_states_inputs) #学習済みlstmレイヤーを使用
decoder_outputs = decoder_dense(decoder_outputs) #学習済denseレイヤーを利用

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_sequence(input_seq, bos_eos, max_output_length = 1000):
    states_value = encoder_model  
    target_seq = np.array(bos_eos[0])  # bos_eos[0]="<s>"に対応するインデックス
    output_seq = bos_eos[0][:]

    while True:
        output_tokens, *states_value = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = [np.argmax(output_tokens[0, -1, :])]
        output_seq += sampled_token_index

        if (sampled_token_index == bos_eos[1] or len(output_seq) > max_output_length):
            break

        target_seq = np.array(sampled_token_index)
    
    return output_seq

detokenizer_en = dict(map(reversed, tokenizer_en.word_index.items()))
detokenizer_ja = dict(map(reversed, tokenizer_ja.word_index.items()))

text_no = 0
input_seq = pad_sequences([x_test[text_no]], seqX_len, padding='post')
bos_eos = tokenizer_ja.texts_to_sequences(["<s>", "</s>"])

print('元の文:', ''.join([detokenizer_en[i] for i in x_test[text_no]]))
print('生成文:', ''.join([detokenizer_ja[i] for i in decode_sequence(input_seq, bos_eos)]))
print('正解文:', ''.join([detokenizer_ja[i] for i in y_test[text_no]]))

#モデルの可視化
from IPython.display import SVG 
from tensorflow.python.keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

#機械翻訳の評価について
#BLEU
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

prediction = ['I', 'am', 'a', 'graduate', 'student', 'at', 'a', 'university']
reference = [['I' 'am', 'a', 'graduate', 'student', \
    'at', 'the', 'university', 'of', 'tokyo']]

text_no = 1
input_seq = pad_sequences([x_test[text_no]], seqX_len, padding='pos')
bos_eos = tokenizer_ja.texts_to_sequences(["<s>", "</s>"])

prediction = [detokenizer_ja[i] for i in decode_sequence(input_seq, bos_eos)]
reference = [[detokenizer_ja[i] for i in y_test[text_no]]]

print(prediction)
print(reference)
print(sentence_bleu(reference, prediction))
