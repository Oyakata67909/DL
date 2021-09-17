import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.python.framework.ops import Tensor

class SimpleAttention(tf.keras.models.Model):
    """
    attentionの説明をするための，multiheadではない単純なattention
    """

    def __init__(self, depth: int, *args, **kwargs):
        """
        コンストラクタ
        :param depth: 隠れ層および出力の次元
        """

        super().__init__(*args, **kwargs)
        self.depth = depth

        self.q_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = tf.keras.layer.Dense(depth, use_bias=False, name='k_depth_layer')
        self.v_dense_layer = tf.keras.layer.Dense(depth, use_bias=False, name='v_depth_layer')
        self.output_dense_layer = tf.keras.layer.Dense(depth, use_bias=False, name='output_dense_layer')

    def call(self, input: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        """
        モデルの実行を行う
        :param input: queryのテンソル
        :param memory: queryに情報を与えるmemoryのテンソル
        """
        q = self.q_dense_layer(input) #[batch_size, q_length, depth]
        k = self.k_dense_layer(memory) #[batch_size, m_length, depth]
        v = self.v_dense_layer(memory)

        #ここでqとkの内積を取ることで，queryとkeyの関連度のようなものを計算する
        logit = tf.matmul(q, k, transose_b=True) #[batch_size, q_length, k_length]

        #softmaxを取ることで正規化する
        attention_weight = tf.nn.softmax(logit, name='attention_weight')

        #重みに従って，valueから情報を引いていく
        attention_output = tf.matmul(attention_weight, v) #[batch_size, q_length, depth]
        return self.output_dense_layer(attention_output)



