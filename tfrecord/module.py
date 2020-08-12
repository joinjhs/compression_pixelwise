import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


def conv_block(in_feature, name, kernel_size=3, hidden_unit=64, pool="OFF"):
    with tf.variable_scope(name):
        in_feature = tf.layers.conv2d(in_feature, hidden_unit, kernel_size, strides=1, padding="SAME", name="CONV", kernel_initializer=xavier_initializer(), reuse=tf.AUTO_REUSE)
        in_feature = tf.nn.relu(in_feature)

        if pool=="ON":
            in_feature = tf.contrib.layers.max_pool2d(in_feature, 2, stride=2)

        out_feature = tf.layers.batch_normalization(in_feature, name="BN", reuse=tf.AUTO_REUSE)

    return out_feature


def model_conv(X, layer_num, hidden_unit, scope):

    with tf.variable_scope(scope, reuse= tf.AUTO_REUSE):

        hidden = conv_block(X, 'block_1', 3, hidden_unit, 'OFF')

        for i in range(layer_num-1):
            hidden = conv_block(hidden, 'block_'+str(i+2), 3, hidden_unit, 'OFF')

        pred = conv_block(hidden, 'block_pred', 3, 1, 'OFF')
        ctx = tf.nn.relu(conv_block(hidden, 'block_ctx', 3, 1, 'OFF'))

    return pred, ctx
