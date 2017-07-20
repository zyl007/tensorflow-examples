# coding=utf-8
"""
Created on 下午3:36 17-7-20

@author: zyl
"""
import tensorflow as tf


# float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)

def cnn_model(image_batch, batch_size=None):
    float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)
    conv2d_layer_one = tf.contrib.layers.convolution2d(
        inputs=float_image_batch,
        num_outputs=32,
        kernel_size=(5,5),
        activation_fn=tf.nn.relu,
        # weights_initializer=tf.random_normal,
        stride=(2,2),
        trainable=True
    )
    # # 输出网络层结构
    # print (conv2d_layer_one.get_shape(),
    #        pool_layer_one.get_shape())

    pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
    conv2d_layer_two = tf.contrib.layers.convolution2d(
        pool_layer_one,
        num_outputs=64,
        kernel_size=(5,5),
        activation_fn=tf.nn.relu,
        # weights_initializer=tf.random_normal,
        stride=(1, 1),
        trainable=True

    )

    pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding="SAME")
    flattened_layer_two = tf.reshape(pool_layer_two,
                                     [batch_size,-1])

    print(flattened_layer_two.get_shape())

    # 全连接层
    hidden_layer_three = tf.contrib.layers.fully_connected(
        flattened_layer_two,
        num_outputs=512,
        # weights_initializer = lambda i, dtype: tf.truncated_normal([38912, 512], stddev=0.1),
        activation_fn=tf.nn.relu,
    )
    # dropout层防止过拟合
    hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)

    # 全连接
    final_fully_connected = tf.contrib.layers.fully_connected(
        hidden_layer_three,
        120,  # 输出是dog的品种数
        # weights_initializer = lambda i, dtype: tf.truncated_normal([512,120], stddev=0.1)
    )
    # 返回最后输出
    return final_fully_connected

def main():
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once("./output/train-images/*.tfrecords")
    )
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
        }
    )

    record_image = tf.decode_raw(features['image'], tf.uint8)

    # 修改图像的形状有助于训练和输出的可视化

    image = tf.reshape(record_image, [250, 151, 1])
    label = tf.cast(features['label'], tf.string)
    min_after_dequeue = 10
    batch_size = 3
    capacity = min_after_dequeue + 3*batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue
    )

    import glob
    labels = list(map(lambda c:c.split('/')[-1], glob.glob("/home/knight/workspace/tensorflow-examples/Images/*")))
    train_labels = tf.map_fn(lambda l: tf.where(tf.equal(labels, l))[0,0:1][0],
                             label_batch, dtype=tf.int64)
    final_full_connected = cnn_model(image_batch, batch_size)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_full_connected,
                                                                         labels=train_labels))
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01,
        batch*3,
        120,
        0.95,
        staircase=True
    )
    optimizer = tf.train.AdamOptimizer(learning_rate,0.9).minimize(loss, global_step=batch)

    train_prediction = tf.nn.softmax(final_full_connected,)


if __name__ == '__main__':
    main()