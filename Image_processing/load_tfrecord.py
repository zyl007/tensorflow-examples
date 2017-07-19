# coding=utf-8
"""
Created on 下午4:58 17-7-19

@author: zyl
"""
import tensorflow as tf
# 1. 直接读取方式

for serialized_example in tf.python_io.tf_record_iterator("tfrecords_path"):
    example = tf.train.Example()  # 创建解析对象
    example.ParseFromString(serialized_example)  # 解析tfrecords

    image = example.features.feature['image'].bytes_list.value
    label = example.features.feature['label'].int64_list.value

    print(image, label)

# 2. 使用队列高效读取

def read_and_decode(fileanme):
    """

    :param fileanme:
    :return: 返回img和标签
    """
    # 根据文件名生成一个队列
    fileanme_queue = tf.train.string_input_producer([fileanme])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(fileanme_queue)
    # 解析文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32)*(1./255)-0.5
    label = tf.cast(features['label'], tf.int32)

    return (img, label)

# 使用shuffle_batch随机打乱顺序
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000)