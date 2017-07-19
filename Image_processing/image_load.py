# coding=utf-8
"""
Created on 下午2:11 17-7-19
使用tensorflow加载image数据
@author: zyl
"""
from __future__ import print_function
import tensorflow as tf
from PIL import Image

image_label = b'\x01'
image_filename = "/home/knight/workspace/tensorflow-examples/Images/n02085782-Japanese_spaniel/n02085782_2.jpg"
# filename_queue = tf.train.string_input_producer(
#     tf.train.match_filenames_once(image_filename)
# )
#
# image_reader = tf.WholeFileReader()
#
# _, image_file = image_reader.read(filename_queue)
#
# image = tf.image.decode_jpeg(image_file)

# 使用PIL读取图像
img = Image.open(image_filename)

print(img.format, img.size, img.mode)

img = img.resize((224, 224))  # 调整图片大小

image_bytes = img.tobytes()

writer = tf.python_io.TFRecordWriter('train.tfrecords')

example = tf.train.Example(features=tf.train.Features(feature={
    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
    }))

# 将样本保存到一个文本文件tfrecord
writer.write(example.SerializeToString())
writer.close()


