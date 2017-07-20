# coding=utf-8
"""
Created on 下午6:16 17-7-19

@author: zyl
"""
import glob
from itertools import groupby
from collections import defaultdict
import tensorflow as tf

# 图像数据存放路径
image_filenames = glob.glob('/home/knight/workspace/tensorflow-examples/Images/n02*/*.jpg')
print(image_filenames[0:2])
sess = tf.InteractiveSession()


def split_train_test(image_filenames):
    training_dataset = defaultdict(list)
    testing_dataset = defaultdict(list)

    # 将文件分解为品种和相应的文件名，品种对应于文件夹名称 /n02085936-Maltese_dog/n02085936_2573.jpg
    image_filename_with_breed = map(lambda filename: (filename.split('/')[-2], filename), image_filenames)

    # 依据品种对图像分组
    for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x:x[0]):
        # 枚举每个品种的图像，20%划分入测试集
        for i, breed_images in enumerate(breed_images):
            if i%5==0:
                testing_dataset[dog_breed].append(breed_images[1])
            else:
                training_dataset[dog_breed].append(breed_images[1])

        # 检查每个品种的测试图像是否至少有全部图像的18%
        breed_training_count = len(training_dataset[dog_breed])
        breed_testing_count = len(testing_dataset[dog_breed])

        assert round(breed_testing_count/(breed_training_count+breed_testing_count), 2) > 0.18, "Not enough testing images"

    return training_dataset, testing_dataset

def write_records_file(dataset, record_location):
    """
    用dataset图像数据填充一个TFRecord,并指定类别
    :param dataset: dict(list)
    这个字典的键对应品种标签，值对应该品类下的所有样本
    :param record_location: str
    TFRecord输出的路径
    """
    writer = None

    # 遍历dataset，每隔100张图像，写入到一个新的TFRcord文件还总，以加快写操作的进程
    current_index = 0
    for breed, images_filenames in dataset.items():
        for image_filename in image_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()
                record_filename = "{record_location}-{current_index}.tfrecords".format(record_location=record_location,
                                                                                       current_index=current_index)
                writer = tf.python_io.TFRecordWriter(record_filename)  # 打开文件准备写入

            current_index += 1
            image_file = tf.read_file(image_filename)
            # try来处理部分无法识别为JPEG的图像
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_filename)
                continue
            # 转化为灰度值
            grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(grayscale_image, (250,151))
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            # 标签编码
            image_label = breed.encode('utf-8')

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))

            writer.write(example.SerializeToString())
    writer.close()

def load_TFrecord():
    pass

def main():
    training_dataset, testing_dataset = split_train_test(image_filenames)
    write_records_file(training_dataset, '../output/train-images/train-image')
    write_records_file(testing_dataset, '../output/test-images/test-image')


if __name__ == '__main__':
    main()
