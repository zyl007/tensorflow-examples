# coding=utf-8
"""
Created on 下午6:16 17-7-19

@author: zyl
"""
import glob
from itertools import groupby
from collections import defaultdict


image_filenames = glob.glob('/home/knight/workspace/tensorflow-examples/Images/n02*/*.jpg')
print(image_filenames[0:2])

training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

# 将文件分解为品种和相应的文件名，品种对应于文件夹名称 /n02085936-Maltese_dog/n02085936_2573.jpg
image_filename_with_breed = map(lambda filename: (filename.split('/')[-2:]), image_filenames)

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

training_dataset['n02085620-Chihuahua']