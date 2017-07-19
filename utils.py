# coding=utf-8
"""
Created on 下午5:40 17-7-19

@author: zyl
"""
import errno
import os


# 确保目录存在
def ensure_directory(directory):
    """
    创建指定路径上尚不存在的目录
    :param directory:
    :return:
    """
    directory = os.path.expanduser(directory)
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:  # 判断是否为存在错误
            raise e

# 下载函数


# 磁盘缓存修饰器
# 属性字典
# 惰性属性修饰器