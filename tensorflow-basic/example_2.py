# coding=utf-8
"""
Created on 下午2:20 17-7-17

@author: zyl
"""

import tensorflow as tf

g = tf.Graph()

with g.as_default():
    a = tf.constant([5,3], name='input_a')
    b = tf.reduce_prod(a, name='prob_b')
    c = tf.reduce_sum(a, name='sum_c')

    d = tf.add(b,c, name="add_d")

# todo
