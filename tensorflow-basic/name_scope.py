# coding=utf-8
"""
Created on 下午6:16 17-7-17

@author: zyl
"""
import tensorflow as tf

with tf.name_scope("Scope_A"):
    a = tf.add(1,3, name='A_add')
    b = tf.multiply(a, 4, name="A_mul")

with tf.name_scope("Scope_B"):
    c = tf.add(4, 5, name="B_add")
    d = tf.multiply(c, 4, name="B_mul")

e = tf.add(b,d, name="output")

writer = tf.summary.FileWriter("./name_scope_1", graph=tf.get_default_graph())
writer.close()