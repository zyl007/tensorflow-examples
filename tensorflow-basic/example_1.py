# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:22:27 2017

@author: knight
"""

import tensorflow as tf

print(tf.__version__)

a = tf.constant([5,5], name="input_a")  #
b = tf.constant([3,3], name="input_b")

c = tf.multiply(a,b, name="mul_c")
d = tf.add(a,b, name="add_d")

e = tf.add(c,d, name="add_e")

# sess = tf.Session()
# print(sess.run(e))

sess = tf.Session()
print(sess.run(e))
writer = tf.summary.FileWriter("./my_graph", sess.graph)

print(tf.get_default_graph())
writer.close()
sess.close()