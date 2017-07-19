# coding=utf-8
"""
Created on 上午11:54 17-7-18

@author: zyl
"""

import tensorflow as tf

# 初始化变量和模型参数

def inference(X):
    # 计算推断模型再数据上的输出
    pass

def loss(X, Y):
    # 依据数据计算损失
    pass

def inputs():
    # 读取输入数据
    pass

def train(total_loss):
    # 模型训练
    pass

def evaluate(sess, X, Y):
    # 模型评估
    pass

with tf.Session() as sess:
    # 在一个会话对象中启动数据流图，搭建流程
    pass


tf.initialize_all_variables().run()

X, Y = inputs()  #读取训练数据

total_loss = loss(X, Y)
train_op = train(total_loss)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

train_steps = 1000
for step in range(train_steps):
    sess.run([train_op])
    #
    if step%10 == 0:
        print("loss: ", sess.run([total_loss]))

evaluate(sess, X, Y)
coord.request_stop()
coord.join(threads)
sess.close()