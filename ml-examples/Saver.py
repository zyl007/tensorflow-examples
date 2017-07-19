# coding=utf-8
"""
Created on 下午2:23 17-7-18
保存训练检查点
@author: zyl
"""

import tensorflow as tf

# 模型定义代码...
# 创建一个Saver对象
saver = tf.train.Saver()

# 在一个会话对象中启动数据流图，搭建流程

with tf.Session() as sess:

    # 模型设置...

    # 模型训练和保存

    for step in range(traing_steps):
        sess.run([train_op])

        # 保存检查点
        if step%1000==0:
            saver.save(sess, 'my-model', global_step=step)

    # 模型评估...
    saver.save(sess, "my-model", global_step=training_steps)
    sess.close()


#########################
