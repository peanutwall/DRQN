import gym
import numpy as np
import os

env = gym.make('Frostbite-v0')
print(env.action_space.n)
print(env.observation_space.shape[0])

for i_episode in range(50):
    env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()#6
        print(action)
        observation, reward, done, info = env.step(action)
        # print(observation[34:194][0])#210*160
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break



#
#
#
# #one_hot is encoding format
# #None means tensor 的第一维度可以是任意维度
# #/255. 做均一化
# input_x=tf.placeholder(tf.float32,[None,28*28])/255
# #输出是一个one hot的向量
# output_y=tf.placeholder(tf.int32,[None,10])
#
# #输入层 [28*28*1]
# input_x_images=tf.reshape(input_x,[-1,28,28,1])
# #从(Test)数据集中选取3000个手写数字的图片和对应标签
#
# test_x=mnist.test.images[:3000] #image
# test_y=mnist.test.labels[:3000] #label
#
#
#
# #隐藏层
# #conv1 5*5*32
# #layers.conv2d parameters
# #inputs 输入，是一个张量
# #filters 卷积核个数，也就是卷积层的厚度
# #kernel_size 卷积核的尺寸
# #strides: 扫描步长
# #padding: 边边补0 valid不需要补0，same需要补0，为了保证输入输出的尺寸一致,补多少不需要知道
# #activation: 激活函数
# conv1=tf.layers.conv2d(
#     inputs=input_x_images,
#     filters=32,
#     kernel_size=[5,5],
#     strides=1,
#     padding='same',
#     activation=tf.nn.relu
# )
# print(conv1)
#
# #输出变成了 [28*28*32]
#
# #pooling layer1 2*2
# #tf.layers.max_pooling2d
# #inputs 输入，张量必须要有四个维度
# #pool_size: 过滤器的尺寸
#
# pool1=tf.layers.max_pooling2d(
#     inputs=conv1,
#     pool_size=[2,2],
#     strides=2
# )
# print(pool1)
# #输出变成了[?,14,14,32]
#
# #conv2 5*5*64
# conv2=tf.layers.conv2d(
#     inputs=pool1,
#     filters=64,
#     kernel_size=[5,5],
#     strides=1,
#     padding='same',
#     activation=tf.nn.relu
# )
#
# #输出变成了  [?,14,14,64]
#
# #pool2 2*2
# pool2=tf.layers.max_pooling2d(
#     inputs=conv2,
#     pool_size=[2,2],
#     strides=2
# )
#
# #输出变成了[?,7,7,64]
#
# #flat(平坦化)
# flat=tf.reshape(pool2,[-1,7*7*64])
#
#
# #形状变成了[?,3136]
#
# #densely-connected layers 全连接层 1024
# #tf.layers.dense
# #inputs: 张量
# #units： 神经元的个数
# #activation: 激活函数
# dense=tf.layers.dense(
#     inputs=flat,
#     units=1024,
#     activation=tf.nn.relu
# )
#
# #输出变成了[?,1024]
# print(dense)
#
# #dropout
# #tf.layers.dropout
# #inputs 张量
# #rate 丢弃率
# #training 是否是在训练的时候丢弃
# dropout=tf.layers.dropout(
#     inputs=dense,
#     rate=0.5,
# )
# print(dropout)
#
# #输出层，不用激活函数（本质就是一个全连接层）
# logits=tf.layers.dense(
#     inputs=dropout,
#     units=10
# )
# #输出形状[?,10]
# print(logits)
#
# #计算误差 cross entropy（交叉熵），再用Softmax计算百分比的概率
# #tf.losses.softmax_cross_entropy
# #onehot_labels: 标签值
# #logits: 神经网络的输出值
# loss=tf.losses.softmax_cross_entropy(onehot_labels=output_y,
#                                      logits=logits)
# # 用Adam 优化器来最小化误差,学习率0.001 类似梯度下降
# print(loss)
# train_op=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
#
#
# #精度。计算预测值和实际标签的匹配程度
# #tf.metrics.accuracy
# #labels：真实标签
# #predictions: 预测值
# #Return: (accuracy,update_op)accuracy 是一个张量准确率，update_op 是一个op可以求出精度。
# #这两个都是局部变量
# accuracy_op=tf.metrics.accuracy(
#     labels=tf.argmax(output_y,axis=1),
#     predictions=tf.argmax(logits,axis=1)
# )[1] #为什么是1 是因为，我们这里不是要准确率这个数字。而是要得到一个op
#
# #创建会话
# sess=tf.Session()
# #初始化变量
# #group 把很多个操作弄成一个组
# #初始化变量，全局，和局部
# init=tf.group(tf.global_variables_initializer(),
#               tf.local_variables_initializer())
# sess.run(init)
#
# for i in range(20000):
#     batch=mnist.train.next_batch(50) #从Train（训练）数据集中取‘下一个’样本
#     train_loss,train_op_=sess.run([loss,train_op],{input_x:batch[0],output_y:batch[1]})
#     if i%100==0:
#         test_accuracy=sess.run(accuracy_op,{input_x:test_x,output_y:test_y})
#         print("Step=%d, Train loss=%.4f,[Test accuracy=%.2f]"%(i,train_loss,test_accuracy))
#
# #测试： 打印20个预测值和真实值 对
# test_output=sess.run(logits,{input_x:test_x[:20]})
# inferenced_y=np.argmax(test_output,1)
# print(inferenced_y,'Inferenced numbers')#推测的数字
# print(np.argmax(test_y[:20],1),'Real numbers')
# sess.close()