import tensorflow as tf
from random import randint
import numpy as np

data=np.ones([11,4,4])
data = np.load('data/train_data_postmel.npy')[:10]
train_labels=np.arange(10)
#train_labels = np.load('data/train_labels.npy')[:10]
x = tf.placeholder(tf.float32, [None, 80, 80])
labels = tf.placeholder(tf.int32, [None])

dataset = tf.data.Dataset.from_tensor_slices((x, labels))
#dataset = dataset.shuffle(buffer_size=11)
dataset = dataset.batch(2)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        sess.run(iterator.initializer,feed_dict={x:data,labels:train_labels})
        while True:
            try:
                data_X, data_y = sess.run(next_element)
                print(data_y)
            except tf.errors.OutOfRangeError:
                print('end')
                break
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        sess.run(iterator.initializer,feed_dict={x:data,labels:train_labels})
        for j in range(4):
            try:
                data_X, data_y = sess.run(next_element)
                print(data_y)
            except tf.errors.OutOfRangeError:
                print('end')
                break
        print('here')
