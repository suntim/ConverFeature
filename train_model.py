# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import math
import tensorflow as tf
from tensorflow.python.framework import graph_util
import input_data
import matplotlib.pyplot as plt
tra_accuracy = []
train_loss = []
test_acc = []
LR = []

tra_data_dir = 'D://AutoSparePart2//IntTF//train.tfrecords'
val_data_dir = 'D://AutoSparePart2//IntTF//val.tfrecords'

max_learning_rate = 0.00001
min_learning_rate = 0.000000002
decay_speed = 2000.0 
lr = tf.placeholder(tf.float32)
batch_size = 20
num_epochs = 5001
W = 214
H = 186
Channels = 3
n_classes = 2

def my_batch_norm(inputs):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), dtype=tf.float32)
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), dtype=tf.float32)
    batch_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    batch_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    
    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
    return inputs, batch_mean, batch_var, beta, scale

def build_network(height, width, channel):
    x = tf.placeholder(tf.float32, shape=[None, height, width, channel], name="input")
    y = tf.placeholder(tf.int32, shape=[None, n_classes], name="labels_placeholder")

    def weight_variable(shape, name="weights"):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name="biases"):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(input, w):
        return tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')

    def pool_max(input,name='pool1'):
        return tf.nn.max_pool(input,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name=name)

    def fc(input, w, b):
        return tf.matmul(input, w) + b

    # conv1
    with tf.name_scope('conv1_1') as scope:
        kernel = weight_variable([3, 3, Channels, 64])
        biases = bias_variable([64])
        conv1_1 = tf.nn.bias_add(conv2d(x, kernel), biases)
        inputs, pop_mean, pop_var, beta, scale = my_batch_norm(conv1_1)
        conv_batch_norm = tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)
        output_conv1_1 = tf.nn.relu(conv_batch_norm, name=scope)
        # 结果可视化
        split = tf.split(output_conv1_1, num_or_size_splits=64, axis=3)
        tf.summary.image('conv1_1_features', split[0], 64)
        
    with tf.name_scope('pool1_1') as scope:
        pool1_1 = pool_max(output_conv1_1,name='pool1-1')

    with tf.name_scope('conv1_2') as scope:
        kernel = weight_variable([3, 3, 64, 64])
        biases = bias_variable([64])
        conv1_2 = tf.nn.bias_add(conv2d(pool1_1, kernel), biases)
        inputs, pop_mean, pop_var, beta, scale = my_batch_norm(conv1_2)
        conv_batch_norm = tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)
        output_conv1_2 = tf.nn.relu(conv_batch_norm, name=scope)
        # 结果可视化
        split = tf.split(output_conv1_2, num_or_size_splits=64, axis=3)
        tf.summary.image('conv1_2_features', split[0], 64)

    with tf.name_scope('pool1_2') as scope:
        pool1_2 = pool_max(output_conv1_2,name='pool1-2')

    # conv2
    with tf.name_scope('conv2_1') as scope:
        kernel = weight_variable([3, 3, 64, 128])
        biases = bias_variable([128])
        conv2_1 = tf.nn.bias_add(conv2d(pool1_2, kernel), biases)
        inputs, pop_mean, pop_var, beta, scale = my_batch_norm(conv2_1)
        conv_batch_norm = tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)
        output_conv2_1 = tf.nn.relu(conv_batch_norm, name=scope)
        # 结果可视化
        split = tf.split(output_conv2_1, num_or_size_splits=128, axis=3)
        tf.summary.image('conv2_1_features', split[0], 128)
        
    with tf.name_scope('pool2_1') as scope:
        pool2_1 = pool_max(output_conv2_1,name='pool2-1')
        

    with tf.name_scope('conv2_2') as scope:
        kernel = weight_variable([3, 3, 128, 128])
        biases = bias_variable([128])
        conv2_2 = tf.nn.bias_add(conv2d(pool2_1, kernel), biases)
        inputs, pop_mean, pop_var, beta, scale = my_batch_norm(conv2_2)
        conv_batch_norm = tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)
        output_conv2_2 = tf.nn.relu(conv_batch_norm, name=scope)
        # 结果可视化
        split = tf.split(output_conv2_2, num_or_size_splits=128, axis=3)
        tf.summary.image('conv2_2_features', split[0], 128)
        
    with tf.name_scope('pool2_2') as scope:
        pool2_2 = pool_max(output_conv2_2,name='pool2-2')

    # fc6
    with tf.name_scope('fc6') as scope:
        shape = int(np.prod(pool2_2.get_shape()[1:]))
        kernel = weight_variable([shape, 120])
        # kernel = weight_variable([shape, 4096])
        # biases = bias_variable([4096])
        biases = bias_variable([120])
        pool5_flat = tf.reshape(pool2_2, [-1, shape])
        output_fc6 = tf.nn.relu(fc(pool5_flat, kernel, biases), name=scope)

    # fc7
    with tf.name_scope('fc7') as scope:
        # kernel = weight_variable([4096, 4096])
        # biases = bias_variable([4096])
        kernel = weight_variable([120, 100])
        biases = bias_variable([100])
        output_fc7 = tf.nn.relu(fc(output_fc6, kernel, biases), name=scope)

    # fc8
    with tf.name_scope('fc8') as scope:
        # kernel = weight_variable([4096, n_classes])
        kernel = weight_variable([100, n_classes])
        biases = bias_variable([n_classes])
        output_fc8 = tf.nn.relu(fc(output_fc7, kernel, biases), name=scope)

    finaloutput = tf.nn.softmax(output_fc8, name="softmax")

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=finaloutput, labels=y)) * 1000
    optimize = tf.train.AdamOptimizer(lr).minimize(cost)

    prediction_labels = tf.argmax(finaloutput, axis=1, name="output")
    read_labels = tf.argmax(y, axis=1)

    correct_prediction = tf.equal(prediction_labels, read_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

    return dict(
        x=x,
        y=y,
        lr=lr,
        optimize=optimize,
        correct_prediction=correct_prediction,
        correct_times_in_batch=correct_times_in_batch,
        cost=cost,
        accuracy=accuracy,
    )


def train_network(graph, batch_size, num_epochs, pb_file_path):
    
    tra_image_batch, tra_label_batch = input_data.read_and_decode2stand(tfrecords_file=tra_data_dir,
                                                 batch_size=batch_size)
    val_image_batch, val_label_batch = input_data.read_and_decode2stand(tfrecords_file=val_data_dir,
                                                    batch_size=batch_size)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # log
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./logs/train", sess.graph)
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        epoch_delta = 10
        try:
            for epoch_index in range(num_epochs):
                learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-epoch_index / decay_speed)
                LR.append(learning_rate)
                tra_images, tra_labels = sess.run([tra_image_batch, tra_label_batch])
                summary, accuracy, mean_cost_in_batch, return_correct_times_in_batch, _ = sess.run([merged, graph['accuracy'],
                                                                                    graph['cost'], graph['correct_times_in_batch'], 
                                                                                    graph['optimize']],
                                                                                    feed_dict={
                                                                                                graph['x']: tra_images,
                                                                                                graph['lr']:learning_rate,
                                                                                                graph['y']: tra_labels})
                
                if epoch_index % epoch_delta == 0:
                    # 开始在 train set上计算一下accuracy和cost
                    print("index[%s]".center(50, '-') % epoch_index)
                    print("Train: cost_in_batch：{},correct_in_batch：{},accuracy：{}".format(mean_cost_in_batch, return_correct_times_in_batch, accuracy))
                    tra_accuracy.append(accuracy)
                    train_loss.append(mean_cost_in_batch)    
    
                    # 开始在 test set上计算一下accuracy和cost
                    val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                    mean_cost_in_batch, return_correct_times_in_batch = sess.run([graph['cost'], graph['correct_times_in_batch']], feed_dict={
                        graph['x']: val_images,
                        graph['y']: val_labels
                    })
                    test_acc.append(return_correct_times_in_batch / batch_size)
                    print("***Val: cost_in_batch：{},correct_in_batch：{},accuracy：{}".format(mean_cost_in_batch, return_correct_times_in_batch, return_correct_times_in_batch / batch_size))
                     
                    
                if epoch_index % 500 == 0: 
                    #log  write
                    train_writer.add_summary(summary, epoch_index)
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
                    with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
                        f.write(constant_graph.SerializeToString())
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

def main():
    pb_file_path = "./output/autosparepart2.pb"
    g = build_network(height=H, width=W, channel=3)
    train_network(g, batch_size, num_epochs, pb_file_path)
    # 正确率绘图
    fig1 = plt.figure('fig1')
    plt.plot(np.linspace(0, num_epochs - 1, len(tra_accuracy)), tra_accuracy, 'b', label='train_acc')
    plt.plot(np.linspace(0, num_epochs - 1, len(test_acc)), test_acc, 'r-.', lw=2, label='test_acc')
    plt.title('Train \ Test Accuracy')
    plt.xlabel('training_iters')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    # 交叉熵绘图
    fig2 = plt.figure('fig2')
    plt.plot(np.linspace(0, num_epochs - 1, len(train_loss)), train_loss, 'm', label='train_loss')
    plt.title('Train_Loss')
    plt.xlabel('training_iters')
    plt.ylabel('cross_entropy loss')
    plt.legend(loc='upper right')
    # 学习曲线
    fig3 = plt.figure('fig3')
    plt.plot(np.linspace(0, num_epochs - 1, len(LR)), LR, 'k-', label='learning_rate')
    plt.title('Learning_Rate')
    plt.xlabel('training_iters')
    plt.ylabel('lr')
    plt.legend(loc='upper right')
    
    plt.show(fig1)
    plt.show(fig2)
    plt.show(fig3)

main()
#tensorboard --logdir=D://logs
#http://192.168.1.102:6006
#http://115.200.76.146:6006
#http://115.200.72.139:6006
#http://172.28.14.131:6006
