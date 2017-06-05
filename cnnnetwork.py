#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf

import tensorlayer as tl
import matplotlib.pyplot as plt
# from tensorlayer.layers import set_keep
import numpy as np

from PIL import Image



y_train=300 #train number

y_test=50 #test number



def read_and_decode(filename,is_train=None):
    #Generate a queue based on the file name
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #Returns the file name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [225, 225, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    if is_train == True:

    # 1. Randomly cropped a part of [height, width] in the image..
        img = tf.random_crop(img, [125, 125, 3])
            # 2. random flip.
        img = tf.image.random_flip_left_right(img)
            # 3. random brightness.
            # img = tf.image.random_brightness(img, max_delta=63)
            # 4. random contrast.
            # img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
            # 5. Standardized (de-mean, variance).
        #img = tf.image.per_image_standardization(img)
    elif is_train == False:
            # 1. Crop the image in the center of the [height, width] section.
        img = tf.image.resize_image_with_crop_or_pad(img, 125, 125)
            # 2. Standardization (de - mean, except for variance).
        #img = tf.image.per_image_standardization(img)
    elif is_train == None:
        img = img

    label = tf.cast(features['label'], tf.int32)
    return img, label







batch_size = 28 # Set the batch size
with tf.device('/cpu:0'): # USE CPU0
    # Define the session, in order to avoid the case where the specified device does not exist,
    # allow_soft_placement is set to True to cause the program to automatically select an existing and supported device to run op
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # Prepare data

    x_train_, y_train_ = read_and_decode("trainsize225data.tfrecords",True)
    x_test_, y_test_ = read_and_decode("testsize225data.tfrecords", False)

    x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_, y_train_],
                                                          batch_size=batch_size,
                                                          capacity=2000,
                                                          min_after_dequeue=1000,
                                                          num_threads=32)  # Set the number of threads here
    # use batch instead of shuffle_batch
    x_test_batch, y_test_batch = tf.train.batch([x_test_, y_test_],
                                                batch_size=batch_size,
                                                capacity=50000,
                                                num_threads=32)



    def inference(x_crop, y_, reuse):
        with tf.variable_scope("model", reuse=reuse):  # Open a scope (scope), specify whether the reuse
            tl.layers.set_name_reuse(reuse)  # Allow the layer name to be multiplexed when you want two or more
                                            # Input placeholders (inferences) to share the same model parameters
            network = tl.layers.InputLayer(x_crop, name='input_layer')  # set input layer
            network = tl.layers.Conv2dLayer(network,  # Set convolution layer
                                            act=tf.nn.relu,  # Define the activation function, using the relu function
                                            shape=[5, 5, 3, 64],  # shape：[eight, width, number of input channels, number of output channels]
                                            strides=[1, 1, 1, 1],  # The sliding step of the convolution kernel strides = [1, 垂直步长, 水平步长, 1]
                                            padding='SAME',  # Set padding ‘SAME’
                                            W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                            # Define the weight matrix of the initializer for the 0 mean and the specified standard deviation of the truncated normal distribution
                                            b_init=tf.constant_initializer(value=0.0),  # The initializer defining the offset vector is a constant initializer with an initial value of zero
                                            name='cnn_layer1')  # : (batch_size, 24, 24, 64)
            network = tl.layers.PoolLayer(network,  # Define the pooling layer
                                          ksize=[1, 3, 3, 1],
                                          strides=[1, 2, 2, 1],
                                          padding='SAME',
                                          pool=tf.nn.max_pool,  # Define the pooling method for maximum pooling
                                          name='pool_layer1', )  # :(batch_size, 12, 12, 64)
            #LRN
            network.outputs = tf.nn.lrn(network.outputs, 4, bias=1.0, alpha=0.001 / 9.0,
                                        beta=0.75, name='norm1')  # Local Response Normalization，
                                                                    # More detail you can see Alex paper
            network = tl.layers.Conv2dLayer(network,
                                            act=tf.nn.relu,
                                            shape=[5, 5, 64, 64],
                                            strides=[1, 1, 1, 1],
                                            padding='SAME',
                                            W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                            b_init=tf.constant_initializer(value=0.1),
                                            name='cnn_layer2')  # ： (batch_size, 12, 12, 64)
            network.outputs = tf.nn.lrn(network.outputs, 4, bias=1.0, alpha=0.001 / 9.0,
                                        beta=0.75, name='norm2')
            network = tl.layers.PoolLayer(network,
                                          ksize=[1, 3, 3, 1],
                                          strides=[1, 2, 2, 1],
                                          padding='SAME',
                                          pool=tf.nn.max_pool,
                                          name='pool_layer2')  # : (batch_size, 6, 6, 64)

            # Define the Flatten layer , flatten the multidimensional input as a vector,
            # prepare for the DenseLayer
            network = tl.layers.FlattenLayer(network, name='flatten_layer')  #: (batch_size, 2304)

            network = tl.layers.DenseLayer(network, n_units=384, act=tf.nn.relu,  # Define the full connection layer
                                           W_init=tf.truncated_normal_initializer(stddev=0.04),
                                           b_init=tf.constant_initializer(value=0.1),
                                           name='relu1')  # : (batch_size, 384)
            network = tl.layers.DenseLayer(network, n_units=192, act=tf.nn.relu,
                                           W_init=tf.truncated_normal_initializer(stddev=0.04),
                                           b_init=tf.constant_initializer(value=0.1),
                                           name='relu2')  # : (batch_size, 192)
            network = tl.layers.DenseLayer(network, n_units=2, act=tf.identity,
                                           W_init=tf.truncated_normal_initializer(stddev=1 ),
                                           b_init=tf.constant_initializer(value=0.0),
                                           name='output_layer')  # : (batch_size, 10)
            y = network.outputs
            # Calculate the cross entropy loss ce, so the network does not have a softmax layer,
            # so use the softmax_cross_entropy_with_logits function
            ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
            # L2 is norm regularization,without it the accuracy rate will drop by 15%.
            L2 = tf.contrib.layers.l2_regularizer(0.004)(network.all_params[4]) + \
                 tf.contrib.layers.l2_regularizer(0.004)(network.all_params[6])
            cost = ce + L2  # Define the loss function of the network

            # correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), y_)
            correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)  # Determine the forecast accurate
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # Calculate the forecast accuracy

            return cost, acc, network

    #Batch normalization
    def inference_batch_norm(x_crop, y_, reuse, is_train):
        """
        For batch normalization, the normalization should be placed after cnn
        with linear activation.
        """
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            network = tl.layers.InputLayer(x_crop, name='input_layer')
            network = tl.layers.Conv2dLayer(network,
                                            act=tf.identity,  # Define the activation function as a linear function : y = x
                                            shape=[5, 5, 3, 64],
                                            strides=[1, 1, 1, 1],
                                            padding='SAME',
                                            W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                            # b_init=tf.constant_initializer(value=0.0),
                                            b_init=None,  # No bias item
                                            name='cnn_layer1')  # : (batch_size, 24, 24, 64)
            #
            # Define the Batch Normalization Layer，Batch Normalization: Accelerating Deep Network Training by  Reducing Internal Covariate Shift

            network = tl.layers.BatchNormLayer(network, is_train=is_train, name='batch_norm1')
            network.outputs = tf.nn.relu(network.outputs, name='relu1')  # Relu operation on the output
            network = tl.layers.PoolLayer(network,
                                          ksize=[1, 3, 3, 1],
                                          strides=[1, 2, 2, 1],
                                          padding='SAME',
                                          pool=tf.nn.max_pool,
                                          name='pool_layer1', )  # : (batch_size, 12, 12, 64)

            network = tl.layers.Conv2dLayer(network,
                                            act=tf.identity,
                                            shape=[5, 5, 64, 64],
                                            strides=[1, 1, 1, 1],
                                            padding='SAME',
                                            W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                            # b_init=tf.constant_initializer(value=0.1),
                                            b_init=None,
                                            name='cnn_layer2')  # : (batch_size, 12, 12, 64)

            network = tl.layers.BatchNormLayer(network, is_train=is_train, name='batch_norm2')
            network.outputs = tf.nn.relu(network.outputs, name='relu2')
            network = tl.layers.PoolLayer(network,
                                          ksize=[1, 3, 3, 1],
                                          strides=[1, 2, 2, 1],
                                          padding='SAME',
                                          pool=tf.nn.max_pool,
                                          name='pool_layer2')  # : (batch_size, 6, 6, 64)
            network = tl.layers.FlattenLayer(network, name='flatten_layer')
            network = tl.layers.DenseLayer(network, n_units=384, act=tf.nn.relu,
                                           W_init=tf.truncated_normal_initializer(stddev=0.04),
                                           b_init=tf.constant_initializer(value=0.1),
                                           name='relu1')  # : (batch_size, 384)
            network = tl.layers.DenseLayer(network, n_units=192, act=tf.nn.relu,
                                           W_init=tf.truncated_normal_initializer(stddev=0.04),
                                           b_init=tf.constant_initializer(value=0.1),
                                           name='relu2')  # : (batch_size, 192)
            network = tl.layers.DenseLayer(network, n_units=2, act=tf.identity,
                                           W_init=tf.truncated_normal_initializer(stddev=1 / 192.0),
                                           b_init=tf.constant_initializer(value=0.0),
                                           name='output_layer')  # : (batch_size, 10)
            y = network.outputs

            ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
            # L2 is norm regularization,without it the accuracy rate will drop by 15%.
            L2 = tf.contrib.layers.l2_regularizer(0.004)(network.all_params[4]) + \
                 tf.contrib.layers.l2_regularizer(0.004)(network.all_params[6])
            cost = ce + L2

            # correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), y_)
            correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return cost, acc, network

    # ## You can also use placeholder to feed_dict in data after using
    # ## val, l = sess.run([x_train_batch, y_train_batch]) to get the data
    # # x_crop = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])
    # # y_ = tf.placeholder(tf.int32, shape=[batch_size,])
    # # cost, acc, network = inference(x_crop, y_, None)
    #
    # with tf.device('/gpu:0'): # Use GPU
    # cost, acc, network = inference(x_train_batch, y_train_batch, None)
    # cost_test, acc_test, _ = inference(x_test_batch, y_test_batch, True)

    #     ## you can try batchnorm
    cost, acc, network = inference_batch_norm(x_train_batch, y_train_batch, None, is_train=True)
    cost_test, acc_test, _ = inference_batch_norm(x_test_batch, y_test_batch, True, is_train=False)
    #
    # ## train
    n_epoch =50 # set epoch number
    learning_rate = 0.0001 # set learning rata
    print_freq = 1 # print
    n_step_epoch = int(y_train / batch_size) #calculate each epoch
    n_step = n_epoch * n_step_epoch # calculate all epoch

    with tf.device('/gpu:0'):# Use GPU
        # Train on GPU
        # train_params = network.all_params # The parameters to be trained are for all network parameters
        # Define the training operation, using the Adaptive Moment Estimator (ADAM) algorithm to minimize the loss function
        train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                          epsilon=1e-08, use_locking=False).minimize(cost)  # , var_list=train_params)


    init = tf.initialize_all_variables()
    # init = tf.global_variables_initializer()

    sess.run(init)
    # sess.run(tf.global_variables_initializer()) # Initialize all variables
    # if resume:
    #     print("Load existing model " + "!" * 10)
    #     saver = tf.train.Saver()
    #     saver.restore(sess, model_file_name) # loading previous model

    network.print_params(False) # Do not print network parameter information
    network.print_layers() # Print the information of each layer

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)
    print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_epoch, n_step))

    coord = tf.train.Coordinator() # Create a thread coordinator
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) #　Create a thread
    # for step in range(n_step):
    step = 0

    temptrainloss = []
    temptrainacc = []
    temptestloss = []
    temptestacc = []
    for epoch in range(n_epoch):
        train_loss, train_acc, n_batch = 0, 0, 0 # Initialize variables to 0

        for s in range(10): # Set epoch number
            ## You can also use placeholder to feed_dict in data after using
        # val, l = sess.run([x_train_batch, y_train_batch])
        # tl.visualize.images2d(val, second=3, saveable=False, name='batch', dtype=np.uint8, fig_idx=2020121)
        # err, ac, _ = sess.run([cost, acc, train_op], feed_dict={x_crop: val, y_: l})
            err, ac, _ = sess.run([cost, acc, train_op]) # Calculate the loss function and the training accuracy rate
            step += 1
            train_loss += err
            train_acc += ac
            n_batch += 1

            # print step,":",err,ac
            # print train_loss
            # print  "train n_batch:", n_batch
        temptrainloss.append(train_loss/n_batch)
        temptrainacc.append(train_acc/n_batch)
        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:# Print the training information at the scheduled
            print("Epoch %d : Step %d-%d of %d took %fs" % (
            epoch, step, step + n_step_epoch, n_step))
            print("   train loss: %f" % (train_loss/n_batch ))

            print("   train acc: %f" % (train_acc/n_batch))

            # print("   train acc: %f" % (train_acc ))

            test_loss, test_acc, n_batch = 0, 0, 0 # print test information

            # for _ in range(int((y_test) /batch_size)):
            # for _ in range(int(len(y_test) / batch_size)):
            for _ in range(1):
                err, ac = sess.run([cost_test, acc_test])
                test_loss += err
                test_acc += ac
                n_batch += 1
            # temptestloss.append(err)
            # temptestloss.append(ac)
                # print step, "test:", err, ac
            # print step, "test n_batch:", n_batch

            print("   test loss: %f" % (test_loss/ n_batch))
            temptestloss.append(test_loss/n_batch)
            print("   test acc: %f" % (test_acc ))
            temptestacc.append(test_acc)

        # if (epoch + 1) % (print_freq * 20) == 0: # save the model
        #     print("Save model " + "!" * 10)
        #     saver = tf.train.Saver()
        #     save_path = saver.save(sess, model_file_name)

    # print '''train loss :''' ,temptrainloss
    #
    # print '''train acc :''' ,temptrainacc
    # print '''test loss :''' ,temptestloss
    # print '''test acc  :''' , temptestacc




        # #Plot the results

    trainloss=np.array(temptrainloss)
    trainacc = np.array(temptrainacc)
    testloss = np.array(temptestloss)
    testacc = np.array(temptestacc)

    print  testacc

    plt.figure()
        # plt.scatter(X, y, c="k", label="data")
        # print len(X),len(y),len(truedatay)
        # plt.plot(np.arange(0,len(Xtrue)), truedatay, '.', label='ture data')
    plt.plot(np.arange(0, 50), trainloss, 'bo--', label='train loss')
    # plt.plot(np.arange(0, 20), trainacc,  c="c", label='train acc')
    plt.plot(np.arange(0, 50), testloss, 'cx--', label='test loss')
    # plt.plot(np.arange(0, 20), testacc,   c="g", label='test acc')


    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train loss")
    plt.legend()
    plt.show()

