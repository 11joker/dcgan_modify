# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:42:55 2019

@author: 25493
"""

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./", one_hot=True)


num_steps = 20000
batch_size = 32

image_dim = 784 
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200

"""
# Generator Network
# Input: Noise, Output: Image
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        x = tf.layers.dense(x, units=6 * 6 * 128)
        x = tf.nn.tanh(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 6, 6, 128)
        x = tf.reshape(x, shape=[-1, 6, 6, 128])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
        # Deconvolution, image shape: (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
        # Apply sigmoid to clip values between 0 and 1
        x = tf.nn.sigmoid(x)
        return x
"""

def generator(x, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        x = tf.layers.dense(x, units=4 * 4 * 32)
        x = tf.nn.tanh(x)

        x = tf.reshape(x, shape=[-1, 4, 4, 32])

        x = tf.layers.conv2d_transpose(x, 16, 4, strides=1, padding="valid")

        x = tf.layers.conv2d_transpose(x, 16, 3, strides=2, padding="same")
        
        x = tf.layers.conv2d_transpose(x, 1, 3, strides=2, padding="same")

        x = tf.nn.sigmoid(x)
        return x


def discriminator(x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):

        x = tf.layers.conv2d(x, 64, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        x = tf.layers.dense(x, 2)
    return x

def accuracy(real_x, fake_x):
    fake_x = generator(fake_x,reuse = True)
    x = tf.concat([real_x, fake_x], axis=0)
    y = tf.concat([tf.ones([1,batch_size]), tf.zeros([1,batch_size])], axis=1)
    output = discriminator(x, reuse=True)
    print(tf.shape(y))
    print(tf.shape(tf.argmax(output, 1)))
    correct = tf.equal(tf.cast(tf.argmax(output, 1), tf.float32), y)
    correct_rate = tf.reduce_mean(tf.cast(correct, tf.float32))
    disc_final_loss = tf.abs(0.5-correct_rate)
    return disc_final_loss

noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

gen_sample = generator(noise_input)

disc_real = discriminator(image_input)
disc_fake = discriminator(gen_sample, reuse=True)

disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.ones_like(disc_real),
                                logits=disc_real))

d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(disc_fake),
                    logits=disc_fake))
disc_loss = d_real_loss + d_fake_loss
    
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.ones_like(disc_fake),
                                logits=disc_fake))

optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.001)

gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

accura = accuracy(image_input, noise_input)
g_loss_list = []
d_loss_list = []
accura_list = []
step = []

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(1, num_steps+1):

        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])

        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        
        feed_dict = {image_input: batch_x, noise_input: z}
        _, _, ac, gl, dl = sess.run([train_disc, train_gen, accura, gen_loss, disc_loss],
                                feed_dict=feed_dict)
        
        g_loss_list.append(gl)
        d_loss_list.append(dl)
        accura_list.append(ac)
        step.append(i)
        if i % 100 == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f, accuracy: %f' % (i, gl, dl, ac))
            
            z = np.random.uniform(-1., 1., size=[1, noise_dim])
            g = sess.run(gen_sample, feed_dict={noise_input: z})
            
            plt.figure(1)
            plt.plot(step, g_loss_list,marker=".")
            plt.plot(step, d_loss_list)
            plt.legend(labels=["Generator", "Discriminator"])
            
            plt.figure(2)
            plt.plot(step, accura_list, marker=".")
            
            plt.figure(3)
            img = np.reshape(g, (28, 28))
            plt.imshow(img)
            plt.show()
#-------------------------------------------------------------------------------------------------------
    # Generate images from noise, using the generator network.
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[4, noise_dim])
        g = sess.run(gen_sample, feed_dict={noise_input: z})
        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()
    plt.waitforbuttonpress()
