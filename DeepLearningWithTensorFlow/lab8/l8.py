#!/bin/env python

import tensorflow as tf

config = tf.ConfigProto()
config.log_device_placement = True

with tf.device("/gpu:1"):
    a = tf.Variable(3.0, name = 'a')
    b = tf.constant(4.0, name = 'b')
    c = a + b

with tf.device("/cpu:0"):
    d = tf.Variable(5.0, name='d')
    e = tf.constant(7.0, name='e')
    f = d*e
    
with tf.device("/gpu:3"):
    h = tf.Variable(15.0, name='h')
    g = tf.constant(97.0, name='g')
    k = h/g
    
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run([a,b,c]))
print(sess.run([f]))
print(sess.run([k]))
