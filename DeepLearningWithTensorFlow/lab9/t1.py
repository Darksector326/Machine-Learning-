#!/bin/env python

import tensorflow as tf

a = tf.Variable(10,name="a");

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(a))

saver = tf.train.Saver()
save_path = saver.save(sess, "z1.ckpt")

print(tf.global_variables())
