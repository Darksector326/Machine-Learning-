#!/bin/env python

import tensorflow as tf

sess = tf.Session()

saver = tf.train.import_meta_graph("z1.ckpt.meta")
saver.restore(sess, "z1.ckpt")

print(sess.run('a:0'))

graph = tf.get_default_graph()
a = graph.get_tensor_by_name("a:0")

print(sess.run(a))

print(tf.global_variables())
