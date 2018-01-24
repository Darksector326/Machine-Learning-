#!/usr/bin/env python

import tensorflow as tf

a = tf.placeholder(tf.float32, name='a')
b = tf.placeholder(tf.float32, name='b')
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
add_and_triple = adder_node * 3.

######
print(a)
print(b)
print(adder_node)

######

sess = tf.Session()

print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
print(sess.run(add_and_triple, {a: 3, b:4.5}))

######

#tf.summary.tensor_summary('a',a)
#tf.summary.tensor_summary('b',b)
#tf.summary.tensor_summary('c',adder_node)
#tf.summary.tensor_summary('d',add_and_triple)

#tf.summary.merge_all()

file_writer = tf.summary.FileWriter('./l2', sess.graph)
file_writer.close()



