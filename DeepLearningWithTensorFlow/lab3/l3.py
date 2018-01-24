#!/usr/bin/env python

import tensorflow as tf

######

W = tf.Variable([.3], dtype=tf.float32, name='W')
b = tf.Variable([-.3], dtype=tf.float32, name='b')
x = tf.placeholder(tf.float32, name='x')
linear_model = W * x + b

######

print(W)
print(b)
print(linear_model)

######

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(tf.float32, name='y')
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


######

#tf.summary.tensor_summary('a',a)
#tf.summary.tensor_summary('b',b)
#tf.summary.tensor_summary('c',adder_node)
#tf.summary.tensor_summary('d',add_and_triple)

#tf.summary.merge_all()

file_writer = tf.summary.FileWriter('./l3', sess.graph)
file_writer.close()



