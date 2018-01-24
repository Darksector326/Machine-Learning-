#!/usr/bin/env python

import numpy as np
import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.INFO)

# Model parameters
W = tf.Variable([.3], dtype=tf.float32, name='W')
b = tf.Variable([-.3], dtype=tf.float32, name='b')
# Model input and output
x = tf.placeholder(tf.float32, name='x')
linear_model = W * x + b
y = tf.placeholder(tf.float32, name='y')
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

file_writer = tf.summary.FileWriter('./l4')
file_writer.add_graph(sess.graph)

tf.summary.scalar('loss', loss)
tf.summary.scalar('W',W[0])
tf.summary.scalar('b',b[0])
merged = tf.summary.merge_all()

for i in range(1000):
      WW,bb,l,m,_ = sess.run([W,b,loss,merged,train], {x:x_train, y:y_train})      
      if(i%100==0):
            file_writer.add_summary(m,i)
            print("%s\t%s\t%s\t%s"%(i,WW,bb,l))
# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

#file_writer = tf.summary.FileWriter('./l4')
file_writer.close()
