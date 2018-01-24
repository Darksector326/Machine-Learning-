#!/usr/bin/env python

import tensorflow as tf
tf.__version__

node1 = tf.constant(3.0, dtype=tf.float32, name='n1')
node2 = tf.constant(4.0, name='n2')
node3 = tf.add(node1, node2)


print(node1, node2, node3)

sess = tf.Session()
print(sess.run([node1, node2]))
print("sess.run(node3): ",sess.run(node3))

#tf.summary.scalar('n3', node3)
#merged = tf.summary.merge_all()

file_writer = tf.summary.FileWriter('./l1', sess.graph)
file_writer.close()

