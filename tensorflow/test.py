import tensorflow as tf

a = tf.placeholder(shape =[None,3],dtype=tf.float32)


feed_dict = {a:[[1,2,3],[3,4,5]]}
sess  = tf.Session()

print(sess.run([tf.shape(a)],feed_dict=feed_dict))