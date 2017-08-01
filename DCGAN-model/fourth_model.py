import tensorflow as tf

Z = tf.placeholder(tf.float32, shape = [1, 100])
G_W_1 = tf.Variable(tf.random_normal([100, 28 * 8 * 4 * 4], stddev = 0.5))
G_b_1 = tf.Variable(tf.random_normal([28 * 8 * 4 * 4], stddev = 0.01))

G_W_2 = tf.Variable(tf.random_normal([5, 5, 112, 224], stddev = 0.5))
G_b_2 = tf.Variable(tf.random_normal([112], stddev = 0.01))

G_W_3 = tf.Variable(tf.random_normal([5, 5, 56, 112], stddev = 0.5))
G_b_3 = tf.Variable(tf.random_normal([56], stddev = 0.01))

G_W_4 = tf.Variable(tf.random_normal([5, 5, 26, 56], stddev = 0.5))
G_b_4 = tf.Variable(tf.random_normal([26], stddev = 0.01))

G_W_5 = tf.Variable(tf.random_normal([5, 5, 3, 26], stddev = 0.5))
G_b_5 = tf.Variable(tf.random_normal([3], stddev = 0.01))
