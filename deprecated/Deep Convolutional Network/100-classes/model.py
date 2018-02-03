import tensorflow as tf
import numpy as np

class Model:
    
    def __init__(self, dimension_picture, learning_rate, penalty, dropout, dimension_output, enable_dropout, enable_penalty):
        
        self.X = tf.placeholder(tf.float32, (None, dimension_picture, dimension_picture, 3))
        self.Y = tf.placeholder(tf.float32, (None, dimension_output))
        
        def convolutionize(x, w):
            return tf.nn.conv2d(input = x, filter = w, strides = [1, 1, 1, 1], padding = 'SAME')
        
        def pooling(wx):
            return tf.nn.max_pool(wx, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        
        first_W_conv = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev = 0.5))
        first_b_conv = tf.Variable(tf.random_normal([32], stddev = 0.1))
        first_hidden_conv = tf.nn.relu(convolutionize(self.X, first_W_conv) + first_b_conv)
        first_hidden_pool = pooling(first_hidden_conv)
        
        if enable_dropout:
            first_hidden_pool = tf.nn.dropout(first_hidden_pool, dropout / 4.0)
        
        second_W_conv = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev = 0.5))
        second_b_conv = tf.Variable(tf.random_normal([64], stddev = 0.1))
        second_hidden_conv = tf.nn.relu(convolutionize(first_hidden_conv, second_W_conv) + second_b_conv)
        second_hidden_pool = pooling(second_hidden_conv)
        
        if enable_dropout:
            second_hidden_pool = tf.nn.dropout(second_hidden_pool, dropout / 2.0)
        
        third_W_conv = tf.Variable(tf.random_normal([14 * 14 * 64, 1024], stddev = 0.5))
        third_b_conv = tf.Variable(tf.random_normal([1024], stddev = 0.1))
        third_hidden_flatted = tf.reshape(second_hidden_pool, [-1, 14 * 14 * 64])
        third_hidden_conv = tf.nn.relu(tf.matmul(third_hidden_flatted, third_W_conv) + third_b_conv)
        
        if enable_dropout:
            third_hidden_conv = tf.nn.dropout(third_hidden_conv, dropout)
        
        W = tf.Variable(tf.random_normal([1024, dimension_output], stddev = 0.5))
        b = tf.Variable(tf.random_normal([dimension_output], stddev = 0.1))
        
        self.y_hat = tf.matmul(third_hidden_conv, W) + b
        
        if enable_penalty:
            regularizers = tf.nn.l2_loss(first_W_conv) + tf.nn.l2_loss(second_W_conv) + tf.nn.l2_loss(third_W_conv)
        
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_hat, self.Y))
        
        if enable_penalty:
            self.cost = tf.reduce_mean(self.cost + penalty * regularizers)
        
        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.cost)
        
        correct_prediction = tf.equal(tf.argmax(self.y_hat, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
