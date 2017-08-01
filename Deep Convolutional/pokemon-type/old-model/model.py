import tensorflow as tf
import numpy as np

class Model:
    
    def __init__(self, dimension_picture, learning_rate, dimension_output):
        
        self.X = tf.placeholder(tf.float32, (None, dimension_picture, dimension_picture, 4))
        self.Y_1 = tf.placeholder(tf.float32, (None, dimension_output))
        self.Y_2 = tf.placeholder(tf.float32, (None, dimension_output))
        
        def convolutionize(x, w):
            return tf.nn.conv2d(input = x, filter = w, strides = [1, 1, 1, 1], padding = 'SAME')
        
        def pooling(wx):
            return tf.nn.max_pool(wx, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        
        first_W_conv = tf.Variable(tf.random_normal([5, 5, 4, 32], stddev = 0.5))
        first_b_conv = tf.Variable(tf.random_normal([32], stddev = 0.1))
        first_hidden_conv = tf.nn.relu(convolutionize(self.X, first_W_conv) + first_b_conv)
        first_hidden_pool = pooling(first_hidden_conv)
        
        second_W_conv = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev = 0.5))
        second_b_conv = tf.Variable(tf.random_normal([64], stddev = 0.1))
        second_hidden_conv = tf.nn.relu(convolutionize(first_hidden_conv, second_W_conv) + second_b_conv)
        second_hidden_pool = pooling(second_hidden_conv)

        third_W_conv = tf.Variable(tf.random_normal([14 * 14 * 64, 1024], stddev = 0.5))
        third_b_conv = tf.Variable(tf.random_normal([1024], stddev = 0.1))
        third_hidden_flatted = tf.reshape(second_hidden_pool, [-1, 14 * 14 * 64])
        third_hidden_conv = tf.nn.relu(tf.matmul(third_hidden_flatted, third_W_conv) + third_b_conv)
        
        W_1 = tf.Variable(tf.random_normal([1024, dimension_output], stddev = 0.5))
        b_1 = tf.Variable(tf.random_normal([dimension_output], stddev = 0.1))
        
        W_2 = tf.Variable(tf.random_normal([1024, dimension_output], stddev = 0.5))
        b_2 = tf.Variable(tf.random_normal([dimension_output], stddev = 0.1))
        
        self.y_hat_1 = tf.matmul(third_hidden_conv, W_1) + b_1
        self.y_hat_2 = tf.matmul(third_hidden_conv, W_2) + b_2
        
        self.cost_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_hat_1, self.Y_1))
        self.cost_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_hat_2, self.Y_2))
        
        self.cost = self.cost_1 + self.cost_2
        
        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.cost)
        
        correct_prediction_1 = tf.equal(tf.argmax(self.y_hat_1, 1), tf.argmax(self.Y_1, 1))
        self.accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction_1, "float"))
        
        correct_prediction_2 = tf.equal(tf.argmax(self.y_hat_2, 1), tf.argmax(self.Y_2, 1))
        self.accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))