import tensorflow as tf
import numpy as np

# I strictly followed the DCGAN paper
# D(x)
# Input 64x64x3 -> conv 5x5 -> 32x32x64
# 32x32x64 -> conv 5x5 -> 16x16x128
# 16x16x128 -> conv 5x5 -> 8x8x256
# 8x8x256 -> conv 5x5 -> 4x4x512
# 4x4x512 -> fully connected -> 1

# G(x)
# Input 1x100 X 100x(1024 * 4 * 4) -> reshape (4, 4, 1024)
# 4x4x1024 -> deconv 5x5 -> 8x8x512
# 8x8x512 -> deconv 5x5 -> 16x16x256
# 16x16x256 -> deconv 5x5 -> 32x32x128
# 32x32x128 -> deconv 5x5 -> 64x64x3

class Model:
    
    def __init__(self, learning_rate, batch_size, dimension_picture, color_layer, output_dimension = 64):
        
        self.X = tf.placeholder(tf.float32, shape = [batch_size, dimension_picture, dimension_picture, color_layer])
        
        self.D_W_1 = tf.Variable(tf.random_normal([5, 5, color_layer, output_dimension], stddev = 0.5))
        self.D_b_1 = tf.Variable(tf.random_normal([output_dimension], stddev = 0.01))
        
        self.D_W_2 = tf.Variable(tf.random_normal([5, 5, output_dimension, output_dimension * 2], stddev = 0.5))
        self.D_b_2 = tf.Variable(tf.random_normal([output_dimension * 2], stddev = 0.01))
        
        self.D_W_3 = tf.Variable(tf.random_normal([5, 5, output_dimension * 2, output_dimension * 4], stddev = 0.5))
        self.D_b_3 = tf.Variable(tf.random_normal([output_dimension * 4], stddev = 0.01))
        
        self.D_W_4 = tf.Variable(tf.random_normal([5, 5, output_dimension * 4, output_dimension * 8], stddev = 0.5))
        self.D_b_4 = tf.Variable(tf.random_normal([output_dimension * 8], stddev = 0.01))
        
        self.D_W_5 = tf.Variable(tf.random_normal([8192, 1], stddev = 0.5))
        self.D_b_5 = tf.Variable(tf.random_normal([1], stddev = 0.01))
        
        backpropagate_D = [self.D_W_1, self.D_b_1, self.D_W_2, self.D_b_2, self.D_W_3, self.D_b_3, self.D_W_4, self.D_b_4, self.D_W_5, self.D_b_5]
        
        self.Z = tf.placeholder(tf.float32, shape = [batch_size, 100])
        
        self.G_W_1 = tf.Variable(tf.random_normal([100, output_dimension * 16 * 4 * 4], stddev = 0.5))
        self.G_b_1 = tf.Variable(tf.random_normal([output_dimension * 16 * 4 * 4], stddev = 0.01))
        
        self.G_W_2 = tf.Variable(tf.random_normal([5, 5, output_dimension * 8, output_dimension * 16], stddev = 0.5))
        self.G_b_2 = tf.Variable(tf.random_normal([output_dimension * 8], stddev = 0.01))
        
        self.G_W_3 = tf.Variable(tf.random_normal([5, 5, output_dimension * 4, output_dimension * 8], stddev = 0.5))
        self.G_b_3 = tf.Variable(tf.random_normal([output_dimension * 4], stddev = 0.01))
        
        self.G_W_4 = tf.Variable(tf.random_normal([5, 5, output_dimension * 2, output_dimension * 4], stddev = 0.5))
        self.G_b_4 = tf.Variable(tf.random_normal([output_dimension * 2], stddev = 0.01))
        
        self.G_W_5 = tf.Variable(tf.random_normal([5, 5, color_layer, output_dimension * 2], stddev = 0.5))
        self.G_b_5 = tf.Variable(tf.random_normal([color_layer], stddev = 0.01))
        
        backpropagate_G = [self.G_W_1, self.G_b_1, self.G_W_2, self.G_b_2, self.G_W_3, self.G_b_3, self.G_W_4, self.G_b_4, self.G_W_5, self.G_b_5]
    
        def lrelu(x, leak = 0.1):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * np.absolute(x)
        
        def convolutionize(x, conv_w, h = 2, w = 2):
            return tf.nn.conv2d(input = x, filter = conv_w, strides = [1, h, w, 1], padding = 'SAME')
        
        def discriminator(z):
            h0 = lrelu(convolutionize(z, self.D_W_1) + self.D_b_1)
            h1 = lrelu(convolutionize(h0, self.D_W_2) + self.D_b_2)
            h2 = lrelu(convolutionize(h1, self.D_W_3) + self.D_b_3)
            h3 = lrelu(convolutionize(h2, self.D_W_4) + self.D_b_4)
            h3 = tf.reshape(h3, [-1, 8192])
            h4 = tf.matmul(h3, self.D_W_5) + self.D_b_5
            return tf.nn.sigmoid(h4), h4
        
        def generator(z):
            h0 = tf.nn.relu(tf.matmul(z, self.G_W_1) + self.G_b_1)
            h0 = tf.reshape(h0, [-1, 4, 4, 1024])
            h1 = tf.nn.relu(tf.nn.conv2d_transpose(h0, self.G_W_2, [batch_size, 8, 8, 512], strides = [1, 2, 2, 1]) + self.G_b_2)
            h2 = tf.nn.relu(tf.nn.conv2d_transpose(h1, self.G_W_3, [batch_size, 16, 16, 256], strides = [1, 2, 2, 1]) + self.G_b_3)
            h3 = tf.nn.relu(tf.nn.conv2d_transpose(h2, self.G_W_4, [batch_size, 32, 32, 128], strides = [1, 2, 2, 1]) + self.G_b_4)
            h4 = tf.nn.conv2d_transpose(h3, self.G_W_5, [batch_size, 64, 64, 3], strides = [1, 2, 2, 1]) + self.G_b_5
            return tf.nn.tanh(h4)
        
        self.generator_sample = generator(self.Z)
        _, D_real = discriminator(self.X)
        _, D_fake = discriminator(self.generator_sample)
        
        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real, targets = tf.ones_like(D_real)))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, targets = tf.zeros_like(D_fake)))
        
        self.D_loss = self.D_loss_real + self.D_loss_fake
        
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, targets = tf.ones_like(D_fake)))
        
        self.D_optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.D_loss, var_list = backpropagate_D)
        self.G_optmize = tf.train.AdamOptimizer(learning_rate).minimize(self.G_loss, var_list = backpropagate_G)
