from model import *
import parse
import tensorflow as tf
import numpy as np
import os
import random
import time

data, vocab = parse.get_vocab('essay')
onehot = parse.embed_to_onehot(data, vocab)

# hyperparameters
## learning rate for both discriminator and generator
learning_rate = 0.0001
## sequence length for RNN models
length_sentence = 64
## batch size for RNN models
batch_size = 20
## how many iteration for discriminator and generator training session
epoch = 100
## how many iteration for sequence model to train
nested_epoch = 15
## hidden layers of RNN models
num_layers = 2
## size layers of RNN models
size_layer = 512
len_noise = 100
## initial tag length for generator part
tag_length = 5
possible_batch_id = range(len(data) - batch_size)

sess = tf.InteractiveSession()
model = Model(num_layers, size_layer, len(vocab), len_noise, length_sentence, learning_rate)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
sample_z = np.random.uniform(-1, 1, size = (1, length_sentence, len_noise))
start_tag = random.randint(0, len(data) - tag_length)
tag = data[start_tag: start_tag + tag_length]

DISC_LOSS, GEN_LOSS = [], []
for i in xrange(epoch):
	last_time = time.time()
	random_sample = np.random.uniform(-1, 1, size = (batch_size, length_sentence, len_noise))
	batch_fake = np.zeros((batch_size, length_sentence, len(vocab)))
	batch_true = np.zeros((batch_size, length_sentence, len(vocab)))
	batch_id = random.sample(possible_batch_id, length_sentence)

	for n in xrange(batch_size):
		id1 = [k + n for k in batch_id]
		batch_fake[n, :, :] = onehot[id1, :]
		start_random = random.randint(0, len(data) - length_sentence)
		batch_true[n, :, :] = onehot[start_random: start_random + length_sentence, :]

	disc_loss, _ = sess.run([model.d_loss, model.d_train_opt], feed_dict = {model.noise: random_sample, model.fake_input: batch_fake, model.true_sentence: batch_true})
	gen_loss, _ = sess.run([model.g_loss, model.g_train_opt], feed_dict = {model.noise: random_sample, model.fake_input: batch_fake, model.true_sentence: batch_true})

	print 'epoch: ' + str(i + 1) + ', discriminator loss: ' + str(disc_loss) + ', generator loss: ' + str(gen_loss) + ', s/epoch: ' + str(time.time() - last_time)
	for nested in xrange(nested_epoch):
		seq_loss, _ = sess.run([model.seq_loss, model.seq_opt], feed_dict = {model.noise: random_sample, model.fake_input: batch_fake, model.true_sentence: batch_true})
		print 'epoch: ' + str(nested + 1) + ', sequence loss: ' + str(seq_loss)
	DISC_LOSS.append(disc_loss)
	GEN_LOSS.append(gen_loss)

	if (i + 1) % 5 == 0:
		print 'checkpoint: ' + str(i + 1)
		print 'generated sentence: '
		tag = generate(sess, length_sentence, sample_z, model, tag, length_sentence, vocab)
		print ' '.join(tag)
