import numpy as np

def get_vocab(data_location):
    
	with open(data_location, 'r') as fopen:
		data = fopen.read()
	
	data = data.lower()
	data = data.split('\n')
	data = filter(None, data)
	data = [i.strip() for i in data]
	data = ' '.join(data)
	data = data.split(' ')
	data = filter(None, data)
	data = [i.strip() for i in data]

	vocab = list(set(data))
    
	return data, vocab

def embed_to_onehot(data, vocab):
    
	onehot = np.zeros((len(data), len(vocab)), dtype = np.float32)

	for i in xrange(len(data)):
		onehot[i, vocab.index(data[i])] = 1.0
    
	return onehot