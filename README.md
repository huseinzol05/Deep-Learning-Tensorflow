# Deep-Learning-Tensorflow
### Purely Tensorflow, no Keras or other abstract libraries of Tensorflow

<img src="https://lh3.googleusercontent.com/hIViPosdbSGUpLmPnP2WqL9EmvoVOXW7dy6nztmY5NZ9_u5lumMz4sQjjsBZ2QxjyZZCIPgucD2rhdL5uR7K0vLi09CEJYY=s688" alt="Drawing" height="200"/>

The code are lack of comments, sorry for that. I will add it later.

## Dependencies
```bash
sudo pip install scipy numpy matplotlib librosa pandas seaborn
```
- I recommended install Tensorflow from source, way more faster
- If you got GPU, compile it with CUDA
- You need to download CIFAR-10, CIFAR-100

## Basic-Seq2Seq
Generate encoder and decoder by creating 2 Deep Recurrent Neural Network to predict incoming text
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/Basic-Seq2Seq/basic%20sequence-to-sequence.ipynb)

## Chatbot-Attention-Seq2Seq
Generate chatbot using attention model on Sequence-to-Sequence Tensorflow API
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/Chatbot-Attention-Seq2Seq/chatbot.ipynb)

## DCGAN (Simplify and Original for House Number)
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/DCGAN/DCGAN.ipynb)

## WGAN Improvement
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/WGAN-improve/DC-WGAN-Improve.ipynb)

## DiscoGAN (original paper and Fashion MNIST)
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/DiscoGAN/discogan.ipynb)

## Residual Network for CIFAR-10
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/residual/residual.ipynb)

## Deep Convolutional
1. trained to label 100 classes
[link folder](https://github.com/huseinzol05/Deep-Learning-Tensorflow/tree/master/Deep%20Convolutional/100-classes)

2. trained to label multitags, a single picture can be more than 1 tag
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/Deep%20Convolutional/multilabel/nn.ipynb)

3. trained to predict pokemon type
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/Deep%20Convolutional/pokemon-type/convnet.ipynb)

## Deep Recurrent
1. trained to predict stock market
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/Deep%20Recurrent/stock-prediction/recurrent.ipynb)

2. trained to generate sentence
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/Deep%20Recurrent/generate-text/nn.ipynb)

3. trained to classify sentiment
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/Deep%20Recurrent/sentiment/nn.ipynb)

## Essay-Attention-Seq2Seq
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/Essay-Attention-Seq2Seq/ringkasan-using-attentionmodel.ipynb)

## Multi-Perceptron
1. Creditcard detection (softmax, l2 loss, 4 hidden layers)
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/Multi-Perceptron/credicard-detection/nn.ipynb)

2. detect-voice (softmax, dropout, l2 loss, 4 hidden layers)
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/Multi-Perceptron/detect-voice/nn.ipynb)

3. iris (3 hidden layers, softmax)
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/Multi-Perceptron/iris/nn.ipynb)

4. pokemon (4 hidden layers, softmax)
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/Multi-Perceptron/pokemon/nn.ipynb)

5. sentiment (6 hidden layers, batch normalization, l2 loss, dropout)
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/Multi-Perceptron/sentiment/neuralnetwork.ipynb)

## Introduction on layer normalization
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/batch-normalization/batch-normalization.ipynb)

## Encoder model, both multi-perceptron and Convolutional
1. multi-perceptron
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/encoder/Encoder-simple.ipynb)

2. Convolutional
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/encoder/Encoder-Convolutional.ipynb)

## Word vector both using softmax and NCE
1. softmax
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/wordvector/wordvector_softmax.ipynb)

2. NCE
[link notebook](https://github.com/huseinzol05/Deep-Learning-Tensorflow/blob/master/wordvector/wordvector_nce.ipynb)
