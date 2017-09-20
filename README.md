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
```text
input: [[  6  80 940 941   0   0   0]]
supposed label: [[ 20   9 955  10 956   2 957   1]]
predict label:[[ 27   9 955  10 956   2 957   1]]
predict text: Kita kau terlahir di dunia yang damai, 

input: [[997 368   7 998   0   0]]
supposed label: [[1021   27  140   14 1022 1023    1]]
predict label:[[  27   27  140   14 1022 1023    1]]
predict text: Kita Kita mula dengan proses penyejukkan. 
```

## Chatbot-Attention-Seq2Seq
Generate chatbot using attention model on Sequence-to-Sequence Tensorflow API
```text
sentence: 1
input: bernama The Company
predict respond: Keadaan tidak 
actual respond: Keadaan tidak baik Awak?

sentence: 2
input: Ruparupanya awak boleh
predict respond: Dia pulihkan akan 
actual respond: Dia cakap dia dapat lihat
```

## DCGAN (Simplify and Original for House Number)
<img src="https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/DCGAN/example.png" alt="Drawing" height="200"/>

## WGAN Improvement
## DiscoGAN (original paper and Fashion MNIST)
## Residual Network for CIFAR-10
<img src="http://yanran.li/images/resnet_2.png" alt="Drawing" height="200"/>

## Deep Convolutional
1. trained to label 100 classes
<img src="https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/Deep%20Convolutional/100-classes/sample.png" alt="Drawing" height="200"/>

2. trained to label multitags, a single picture can be more than 1 tag
<img src="https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/Deep%20Convolutional/multilabel/Screenshot%20from%202017-08-04%2010-08-25.png" alt="Drawing" height="200"/>

3. trained to predict pokemon type
<img src="https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/Deep%20Convolutional/pokemon-type/download.png" alt="Drawing" height="200"/>

## Deep Recurrent
1. trained to predict stock market
<img src="https://raw.githubusercontent.com/huseinzol05/Predicting-Stock-Recurrent-Neural-Network/master/output/latestunited.png" alt="Drawing" height="200"/>

2. trained to generate sentence
```text
mercy; the fool
Has received them? and now out still!
I will spironed this brat, gentleman,
Whoreson's equally to that.

KING RICHARD III:
Belowlance to the rige, come.
```
3. trained to classify any length of sound
4. trained to classify sentiment
```text
total accuracy during training: 0.998590225564
epoch: 20, loss: 0.751066319476, speed: 0.265403051692s / batch
total accuracy during testing: 0.722925
             precision    recall  f1-score   support

   negative       0.72      0.74      0.73      1075
   positive       0.73      0.70      0.72      1058

avg / total       0.72      0.72      0.72      2133
```

## Essay-Attention-Seq2Seq
Generate simplified sentence for an essay using Attention Seq2Seq
```text
actual text: Pemberian kerja rumah bermotif untuk memupuk unsur pembelajaran kendiri dalam sanubari murid. Kerja rumah turut berperanan sebagai aktiviti pengukuhan bagi pembelajaran di dalam kelas. Tambahan pula, kerja rumah memberi peluang keemasan kepada ahli-ahli keluarga untuk bersama dengan anak-anak semasa mereka belajar. Malahan, kerja rumah merupakan platform kejayaan murid-murid dalam pelajaran kerana banyak latihan yang dilakukan. Lebih-lebih lagi, kerja rumah diberikan bertujuan untuk mengisi masa lapang mereka dengan aktiviti berfaedah yang mampu mendorong mereka berjaya dalam pelajaran

predict text: Pemberian kerja rumah bermotif memupuk unsur pembelajaran kendiri dalam sanubari murid. Kerja rumah turut berperanan sebagai anak-anak kerja bagi pembelajaran di dalam kelas. Tambahan memberi peluang keemasan kepada ahli-ahli keluarga untuk bersama dengan aktiviti kerja belajar. Malahan, kerja rumah merupakan platform kejayaan murid-murid dalam pelajaran banyak unsur yang dilakukan. Lebih-lebih lagi, kerja rumah diberikan bertujuan untuk mengisi masa lapang mereka dengan yang mampu mendorong mereka berjaya dalam pelajaran kerja Tambahan aktiviti kerja 
````
## Multi-Perceptron

1. Creditcard detection (softmax, l2 loss, 4 hidden layers)
```text
testing accuracy: 0.998244
             precision    recall  f1-score   support

        non       1.00      1.00      1.00     56865
      fraud       0.00      0.00      0.00        97

avg / total       1.00      1.00      1.00     56962
```
2. detect-voice (softmax, dropout, l2 loss, 4 hidden layers)
```text
testing accuracy: 0.968454
             precision    recall  f1-score   support

     female       0.96      0.97      0.97       319
       male       0.97      0.96      0.97       315

avg / total       0.97      0.97      0.97       634
```
3. iris (3 hidden layers, softmax)
```text
testing accuracy: 0.966667
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        12
Iris-versicolor       1.00      0.91      0.95        11
 Iris-virginica       0.88      1.00      0.93         7

    avg / total       0.97      0.97      0.97        30
```
4. pokemon (4 hidden layers, softmax)
```text
testing accuracy: 0.2
             precision    recall  f1-score   support

        Bug       0.36      0.33      0.34        15
       Dark       0.00      0.00      0.00         8
     Dragon       0.15      0.29      0.20         7
   Electric       0.57      0.40      0.47        10
      Fairy       0.00      0.00      0.00         1
   Fighting       0.29      0.50      0.36         4
       Fire       0.06      0.12      0.08         8
     Flying       0.00      0.00      0.00         0
      Ghost       0.25      0.29      0.27         7
      Grass       0.00      0.00      0.00        15
     Ground       0.00      0.00      0.00         6
        Ice       0.33      0.12      0.18         8
     Normal       0.41      0.35      0.38        20
     Poison       0.00      0.00      0.00         3
    Psychic       0.44      0.40      0.42        10
       Rock       0.00      0.00      0.00         5
      Steel       0.50      0.14      0.22         7
      Water       0.17      0.12      0.14        26

avg / total       0.24      0.20      0.21       160
```
5. sentiment (6 hidden layers, batch normalization, l2 loss, dropout)
```text
total accuracy during testing: 0.730427
total accuracy during training: 0.999881628788
epoch: 20, loss: 688.440786651, speed: 0.635681152344 s / batch
total accuracy during testing: 0.729958
             precision    recall  f1-score   support

   negative       0.77      0.67      0.71      1070
   positive       0.70      0.79      0.75      1063

avg / total       0.73      0.73      0.73      2133

'this is a film well worth seeing'
output: [[-7356.93554688 -5659.93994141]]
Normalized: [[-0.79258448 -0.60976213]]
[[ 0.45442131  0.54557872]]
[[ 0.  1.]]
```
6. sound-classification

## Introduction on layer normalization
<img src="https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/batch-normalization/Screenshot%20from%202017-08-04%2010-24-08.png" alt="Drawing" height="200"/>

## Encoder model, both multi-perceptron and Convolutional
1. multi-perceptron
<img src="https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/encoder/download%20(3).png" alt="Drawing" height="200"/>

2. Convolutional
<img src="https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/encoder/download%20(4).png" alt="Drawing" height="200"/>

## Word vector both using softmax and NCE
<img src="https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/wordvector/download%20(1).png" alt="Drawing" height="200"/>
