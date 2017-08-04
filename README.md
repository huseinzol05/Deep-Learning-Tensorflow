# Deep-Learning-Tensorflow
### Purely Tensorflow, no Keras or other abstract libraries of Tensorflow

![alt text](https://lh3.googleusercontent.com/hIViPosdbSGUpLmPnP2WqL9EmvoVOXW7dy6nztmY5NZ9_u5lumMz4sQjjsBZ2QxjyZZCIPgucD2rhdL5uR7K0vLi09CEJYY=s688)

## Dependencies
```bash
sudo pip install scipy numpy matplotlib librosa pandas seaborn
```
- I recommended install Tensorflow from source, way more faster
- If you got GPU, compile it with CUDA

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
## DCGAN Model
![alt text](http://img.blog.csdn.net/20160918133222494)
Simplify a lot of papers for DCGAN

## DCGAN
![alt text](data:image/png;base64,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%0ApuvXWFftps8+95Jtv5/Z7/730jlrizd8fuXWlj9f7ZUt/94mJ/LowN9v2sLXiO13p9ve+deftdtO%0A5v697+z4a2h2OT0emvvmHpt9YnzgLAKv/5MYY2ZUAsCFoD4BGCJqE4AsfvURAAAAAAbmcSdqUdK/%0ADSH8fgjh/WfRIQA4I9QnAENEbQJwIo/7q4/fFGN8PYTwrKRfDyH8aYzxtx7+guMi9H5J2rl86THf%0ADgBOzNanh2vT9pb/2x8AOEOnenaajvzfuQJ4ej3WT9RijK8f//+WpF+R9N63+JoPxBhfjjG+vLGZ%0AWZwCAM5Irj49XJumU2oTgPNx2menUX0WywkAeDt65IlaCGEzhLD95seS/q6kj59VxwDgUVGfAAwR%0AtQnAaTzOt2mek/Qr4cHa3JWk/zPG+K/tFn1UmKeXPt6+lP7xfjf3yzE3V7Zte5ill9+XpMUs/R31%0A1cQvOdoVfhnqI7Pk+kbvl3ndu+eXoJ2/wy97vJyll2vuCr8k6Wzpl+WNVXqev1r5ZV7blT+f432/%0A34tpun07swb+OPjla+vOt5ej9HHrev+9j9IsNyxJ5WG6rd/wx0RmCXlJqmq/X+tZuu+jIr2Ec9Ag%0A1+c/VX2KQWrMeC6KdN2aZJZrP8pEbFSZ5bFH43T7tMksaVz5c1OaaJGROR6S1G749y72bbPUm8iV%0A4JfI79eZyINgrpUj3+/M6VTY8HWxuJTer3Xrr9HRjr/XXCv8NbxfpjtfycdIKLPKfVil71Uh+o1D%0AZiytMxkfu/P0/bkd5UJPBufUz059H3W0TO9n64aFu84kxWBuOpLCyv+2QTtJX2v13L93X/t7Wi/z%0AbFX5pc2rua8R7RX/3LbcTB/v2f3Ms9HC169Qp997/E6//P72C/58bN31S8m/MrudbDtY+n5XU19/%0ALi99+2GTXuR01fql/6djX79e6q4k23YzcVCv3vDP4bPOv/fobnq/Z2bOo/4JL88fY3xV0n/0qNsD%0AwJNCfQIwRNQmAKfB8vwAAAAAMDBM1AAAAABgYJioAQAAAMDAMFEDAAAAgIFhogYAAAAAA8NEDQAA%0AAAAG5lzj7suq0O6VdI5MNU13Z1L5/JmdTO7OovCZGn1jts9kghUbPtdibLr+ucLnO4Rn/Skatz4T%0ASBvpvI515zNMdgv/3kVMv3cz9t8D6DPRN8tdn8tTL9K5FfGKz+RZZfarjL69MdkXI5OZJ0ltJjaj%0AnKSPab/2+1UEPxb6zOV+dTed4TSq0+O0zmR1vR0ESaXMyTHHNla5nLRMxuM4kxnWpc9b2PLHfmPi%0Ar8NQpl97q8hcw1N/Ec+XvubOpyaLaObzl4qNPdveLtJ9n4981s9m4TN1+tYHrY1ier/7DX/Mdnt/%0AnzvY8mNpY236nhlnIfrzXZmxVkd/D5TfLY1bX5v6UToHNfRv//qTF1TF9H72Jl+v7X0GVF35ay2M%0A/HNCFdN1s7jka2Ps/X2+MrWvCP467XLjYuXHezxI9/2g8Me0q/x+Tev0MX/ukq9t833f/urr6awy%0ASarnpu6avFBJGke/3zf2fH1r5+lzMt32Y+XK+Bnb3jxncgYr3+/x8jXbfuO1z9v2I/NMuSrTY6HP%0A5Ee+iZ+oAQAAAMDAMFEDAAAAgIFhogYAAAAAA8NEDQAAAAAGhokaAAAAAAwMEzUAAAAAGBgmagAA%0AAAAwMOeaoxZCobo0eSt1OlOg2vHZN/fWO7Z9VfntN+bpnIW9nandtlr5vKDD9jDZVmS2XWays/rS%0AZ/rUTToDpVn70x99fIpiaXJdep/pVUSf3TKe37PtYdPkq8wzx6TwGSd98FkgZUxn0ymTi1E2mewW%0A895d6UPYukw+XLX210A3MselN699wiyQIQshaFSl97Er0mOqNNtJ0iSTyzVf+mthOkpn25STTAmf%0AmLEqaTuka1smVktV5hpXNHk9khTSfatKn3ujTN5YKNPZl0XIZLSVfsc7X14UTDbmqPc5aKsqk2PU%0A+fPZbqev4dBcsduGKn2fkqR+lN7xOpMtVzc+Z21V7dt2meuvGr39609Or16zmB5XReOuJT9muiqT%0Av7nK5JKarMbx3D87rWs/3t0tqyr9a7e9z+VadL59uryZbpz58b6x8s+j1bPpa7HNPBPe/9Pbtn32%0A2hu2/W6fvtZGM3+dmnhKSdIkk4V2f5kea9Xcj8PNTf/88qK5D9/+c38v6luflRyif2a8arOW089t%0A/sr8S/xEDQAAAAAGhokaAAAAAAwMEzUAAAAAGBgmagAAAAAwMEzUAAAAAGBgmKgBAAAAwMCc6/L8%0AdVHo+e30kqp9m16eczH3y2O65fUlaX7ol4Hd30zPWTcmflnj8YZfx767kz7Mq6Vfcj1EvzTu5tIv%0Ah7rq0u+9ODyw2y62/DLWWzG9jGyVWUY6jHz7eCOzhHafXqp1d8ePldqv8qpR5ZfHLcyy4iFmliyv%0A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%0AMcafDCFckfQLkt4l6dOSvjvGzBqWsZfa9LKmh5ol28Zbfin5mV81VPOrfhnMnXX6UKwX6UgBSSoz%0AS8yuzXLO3cIvX1vu+34fTf0StRu3t5NtxTW/FOtG6Q9q2aXPyXqUPpeS1BV+idmRb9ZiK923K0d+%0AOfSQWSE/VH659MJ8fyO3um2RWY21N1/QB9+vLrNOfmwysQNmmXi3pO9FOcvaFCXFkD4+bZmuD7H1%0A421U+aV/VxM/aLp1epnoMrMcsnJLORfpvsdMv5SJFYgmVkCSZF6+mvr3LlsfHbI06/MXcddu2418%0A3Qt1uqZKUmU2bzJrUK/XvkDUI1+8iia9fchEgzSlX9a7KNNjrVn5e2TmFqll8NfQpomCiSEzTi/I%0AmdanGNR26eMfTbRC1WdiNDLXWp9Zxn61ZZaxX2eejUzsiSSFkB6TMXNPWmfqT++HrI08WBf+vZvM%0Ao3VrLvNMwo/WmTiKceevpW6Vfubc6+Z220Y+lqDKRNG0V9Ptu8/4Y7otX3dfupxenv/W6LLdtrni%0Ax+H8ZuY5vftMsm21MLUrE5n0ppM8fbWS/kmM8esk/Q1J/ziE8HWSflDSb8QY3y3pN47/DQDnhdoE%0AYKioTwAeW3aiFmO8EWP8g+OPDyV9QtKLkr5D0oePv+zDkr7zSXUSAL4UtQnAUFGfAJyFU/0+Uwjh%0AXZK+QdLvSHouxnjjuOkNPfjxPgCcO2oTgKGiPgF4VCeeqIUQtiT9kqQfiDEePNwWY4x68DvYb7Xd%0A+0MIr4QQXjk69L/nCQCndRa1aT7zf1MJAI/iLOpT22X+uBnAU+tEE7Xw4C86f0nSz8YYf/n40zdD%0ACNeP269LuvVW28YYPxBjfDnG+PLWdvqP/QDgtM6qNm1s+sWKAOC0zqo+VWYxIwBPt+xELYQQJP2M%0ApE/EGH/8oaaPSHrf8cfvk/SrZ989AHhr1CYAQ0V9AnAWTvJtmr8l6R9I+qMQwkePP/dDkn5U0i+G%0AEL5P0mckffeT6SIAvCVqE4Choj4BeGzZiVqM8belZEDT3znNm8VYqOknyfbp/XSmwGLu8x2K1dS2%0AjyufX7Ms0n8/t29yjCRpuvL5NO0q/fcvi8yvnjfTXM6I71tbptu72uc7rVc+X2VqDmkmgk3rzI4v%0AFj4DJeymf1WtmPj9Uu3zgvrOn89YpLevGr9tV/hjWsmdT79tbbeV+szPz3uTVdRV6f2K/tJ6Ys6y%0A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%0A/pZd5lrZ8p1rY/r1N7Z9v6uRP59fv/bts5deSLbN/5rPKtuc+b5tPZOuy93c92unSecoS1L4vK/L%0Au1s3k23xID03KU+YQstP1AAAAABgYJioAQAAAMDAMFEDAAAAgIFhogYAAAAAA8NEDQAAAAAGhoka%0AAAAAAAwMEzUAAAAAGJhzzVHrY6fV8iDZfniQztRoOp+T9tzMZxXd2vMZDUWXDt9a7vgMhmKWme/2%0A6ey4126nM0gkaee6z454fuWPiw7uJZvuftYHjk22/PDom3SOyb31G3bbW/f9fm9u+GNamvyVuOfz%0Am5Y7Prui3fT5K0WRPm5F6cdKyGSOdSZWpjKZV5Ikm3Umxdz3ZYp00FJXmOPtX/VtIYSgsjaZPU06%0Ao2UvkzdWdP68zFp/na1W6VyclRswksqQCfwx+VSL3Fg1178ktSb/TZJapfvWNZksw4Wv563JQpzE%0ATL8y2U995jIambzCss7kw0Wf0dZlzklZpfctZjIeY/D1pTXZl+Pg63nMZFopc1xWJnOvLp6GCuQV%0AKjQt0vf6vkoPyqORv1bK1g/oSr6+NU363HZlJsgxY2r61td+TNWZfLhi7HPUpkrnlZXb/tnp8qa/%0AHtaT9Lkc7/l+b1/1z3yXdq7a9tXM3E8muWxMnye2vOTvZVfNlGMr+nHYjPx+x530Mb8688d03vnn%0A7NEVn6W816WP+fZGes6jTBbgF77sRF8FAAAAADg3TNQAAAAAYGCYqAEAAADAwDBRAwAAAICBYaIG%0AAAAAAAPDRA0AAAAABiY7UQshTEIIvxtC+MMQwh+HEH7k+PNfGUL4nRDCJ0MIvxBC8GudAsAZoz4B%0AGCJqE4CzcJIctZWkb4kxHoUQakm/HUL4fyT915J+Isb48yGEfy7p+yT9lHuh2Bdq5ulsiuVBOm/g%0AksmmkaRbK58ntDXesu0HB/vJtvWBz5ZYFT63qw3pbJx+7HMpmnjZth8Gv/380+ljurHpMxzm9/w8%0Avpim8+EOFj6XYnvsc0ZuLq/b9tEL6ffeGmXyOJ71Y2n/ps/r2HwunUNSZS6pzdpn/jQxPU5DZtu1%0AycSSpEkm02a1So/Tok/vsz+aT9zZ1KeyULmVPvbdYfoarzt/zmcmn06Sis7nHK1Nrle7zoynOpPx%0AtkhvP8pkAnaNr7md/H7FNn0v6PvMttEf8yKm62I/ymwb/IieZO6avbki6rGvi3Hh67kyWWihMDls%0A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%0ARAUAAAAASUVORK5CYII=%0A)
Trained Discriminate and Generative networks to generate house numbers

## Deep Convolutional

1. trained to label 100 classes
![alt text](https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/Deep%20Convolutional/100-classes/sample.png)
2. trained to label multitags, a single picture can be more than 1 tag
![alt text](https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/Deep%20Convolutional/multilabel/Screenshot%20from%202017-08-04%2010-08-25.png)
3. trained to predict pokemon type
![alt text](https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/Deep%20Convolutional/pokemon-type/download.png)
## Deep Recurrent
1. trained to predict stock market
![alt text](https://raw.githubusercontent.com/huseinzol05/Predicting-Stock-Recurrent-Neural-Network/master/output/latestunited.png)
2. trained to generate sentence
```text
mercy; the fool
Has received them? and now out still!
I will spironed this brat, gentleman,
Whoreson's equally to that.

KING RICHARD III:
Belowlance to the rige, come.
```
3. trained to classify any length of sound [In training]

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
6. sound-classification [in training]

#### 9- Introduction on layer normalization
![alt text](https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/batch-normalization/Screenshot%20from%202017-08-04%2010-24-08.png)
#### 10- Encoder model, both multi-perceptron and Convolutional
1. multi-perceptron
![alt text](https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/encoder/download%20(3).png)
2. Convolutional
![alt text](https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/encoder/download%20(4).png)
#### 11- Word vector both using softmax and NCE
![alt text](https://raw.githubusercontent.com/huseinzol05/Deep-Learning-Tensorflow/master/wordvector/download%20(1).png)
