# Self-Optimize-Neural-Network
Self optimized Feed Forward, Recurrent and Convolutional Neural Network using Bayesian Optimization

#### 1. Feed-Forward Neural Network to classify Iris type

#### 2. Recurrent Neural Network to classify text sentiment

#### 3. Convolutional Neural Network to classify CIFAR 10

Bayesian optimization works by constructing a posterior distribution of functions (gaussian process) that best describes the function you want to optimize. As the number of observations grows, the posterior distribution improves, and the algorithm becomes more certain of which regions in parameter space are worth exploring and which are not, as seen in the picture below.

![alt text](https://raw.githubusercontent.com/fmfn/BayesianOptimization/master/examples/bo_example.png)

## Feed-Forward Iris
```text
Initialization
-------------------------------------------------------------------------------------------------------------------------
 Step |   Time |      Value |   activation |      beta |   dropout_rate |   learning_rate |   num_hidden |   size_layer | 
stop after 200 iteration with train cost 9.328038, valid cost 6.830622, train acc 0.477589, valid acc 0.344333
    1 | 00m00s |    0.34433 |       0.1557 |    0.0741 |         0.5189 |          0.6100 |       2.2780 |     109.6324 | 
stop after 200 iteration with train cost nan, valid cost nan, train acc 0.339286, valid acc 0.333333
    2 | 00m06s |    0.33333 |       1.5493 |    0.2110 |         0.9648 |          0.6746 |       8.4695 |     406.6496 | 
stop after 200 iteration with train cost 353764719888.678528, valid cost 323302490112.000000, train acc 0.407589, valid acc 0.338667
    3 | 00m19s |    0.33867 |       1.2247 |    0.0866 |         0.4345 |          0.1704 |      10.1839 |     715.3899 | 
    
.....

Maximum NN accuracy value: 0.731667
Best NN parameters:  {'dropout_rate': 0.98999999999999999, 'beta': 9.9999999999999995e-07, 'learning_rate': 0.0001, 'size_layer': 979.22638102861038, 'activation': 2.0, 'num_hidden': 2.0}
```

## Recurrent Neural Network Sentiment analysis
```text
Initialization
-------------------------------------------------------------------------------------------------------------------------------------
 Step |   Time |      Value |   activation |      beta |   dropout_rate |   learning_rate |   num_hidden |   seq_len |   size_layer | 
stop after 50 iteration with train cost 42.381874, valid cost 36.617153, train acc 0.546000, valid acc 0.500000
    1 | 00m10s |    0.50000 |       0.6067 |    0.3674 |         0.5718 |          0.6362 |       2.9054 |    6.3420 |     435.3363 | 
stop after 50 iteration with train cost 3.349305, valid cost 2.959107, train acc 0.527750, valid acc 0.500000
    2 | 00m04s |    0.50000 |       0.6907 |    0.2037 |         0.8793 |          0.1485 |       2.9604 |   14.1557 |      34.4793 | 
stop after 50 iteration with train cost 15.869358, valid cost 14.480788, train acc 0.546250, valid acc 0.500000
....

Maximum NN accuracy value: 0.554000
Best NN parameters:  {'num_hidden': 2.3178396780664174, 'dropout_rate': 0.16724952060867815, 'beta': 0.099189911765081795, 'learning_rate': 0.049638440024850142, 'size_layer': 420.96604492562358, 'activation': 1.1015571286131713, 'seq_len': 19.844388160934063}
```

## Convolutional Neural Network CIFAR 10
```text
Initialization
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Step |   Time |      Value |   activation |   batch_normalization |      beta |   dropout_rate |   fully_conn_size |   kernel_size |   learning_rate |   len_layer_conv |   multiply |   pooling_size | 
(?, 1, 1, 100)
stop after 20 iteration with train cost 2008170.116490, valid cost 486246.000000, train acc 0.151042, valid acc 0.133333
    1 | 00m06s |    0.13333 |       1.4565 |                0.8903 |    0.2951 |         0.5427 |           61.9462 |        4.0407 |          0.1692 |           4.7475 |     2.0314 |         3.4064 | 
(?, 1, 1, 40)
stop after 20 iteration with train cost 359.077862, valid cost 41.487034, train acc 0.200417, valid acc 0.131667
    2 | 00m02s |    0.13167 |       1.0275 |                0.4410 |    0.1257 |         0.6587 |           59.6901 |        3.0467 |          0.6025 |           3.9315 |     1.9162 |         3.0427 | 
....

Maximum NN accuracy value: 0.164167
Best NN parameters: {'fully_conn_size': 75.233542680827611, 'len_layer_conv': 5.0, 'activation': 2.0, 'dropout_rate': 0.98999999999999999, 'batch_normalization': 1.0, 'beta': 9.9999999999999995e-07, 'pooling_size': 4.0, 'multiply': 3.0, 'learning_rate': 0.0001, 'kernel_size': 7.0}
```
