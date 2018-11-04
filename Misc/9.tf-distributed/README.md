## How-to

1. Run the main supervisor called `ps`,
```bash
python3 tf-distributed.py --job_name=ps --task_index=0
```

```text
2018-11-05 00:29:14.592372: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-11-05 00:29:14.599671: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:222] Initialize GrpcChannelCache for job ps -> {0 -> localhost:2222}
2018-11-05 00:29:14.599699: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:222] Initialize GrpcChannelCache for job worker -> {0 -> localhost:2223, 1 -> localhost:2224, 2 -> localhost:2225}
2018-11-05 00:29:14.601901: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:381] Started server with target: grpc://localhost:2222
```

2. Open new tab, run worker manager, index 0,
```bash
python3 tf-distributed.py --job_name=worker --task_index=0
```

```text
2018-11-05 00:30:13.797170: I tensorflow/core/distributed_runtime/master.cc:267] CreateSession still waiting for response from worker: /job:worker/replica:0/task:1
2018-11-05 00:30:13.797230: I tensorflow/core/distributed_runtime/master.cc:267] CreateSession still waiting for response from worker: /job:worker/replica:0/task:2
```

It saying it is waiting for worker 1 and 2.

3. Open new tab, run worker 1,
```bash
python3 tf-distributed.py --job_name=worker --task_index=1
```

4. Open new tab, run worker 2,
```bash
python3 tf-distributed.py --job_name=worker --task_index=2
```

randomly, the output will print between worker 0 - 2,
```text
iteration 116, loss 1.014565
iteration 117, loss 0.124767
iteration 118, loss 0.185887
iteration 119, loss 0.101931
iteration 120, loss 0.230364
iteration 121, loss 0.222517
iteration 122, loss 0.282432
iteration 123, loss 0.470098
iteration 124, loss 0.516637
iteration 125, loss 0.265697
iteration 126, loss 0.253511
iteration 127, loss 0.164691
iteration 128, loss 0.075810
iteration 129, loss 0.077431
iteration 130, loss 0.165390
iteration 131, loss 0.242567
```

5. Open tensorboard located in `test-tf-distributed`,
```bash
tensorboard --logdir=test-tf-distributed
```

![alt text](tensorboard.png)
