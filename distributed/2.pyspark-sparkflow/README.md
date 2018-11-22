## How-to

1. Run docker compose,
```bash
docker-compose -f docker-compose.yml up --build
```

2. Run any sparkflow notebooks location `sparkflow/`

When you run a notebook, the training output will be print on the terminal,
```text
master    | Partition Id: 8757c142f004440785df7fff23d8f1e7, Iteration: 43, Loss: 0.106797
master    | Partition Id: 0578b7e694ab4a7c894898d52effb6c0, Iteration: 43, Loss: 0.125814
master    | Partition Id: 8757c142f004440785df7fff23d8f1e7, Iteration: 44, Loss: 0.089998
master    | Partition Id: 0578b7e694ab4a7c894898d52effb6c0, Iteration: 44, Loss: 0.109109
master    | Partition Id: 8757c142f004440785df7fff23d8f1e7, Iteration: 45, Loss: 0.091071
master    | Partition Id: 0578b7e694ab4a7c894898d52effb6c0, Iteration: 45, Loss: 0.111739
master    | Partition Id: 8757c142f004440785df7fff23d8f1e7, Iteration: 46, Loss: 0.094186
master    | Partition Id: 0578b7e694ab4a7c894898d52effb6c0, Iteration: 46, Loss: 0.113408
master    | Partition Id: 8757c142f004440785df7fff23d8f1e7, Iteration: 47, Loss: 0.090381
master    | Partition Id: 0578b7e694ab4a7c894898d52effb6c0, Iteration: 47, Loss: 0.105846
master    | Partition Id: 8757c142f004440785df7fff23d8f1e7, Iteration: 48, Loss: 0.088096
master    | Partition Id: 0578b7e694ab4a7c894898d52effb6c0, Iteration: 48, Loss: 0.100501
master    | Partition Id: 8757c142f004440785df7fff23d8f1e7, Iteration: 49, Loss: 0.085383
```
