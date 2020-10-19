# Domain Intersection and Domain Difference ([arxiv](https://arxiv.org/abs/1908.11628)).

Pytorch Implementation of "Domain Intersection and Domain Difference" (ICCV 2019)

## Prerequisites
- Python 3.6+
- Pytorch 0.4

### Rotations Prediction
Run ```rotation_prediction/main.py```. You can use the following example to run:
```
python main.py --lr 0.001 --epochs 10
```

### Validating Assumption 1
Run ```validating_assumption_1/main.py```. You can use the following example to run:
```
python validating_assumption_1/train.py --lr 0.01 --epochs 100 --dataset MNIST
```

### Validating Assumption 2
Run ```validating_assumption_2/train.py```. You can use the following example to run:
```
python validating_assumption_2/train.py --lr 1.0 --epochs 5 --dataset MNIST
```

## Reference
If you found this code useful, please cite the following paper:
```
@inproceedings{galanti2020modularity,
  title={On the Modularity of Hypernetworks},
  author={Tomer Galanti and Lior Wolf},
  booktitle={NeurIPS},
  year={2020}
}
```
