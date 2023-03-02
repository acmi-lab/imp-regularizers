# Disentangling the Mechanisms Behind Implicit Regularization in SGD
This is the official implementation for the ICLR 2023 paper: [Disentangling the Mechanisms Behind Implicit Regularization in SGD](https://arxiv.org/abs/2211.15853). If you find this repository useful or use this code in your research, please cite the following paper:
​

> Novack, Z. et al. (2023). Disentangling the Mechanisms Behind Implicit Regularization in SGD. In International Conference on Learning Representations (ICLR), 2023.

```
@inproceedings{novack2023disentangling,
  title={Disentangling the Mechanisms Behind Implicit Regularization in SGD},
  author={Novack, Zachary and Kaur, Simran and Marwah, Tanya and Garg, Saurabh and Lipton, Zachary C},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year={2023}
}
```


## Requirements
To install requirements:
​
```setup
conda env create -f environment.yml
```
​
## Training
All the regularization experiments are run via `train.py` through the following:
​
* `Vanilla` - Normal small-batch SGD (batch size controlled by the `--micro-batch-size` argument)
* `PseudoGD` - Accumulated, large-batch SGD (batch size controlled by the `--batch-size` argument, and `--micro-batch-size` sets the size of the accumulated micro-batch)
* `RegLoss` - Large-batch SGD + average microbatch gradient norm regularization
* `FishLoss` - Large-batch SGD + average microbatch Fisher trace regularization
* `AvgJacLoss` - Large-batch SGD + average Jacobian regularization
* `UnitJacLoss` - Large-batch SGD + Unit Jacobian regularization
* `IterGraft` - Large-batch SGD w/Iterative Grafting
* `ExterGraft` - Large-batch SGD w/External Grafting 
* `NGD` - Normalized Large-batch SGD

Note: for `ExterGraft`, external gradient norm data is needed to run this experiment. This can be generated by first running a `Vanilla` SGD run, downloading the gradient norm data from the respective `wandb` project, and providing the correct path and column name to the file in the `--exter-path` and `--exter-col` arguments respectively.


All hyperparameters are set via the `--learning-rate, --micro-batch-size, --batch-size` and `--exter-lambda` (which controls the regularization strength) arguments. 

In order to recreate the experiments, the learning rate (η) and lambda values (λ) are listed in the tables below:

**Main Regularization Experiments**
| Model/Dataset      | SB SGD | LB SGD | LB + GN       | LB + FT       | LB + AJ        | LB + UJ        |
|--------------------|--------|--------|---------------|---------------|----------------|----------------|
| ResNet-18/CIFAR10  | η=0.1  | η=0.1  | η=0.1, λ=0.01 | η=0.1, λ=0.01 | η=0.1, λ=0.001 | η=0.1, λ=0.001 |
| ResNet-18/CIFAR100 | η=0.1  | η=0.5  | η=0.1, λ=0.01 | η=0.1, λ=0.01 | η=0.1, λ=5e-5  | η=0.1, λ=0.001 |
| VGG-11/CIFAR10     | η=0.15 | η=0.01 | η=0.01, λ=0.5 | η=0.01, λ=0.5 | η=0.01, λ=2e-5 | N/A            |

**Large Micro-batch Experiments** (note: this table is pivoted from the others for ease of reading)
| Model/Dataset      | Experiment | Learning Rate (η)| Regularization Strength (λ) |
|--------------------|------------|------------------|-----------------------------|
| ResNet-18/CIFAR10  | LB + GN (mb size=2560)    | 0.5  | 0.0025  |
| ResNet-18/CIFAR100 | LB + FT (mb size=2560)    | 0.1  | 0.01  |
| VGG-11/CIFAR10     | LB + GN (mb size=1000)    | 0.01 | 0.25 |
| VGG-11/CIFAR10     | LB + GN (mb size=2500)    | 0.01 | 0.25 |

**Sample Micro-batch Experiments**
| Model/Dataset      | SB SGD | LB + GN (mb size = 50) | LB + GN (mb size = 100) | LB + GN (mb size = 1000) | LB + GN (mb size = 2500) | 
|--------------------|--------|--------|--------------------|-------------------|-----------|
| VGG-11/CIFAR10     | η=0.01 | η=0.01, λ=0.25  | η=0.01, λ=0.5    | η=0.01, λ=0.5      | η=0.01, λ=0.5 |


**Grafting Experiments**
| Model/Dataset      | SB SGD | LB SGD | Iterative Grafting | External Grafting | NGD       | 
|--------------------|--------|--------|--------------------|-------------------|-----------|
| ResNet-18/CIFAR10  | η=0.1  | η=0.1  | η=0.1              | η=0.1             | η=0.2626  |
| ResNet-18/CIFAR100 | η=0.1  | η=0.5  | η=0.1              | η=0.1             | η=0.3951  |
| VGG-16/CIFAR10     | η=0.05 | η=0.1  | η=0.05             | η=0.05            | η=0.2388  |
| VGG-16/CIFAR100    | η=0.1  | η=0.1  | η=0.1              | η=0.1             | η=0.4322  |


For example, running the following command trains a Resnet-18 on CIFAR-10 with average micro-batch gradient norm regularization (where batch size is 5120, learning rate is 0.1, regularization penalty is 0.01, and micro-batch size is 128)
​
```setup
python train.py --model='resnet' --dataset='cifar10' --batch-size=5120 --learning-rate=0.1 --exter-lambda=0.01 --micro-batch_size=128 --test='RegLoss'
```
​
## Evaluation
After training is complete, the model can be evaluated using `eval.py`. As long as the `--no-logging` flag is not turned on during training, the best performing model (in terms of validation accuracy) will be saved within a `saved_models/run_name` directory as `checkpoint_best.pth`. To evaluate the model, we must provide the path to this file in the `--path` argument to `eval.py`.
​
Building off of our Resnet-18 example earlier, we can run the following command to obtain the final test accuracy:
​
```setup
python eval.py --model='resnet' --dataset='cifar10' --batch-size=5120 --lr=0.1 --exter-lambda=0.01 --micro-batch_size=128 --test='RegLoss' --path='saved_models/run_name/checkpoint_best.pth'
```
​
## Results
​
Our models achieves the following test accuracies for various regularization penalties:

**Main Regularization Experiments**
| Model/Dataset      | SB SGD | LB SGD | LB + GN | LB + FT | LB + AJ | LB + UJ |
|--------------------|--------|--------|---------|---------|---------|---------|
| ResNet-18/CIFAR10  | 92.64  | 89.83  | 91.75   | 91.50   | 90.13   | 90.15   |
| ResNet-18/CIFAR100 | 71.31  | 67.27  | 70.65   | 71.20   | 66.08   | 66.26   |
| VGG-11/CIFAR10      | 78.19  | 73.90  | 77.62   | 78.40   | 74.09   | N/A    |

**Large Micro-batch Experiments** (note: this table is pivoted from the others for ease of reading)
| Model/Dataset      | Experiment | Test Accuracy |
|--------------------|------------|---------------|
| ResNet-18/CIFAR10  | LB + GN (mb size=2560)    | 65.09  |
| ResNet-18/CIFAR100 | LB + FT (mb size=2560)    | 64.89  |
| VGG-11/CIFAR10     | LB + GN (mb size=1000)    | 75.07 |
| VGG-11/CIFAR10     | LB + GN (mb size=2500)    | 75.21 |

**Sample Micro-batch Experiments**
| Model/Dataset   | SB SGD (η=0.15) | SB SGD (η=0.01) | LB SGD (η=0.01) | LB + GN (mb size = 50) | LB + GN (mb size = 100) | LB + GN (mb size = 1000) | LB + GN (mb size = 2500) |
|----------------------|---------------------|---------------------|---------------------|----------------------------|------------------------------|-------------------------------|-------------------------------|
| VGG-11/CIFAR10 (best) | 78.19          | 75.94          | 73.90          | 77.34               | 77.23                | 75.73                 | 75.64                 |
| VGG-11/CIFAR10 (final) | 77.56          | 74.73          | 73.60          | 76.57               | 76.60                | 75.48                 | 75.36                 |

**Grafting Experiments**
| Model/Dataset      | SB SGD | LB SGD | Iterative Grafting | External Grafting | NGD       | 
|--------------------|--------|--------|--------------------|-------------------|-----------|
| ResNet-18/CIFAR10  | 92.64  | 89.83  | 92.12              | 92.16             | 92.10  |
| ResNet-18/CIFAR100 | 71.31  | 67.27  | 68.30              | 68.40             | 66.83  |
| VGG-16/CIFAR10     | 89.56  | 86.97  | 88.65              | 89.06             | 89.39  |
| VGG-16/CIFAR100    | 64.26  | 55.94  | 59.71              | 63.48             | 58.05  |
