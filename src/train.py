import os
import sys
import json
import argparse
import numpy as np
import random
from math import ceil
from train_class import *
sys.path.append('..')
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as torchm
import matplotlib
matplotlib.use('Agg')

from networks.cifar.resnet import *
from networks.cifar.vgg import *
from utils.tensor_defns import *
from utils.data_loading_utils import *
from utils.util_functions import *
from utils.colors import bcolors
import wandb


parser = argparse.ArgumentParser(description='Arguments for Training Vanilla SGD')
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--no-logging', action='store_false', default=True)
parser.add_argument('--logging-period', type=int, default=20)
parser.add_argument('--test-name', default='0')
parser.add_argument('--test', default='Vanilla')
parser.add_argument('--seed', type=int, default=12)
parser.add_argument('--checkpoint', type=str, default="0")
parser.add_argument('--model', default='resnet')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--scale', type=str, default="0.0001")
parser.add_argument('--log-steps', type=str, default="1000,2000,3000,4000,5000")
parser.add_argument('--run-name', type=str, default="0")
parser.add_argument('--learning-rate', type=float, default=0.1)
parser.add_argument('--micro-batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=128)
parser.add_argument('--train-dataset-size', type=int, default=40000)
parser.add_argument('--test-dataset-size', type=int, default=10000)
parser.add_argument('--real-test-dataset-size', type=int, default=10000)
parser.add_argument('--batch-size', type=int, default=5120)
parser.add_argument('--save-step', type=int, default=0)
parser.add_argument('--exter-run', type=str, default='Vanilla|0.1|128-1')
parser.add_argument('--exter-lambda', type=float, default=1.)


if __name__=='__main__':
    input_args = parser.parse_args()
    set_seeds(input_args.seed)
    input_args.log_steps = [int(x) for x in input_args.log_steps.split(",")]

    trainer_obj = SGDTrainer(input_args=input_args)
    trainer_obj.train(
        num_epochs=int(10e6),
    )