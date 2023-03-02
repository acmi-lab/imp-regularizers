import os
import sys
import ipdb
import argparse
from torchvision import datasets, transforms
sys.path.append('..')
from utils.tensor_defns import *

def load_dataset(self, dataset_type, act_true=False):
    '''Function for Loading the dataset
    '''
    if dataset_type =='cifar100':
        print('==> Preparing data..')
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])
        self.train_dataset = datasets.CIFAR100(
            root='./data', 
            train=True, 
            download=True, 
            transform=self.transform_train)
        self.train_dataset = torch.utils.data.Subset(
            self.train_dataset, range(0,self.args.train_dataset_size))
        
        self.test_dataset = datasets.CIFAR100(
            root='./data',
            train=True, 
            download=True, 
            transform=self.transform_test)
        self.test_dataset = torch.utils.data.Subset(
            self.test_dataset, range(self.args.train_dataset_size, self.args.train_dataset_size + self.args.test_dataset_size))
        

    if dataset_type == 'cifar10':
        # Data
        print('==> Preparing data..')
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_dataset = datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=self.transform_train)
        self.train_dataset = torch.utils.data.Subset(
            self.train_dataset, range(0, self.args.train_dataset_size))

        self.test_dataset = datasets.CIFAR10(
            root='./data',
            train=False, 
            download=True, 
            transform=self.transform_test)
        self.test_dataset = torch.utils.data.Subset(
            self.test_dataset,  range(0, self.args.test_dataset_size))
        self.classes = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4:'deer', 
                        5: 'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}


def load_dataloader(self, secondary_shuffle=True, sdl_bs=128):
    self.dataloader = torch.utils.data.DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=self.num_workers)

    self.secondary_dataloader = torch.utils.data.DataLoader(
        self.train_dataset,
        batch_size=sdl_bs,
        shuffle=secondary_shuffle,
        num_workers=self.num_workers)

    self.test_dataloader = torch.utils.data.DataLoader(
        self.test_dataset,
        batch_size=self.args.test_batch_size, 
        shuffle=False, 
        num_workers=self.num_workers)
    


def load_real_test(self, dataset_type):
    if dataset_type == 'cifar10':
        self.real_test_dataset = datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=self.transform_test)
        self.real_test_dataset = torch.utils.data.Subset(
            self.real_test_dataset, range(self.args.train_dataset_size, self.args.train_dataset_size + self.args.real_test_dataset_size))
    
    if dataset_type == 'cifar100':
        self.real_test_dataset = datasets.CIFAR100(
            root='./data', 
            train=False, 
            download=True, 
            transform=self.transform_test)


    self.real_test_dataloader = torch.utils.data.DataLoader(
        self.real_test_dataset,
        batch_size=self.args.test_batch_size, 
        shuffle=False, 
        num_workers=self.num_workers)