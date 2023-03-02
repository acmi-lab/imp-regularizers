import os
import sys
import json
import argparse
import numpy as np
import random
from math import ceil
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
parser.add_argument('--save-step', type=int, default=25000)
parser.add_argument('--exter-run', type=str, default='Vanilla|0.1|128-1')
parser.add_argument('--exter-lambda', type=float, default=1.)

TESTS = ['Vanilla', 'PseudoGD',  'RegLoss', 'FishLoss', 'JacLoss', 'UnitJacLoss', 'AvgJacLoss']


class SGDTrainer:
    def __init__(self, input_args=None, is_train=True):
        self.is_train = is_train
        self.args = input_args
        self.num_workers = self.args.num_workers
        self.passed = True
        self.lr = self.args.learning_rate
        self.batch_size = self.args.micro_batch_size
        self.secondary_batch_size = self.args.batch_size
        self.train_dataset_size = self.args.train_dataset_size
        if self.args.save_step > self.args.log_steps[-1]:
            self.args.save_step = self.args.log_steps[-1] // 2

        
        # ==== Import train/test dataloaders ====
        load_dataset(self, self.args.dataset)
        self.test_type = self.args.test

        if self.test_type not in TESTS:
            raise ValueError(f"Invalid test type: {self.test_type}")
        self.test_batch_size = self.args.test_batch_size
        self.logging = self.args.no_logging
        if self.args.test_name == '0':
            self.project_name = self.args.model.upper() + self.args.dataset
        else:
            self.project_name = self.args.test_name + self.args.model.upper() + self.args.dataset
        print(bcolors.c_cyan(self.project_name))
        if self.logging:
            self.run = wandb.init(
                reinit=True,
                project=self.project_name, 
                config={'batch_size': self.batch_size,
                        'secondary_batch_size': self.secondary_batch_size,
                        'learning_rate': self.lr,
                        'seed': self.args.seed})
            if self.args.run_name == "0":
                if self.test_type == 'Vanilla':
                    idx = wandb.run.name.split("-")[-1]
                    wandb.run.name = f"{self.test_type}|{self.args.learning_rate}|{self.batch_size}-{idx}"
                elif self.test_type == 'PseudoGD':
                    idx = wandb.run.name.split("-")[-1]
                    wandb.run.name = f"{self.test_type}|{self.args.learning_rate}|{self.batch_size}|{self.secondary_batch_size}-{idx}"
                elif self.test_type in ['RegLoss', 'UnitJacLoss', 'FishLoss', 'AvgJacLoss']:
                    idx = wandb.run.name.split("-")[-1]
                    wandb.run.name = f"{self.test_type}|{self.args.learning_rate}|{self.batch_size}|{self.secondary_batch_size}|{self.args.exter_lambda}-{idx}"
                else:
                    raise ValueError(f"Invalid test type: {self.test_type}")
            else:
                wandb.run.name = self.args.run_name + "-" + wandb.run.name.split("-")[-1]

            wandb.run.save()
            self.out_dir = os.path.join(
                 'saved_models/' + self.project_name, 
                wandb.run.name)
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

        if self.test_type == 'Vanilla':
            self.iter_count = 1
        else:
            self.iter_count = ceil(self.secondary_batch_size / self.batch_size)
        
        load_dataloader(self)
        assert self.args.dataset in ['cifar10', 'cifar100'], self.args.dataset
        if self.args.model == 'resnet':
            if self.args.dataset == 'cifar100':
                self.net = ResNet18(num_classes=100)
            else:
                self.net = ResNet18()
        elif self.args.model == 'vgg11':
            # TODO: add cifar100 support
            self.net = vgg11_no_dropout()
        elif self.args.model == 'vgg16':
            if self.args.dataset == 'cifar100':
                self.net = torchm.vgg16_bn(num_classes=100)
            else:
                self.net = torchm.vgg16_bn(num_classes=10)
        
        if use_cuda:
            self.net.cuda()
            cudnn.benchmark = True

        
        if self.args.checkpoint != "0":
            self.net.load_state_dict(torch.load(self.args.checkpoint)['main_network'])

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr, 
            momentum=0,
            weight_decay=0
        )


    def log_gradient_norms(self, net, step, name="base"):
        total_norm = 0
        for n, p in net.named_parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        if name == 'base':
            wandb.log({"train/gradient_norm": total_norm, "custom_step": step})
            wandb.log({"train/step_length": total_norm * self.lr, "custom_step": step})
        else:
            wandb.log({f"train/gradient_norm_{name}": total_norm, "custom_step": step})
            wandb.log({f"train/step_length_{name}": total_norm * self.lr, "custom_step": step})

    def prep_inputs(self, inputs, targets):
        inputs, targets = inputs.to(device), targets.to(device)
        return inputs, targets

    def save_checkpoint(self, save_path, test_acc=None):
        model_dict = dict()
        model_dict['main_network'] = self.net.state_dict()
        if test_acc is not None:
            model_dict['test_acc'] = test_acc
        torch.save(model_dict, save_path)

    def test(self, step, epoch, print_it=True):
        self.net.eval()
        avg_test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_dataloader):
                inputs, targets = self.prep_inputs(inputs, targets)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                batch_loss = loss.item()
                avg_test_loss += batch_loss
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            avg_test_loss = avg_test_loss / len(self.test_dataset)
            if print_it:
                progress_bar(
                    batch_idx, len(self.test_dataloader), 
                    bcolors.c_red('Epoch Num: %d | Avg Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (epoch, avg_test_loss, 100.*correct/total, correct, total)))

            if self.logging:
                wandb.log({"test/avg_test_loss": avg_test_loss, 'custom_step': step})
                wandb.log({"test/test_accuracy": 100.*correct/total, 'custom_step': step})
        return 100.*correct/total

    def get_exter_reg(self, net, step):
        total_norm = 0.
        for n, p in net.named_parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        exter_norm = self.norm_data.loc[(step // 20) * 20][self.norm_idx]
        reg_term = (exter_norm - total_norm)**2
        if step % self.args.logging_period == 0:
            wandb.log({"train/gradient_norm_noreg": total_norm, "custom_step": step})
            print(exter_norm)
            print(reg_term)
        return reg_term
    
    def log_loss_term_grad_norm(self, grads, step, name="base"):
        cur_norm = None
        for grad in grads:
            if cur_norm is None:
                cur_norm = grad.pow(2).sum()
            else:
                cur_norm += grad.pow(2).sum()
        total_norm = cur_norm ** (1. / 2)
        if name == 'base':
            wandb.log({"train/loss_term_gradient_norm": total_norm, "custom_step": step})
            wandb.log({"train/loss_term_step_length": total_norm * self.lr, "custom_step": step})
        else:
            wandb.log({f"train/loss_term_gradient_norm_{name}": total_norm, "custom_step": step})
            wandb.log({f"train/loss_term_step_length_{name}": total_norm * self.lr, "custom_step": step})

    def get_batch(self):
        inputs, targets = next(iter(self.secondary_dataloader))
        inputs, targets = inputs.to(device), targets.to(device)
        return inputs, targets

    def log_avg_sgd_norm(self, step, num_iters,regs=None):
        if self.test_type == 'RegLoss' and self.batch_size == 128:
            wandb.log({"train/avg_sgd_norm": regs, 'custom_step': step})
            return
        else:
            self.net.train()
            norms = np.zeros(num_iters)
            for i in range(num_iters):
                inputs, targets = self.get_batch()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                grads = torch.autograd.grad(loss, inputs=self.net.parameters())
                cur_norm = None
                for grad in grads:
                    if cur_norm is None:
                        cur_norm = grad.pow(2).sum()
                    else:
                        cur_norm += grad.pow(2).sum()
                norms[i] = cur_norm
            wandb.log({"train/avg_sgd_norm": norms.mean(), 'custom_step': step})

    def train(self, num_epochs):
        not_vanilla = self.test_type not in ["Vanilla", 'DropGraft']
        step = 0
        num_iters = ceil(self.train_dataset_size / (self.iter_count * self.batch_size))
        best_acc = None
        for epoch in range(num_epochs):
            running_mean = 0
            correct = 0
            total = 0
            avg_train_acc = []
            avg_test_acc = [] 
            iters = 0
            
            self.optimizer.zero_grad()
            if step % self.args.logging_period == 0:
                log_grads = None
                log_regs = 0
            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
                
                self.net.train()
                inputs, targets = self.prep_inputs(inputs, targets)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                if self.test_type not in ['Vanilla', 'PseudoGD']:
                    if self.test_type == 'FishLoss':
                        pred_dist = torch.distributions.categorical.Categorical(logits=outputs)
                        yhats = pred_dist.sample()
                        grads = torch.autograd.grad(self.criterion(outputs, yhats), inputs=self.net.parameters(), create_graph=True)
                    elif self.test_type == 'AvgJacLoss':
                        grads = torch.autograd.grad(outputs.mean(axis=0).sum(), inputs=self.net.parameters(), create_graph=True)
                    elif self.test_type == 'UnitJacLoss':
                        unit_vec = torch.randn(outputs.shape[1]).to(device)
                        unit_vec /= unit_vec.norm(2)
                        grads = torch.autograd.grad(outputs.mean(axis=0) @ unit_vec, inputs=self.net.parameters(), create_graph=True)
                    elif self.test_type == 'RegLoss':
                        grads = torch.autograd.grad(loss, inputs=self.net.parameters(), create_graph=True)


                    if step % self.args.logging_period == 0:
                        if log_grads is None:
                            log_grads = []
                            for grad in grads:
                                log_grads.append(grad / self.iter_count)
                        else:
                            new_grads = []
                            for i in range(len(log_grads)):
                                new_grads.append(log_grads[i] + grads[i] / self.iter_count)
                            log_grads = new_grads
                    cur_norm = None
                    for grad in grads:
                        if cur_norm is None:
                            cur_norm = grad.pow(2).sum()
                        else:
                            cur_norm += grad.pow(2).sum()
                    reg_term = cur_norm
                    if step % self.args.logging_period == 0:
                        log_regs += (reg_term / self.iter_count)
                    reg_loss = loss + self.args.exter_lambda * reg_term
                    reg_loss /= self.iter_count
                    reg_loss.backward()
                else:
                    loss /= self.iter_count
                    loss.backward()

                if self.test_type in ['RegLoss', 'FishLoss', 'UnitJacLoss', 'AvgJacLoss']:
                    batch_loss = reg_loss.item() * self.iter_count
                else:
                    batch_loss = loss.item() * self.iter_count
                running_mean += batch_loss / self.iter_count
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                randreg = False
                if (batch_idx + 1) % self.iter_count == 0 or batch_idx == len(self.dataloader) - 1:
                    if step % self.args.logging_period == 0:
                        if self.test_type in ['RegLoss', 'FishLoss', 'UnitJacLoss', 'AvgJacLoss']:
                            self.log_gradient_norms(self.net, step)
                            self.log_loss_term_grad_norm(log_grads, step)
                            wandb.log({"train/reg_term": log_regs, 'custom_step': step})
                            wandb.log({"train/reg_term_lambda": self.args.exter_lambda * log_regs, 'custom_step': step})
                            self.log_avg_sgd_norm(step, self.iter_count, log_regs)
                            log_grads = None
                            log_regs = 0
                        else:
                            self.log_avg_sgd_norm(step, self.iter_count)

                    # ==== Take your gradient step ====
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    randreg = True
                    # ==== Start logging ====
                    logging_loss = running_mean/(iters + 1)
                    acc = 100.*correct/total
                    progress_bar(
                        iters, num_iters, 
                        bcolors.c_cyan('Epoch Num: %d, Batch Loss: %.3f | Avg Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (epoch, batch_loss, logging_loss, acc, correct, total)))
                    if step % self.args.logging_period == 0:
                        if self.logging:
                            wandb.log({"train/batch_loss": batch_loss, 'custom_step': step})
                            wandb.log({"train/avg_loss": logging_loss, 'custom_step': step})
                            wandb.log({"train/accuracy": acc, 'custom_step': step})
                        test_acc = self.test(step, epoch, print_it=False)
                        if step > self.args.save_step and (best_acc is None or test_acc > best_acc):
                            self.save_checkpoint(
                                os.path.join(self.out_dir, f'checkpoint_best.pth'),
                                test_acc
                            )
                            best_acc = test_acc
                        avg_test_acc.append(test_acc)

                    avg_train_acc.append(acc)
                    
                    step += 1
                    if step in self.args.log_steps:
                        self.save_checkpoint(
                            os.path.join(self.out_dir, f'checkpoint{step}.pth'),
                            test_acc
                        )
                        if step == self.args.log_steps[-1]:
                            return
                    iters += 1
            else:
                test_acc = self.test(step, epoch)

    def final_acc(self, path=None):
        load_real_test(self, self.args.dataset)
        if path is None:
            raise ValueError(f"Invalid path: {path}")
        else:
            best_mod = torch.load(path)
        self.net.load_state_dict(
            best_mod['main_network']
        )
        print(best_mod['test_acc'])
        self.net.eval()
        avg_test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.real_test_dataloader):
                inputs, targets = self.prep_inputs(inputs, targets)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                batch_loss = loss.item()
                avg_test_loss += batch_loss
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            avg_test_loss = avg_test_loss / len(self.real_test_dataset)
            # wandb.log({"test/final_avg_test_loss": avg_test_loss})
            # wandb.log({"test/final_test_accuracy": 100.*correct/total})
        print("Test Loss:", avg_test_loss)
        print("Test Accuracy:", 100.*correct/total)
        return 100.*correct/total



def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
