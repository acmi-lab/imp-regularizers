'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os, sys
import time
import torch
from copy import deepcopy
from utils.tensor_defns import *
import torch.nn as nn
import torch.nn.init as init

def random_noise(arr_shape, dist="Gauss"):
    if dist == 'Gauss':
        return torch.normal(0, 1, size=arr_shape).to(device)
    else:
        assert dist == "T", dist
        return torch.normal(0, 1, size=arr_shape).to(device)


def save_grad(clones):
    '''Util function to save gradients for later analysis
    '''
    model_grads = []
    for acc in clones:
        grads = {}
        temp = []
        for n, p in acc.named_parameters():         
            if n not in grads:
                grads[n] = p.grad

        for k in grads:
            temp.append(grads[k].view(-1))                    
        model_grads.append(np.squeeze(torch.cat(temp, 0).detach().cpu().numpy()))

    return np.array(model_grads)

class MaskCollect:
    '''Collects gradients for gradient descent
    TODO: add hyperparameters for noise
    '''
    def __init__(self, masks, cpu=False):
        self.masks = masks
        self.gradient_init = dict()
        self.length = 0
        self.cpu = cpu
        self.outputs = []
        self.targets = []
    
    def accumulate(self, net):
        self.length += 1
        for name, param in net.named_parameters():
            if self.cpu:
                param_arr = deepcopy(param.grad).cpu()
            else:
                param_arr = deepcopy(param.grad)
            if self.gradient_init.get(name) is None:
                self.gradient_init[name] = []
            self.gradient_init[name].append(param_arr.to(torch.device('cpu')) * self.masks[name].to(torch.device('cpu')))
            
    def combine(self):
        assert len(self.gradient_init) != 0, "Running Dictionary is empty"
        for name in self.gradient_init.keys():
            self.gradient_init[name] = torch.stack(self.gradient_init[name], dim=0)

    def calculate_moments(self, var=False, lwsv=False):
        self.mean_dict = dict.fromkeys(self.gradient_init.keys(),None)
        if var:
            self.var_dict = dict.fromkeys(self.gradient_init.keys(),None)
        for name in self.gradient_init.keys():
            self.mean_dict[name] = self.gradient_init[name].mean(dim=0).to(device)
            if var:
                self.var_dict[name] = torch.var(self.gradient_init[name], dim=0, unbiased=False).to(device)
                if lwsv:
                    self.var_dict[name] = torch.ones_like(self.var_dict[name]) \
                    * torch.mean(torch.var(self.gradient_init[name], dim=0, unbiased=False).to(device))

    def calculate_noise(self):
        self.noise = {}
        for name in self.gradient_init.keys():
            self.noise[name] = self.gradient_init[name] - self.mean_dict[name].to(torch.device('cpu'))

    def get_keys(self):
        assert len(self.gradient_init) != 0, "Running Dictionary is empty"
        return self.gradient_init.keys()

    def __len__(self):
        return self.length


class IndexCollect:
    '''Collects gradients for gradient descent
    TODO: add hyperparameters for noise
    '''
    def __init__(self, indices, cpu=False):
        self.indices = indices
        self.gradient_init = dict()
        self.length = 0
        self.cpu = cpu
        self.outputs = []
        self.targets = []
        self.all_data = None
    
    def accumulate(self, net):
        self.length += 1
        for name, param in net.named_parameters():
            if self.indices == 'all':
                if self.cpu:
                    param_arr = deepcopy(param.grad.view(-1)).cpu()
                else:
                    param_arr = deepcopy(param.grad.view(-1))
                if self.gradient_init.get(name) is None:
                    self.gradient_init[name] = []
                self.gradient_init[name].append(param_arr)
            elif name in self.indices.keys():
                if self.cpu:
                    param_arr = deepcopy(param.grad.view(-1)[self.indices[name]]).cpu()
                else:
                    param_arr = deepcopy(param.grad.view(-1)[self.indices[name]])
                if self.gradient_init.get(name) is None:
                    self.gradient_init[name] = []
                self.gradient_init[name].append(param_arr)
            
    def combine(self):
        assert len(self.gradient_init) != 0, "Running Dictionary is empty"
        for name in self.gradient_init.keys():
            self.gradient_init[name] = torch.stack(self.gradient_init[name], dim=0)
    
    def full_combine(self):
        self.all_data = torch.cat(list(self.gradient_init.values()), dim=1)

    def calculate_moments(self, var=False, lwsv=False):
        if self.all_data is None:
            self.mean_dict = dict.fromkeys(self.gradient_init.keys(),None)
            if var:
                self.var_dict = dict.fromkeys(self.gradient_init.keys(),None)
            for name in self.gradient_init.keys():
                self.mean_dict[name] = self.gradient_init[name].mean(dim=0).to(device)
                if var:
                    self.var_dict[name] = torch.var(self.gradient_init[name], dim=0, unbiased=True).to(device)
                    if lwsv:
                        self.var_dict[name] = torch.ones_like(self.var_dict[name]) \
                        * torch.mean(torch.var(self.gradient_init[name], dim=0, unbiased=False).to(device))
        else:
            self.mean_tens = self.all_data.mean(axis=0).to(device)
            if var:
                self.var_tens = torch.var(self.all_data, dim=0, unbiased=False).to(device)
                if lwsv:
                    self.var_dict[name] = torch.ones_like(self.var_tens) \
                    * torch.mean(torch.var(self.all_data, dim=0, unbiased=False).to(device))
            

    def calculate_noise(self, var=False):
        self.noise = {}
        for name in self.gradient_init.keys():
            self.noise[name] = self.gradient_init[name] - self.mean_dict[name]
            if var:
                self.noise[name] /= torch.sqrt(self.var_dict[name].to(torch.device('cpu')))
            

    def get_keys(self):
        assert len(self.gradient_init) != 0, "Running Dictionary is empty"
        return self.gradient_init.keys()

    def __len__(self):
        return self.length


class GradientCollect:
    '''Collects gradients for gradient descent
    TODO: add hyperparameters for noise
    '''
    def __init__(self, cpu=False):
        self.gradient_init = dict()
        self.length = 0
        self.cpu = cpu
        self.outputs = []
        self.targets = []
    
    def accumulate(self, net):
        self.length += 1
        for name, param in net.named_parameters():
            if self.cpu:
                param_arr = deepcopy(param.grad).cpu()
            else:
                param_arr = deepcopy(param.grad)
            if self.gradient_init.get(name) is None:
                self.gradient_init[name] = []
            self.gradient_init[name].append(param_arr)
    
    def acc_ot(self, outputs, targets):
        self.length += 1
        self.outputs.append(outputs)
        self.targets.append(targets)
    
    def cat_ot(self):
        outputs = torch.cat(self.outputs, axis=0)
        targets = torch.cat(self.targets, axis=0)
        return outputs, targets
            
    def combine(self):
        assert len(self.gradient_init) != 0, "Running Dictionary is empty"
        for name in self.gradient_init.keys():
            self.gradient_init[name] = torch.stack(self.gradient_init[name], dim=0)

    def calculate_moments(self, var=False, lwsv=False, kurt=False):
        self.mean_dict = dict.fromkeys(self.gradient_init.keys(),None)
        if var:
            self.var_dict = dict.fromkeys(self.gradient_init.keys(),None)
        if kurt:
            self.kurt_dict = dict.fromkeys(self.gradient_init.keys(),None)
        for name in self.gradient_init.keys():
            self.mean_dict[name] = self.gradient_init[name].mean(dim=0).to(device)
            if var:
                self.var_dict[name] = torch.var(self.gradient_init[name], dim=0, unbiased=True).to(device)
                if lwsv:
                    self.var_dict[name] = torch.ones_like(self.var_dict[name]) \
                    * torch.mean(torch.var(self.gradient_init[name], dim=0, unbiased=True).to(device))
            if kurt:
                self.kurt_dict[name] = ((self.gradient_init[name] - self.mean_dict[name])**4).mean(dim=0) / self.var_dict[name]**2 - 3
                self.kurt_dict[name][self.kurt_dict[name] <=0 ] = 1e-3
                assert torch.all(self.kurt_dict[name] > 0)


    def sample_empirical(self):
        self.random_grad_dict = dict.fromkeys(self.gradient_init.keys(),None)
        for name in self.gradient_init.keys():
            grad_arr = self.gradient_init[name]
            arr_shape = grad_arr.shape[1:]
            grad_arr_reshape = grad_arr.reshape(self.length, -1)
            random_vals = self.get_random_vals(self.length, arr_shape)
            out = grad_arr_reshape[random_vals, np.arange(np.prod(arr_shape))]
            self.random_grad_dict[name] = out.reshape(arr_shape)

    
    def get_random_vals(self, max_val, arr_shape):
        out = np.random.randint(max_val, size=np.prod(arr_shape))
        return out

    def get_keys(self):
        assert len(self.gradient_init) != 0, "Running Dictionary is empty"
        return self.gradient_init.keys()

    def reset(self):
        self.gradient_init = dict()

    def __len__(self):
        return self.length

class GradientCollect2:
    '''Collects gradients for gradient descent
    TODO: add hyperparameters for noise
    '''
    def __init__(self):
        self.prev_init = dict()
        self.gradient_init = dict()
        self.length = 0
    
    def accumulate(self, net):
        self.length += 1
        for name, param in net.named_parameters():
            param_arr = param.grad
            if self.gradient_init.get(name) is None:
                self.gradient_init[name] = []
                self.gradient_init[name].append(param_arr)
            else:
                self.gradient_init[name].append(param_arr - self.prev_init[name])
            self.prev_init[name] = param_arr
            
    def combine(self):
        for name in self.gradient_init.keys():
            self.gradient_init[name] = torch.stack(self.gradient_init[name], dim=0)

    def calculate_moments(self, var=True, mean_of_mean=False):
        self.mean_dict = dict.fromkeys(self.gradient_init.keys(),None)
        if var:
            self.var_dict = dict.fromkeys(self.gradient_init.keys(),None)
        for name in self.gradient_init.keys():
            self.mean_dict[name] = torch.mean(self.gradient_init[name], dim=0)
            if var:
                self.var_dict[name] = torch.var(self.gradient_init[name], dim=0, unbiased=True)
                if mean_of_mean:
                    self.var_dict[name] = torch.ones_like(self.var_dict[name]) * torch.mean(torch.var(self.gradient_init[name], dim=0, keepdim=True, unbiased=True))
    
    def sample_empirical(self):
        self.random_grad_dict = dict.fromkeys(self.gradient_init.keys(),None)
        for name in self.gradient_init.keys():
            grad_arr = self.gradient_init[name]
            arr_shape = grad_arr.shape[1:]
            grad_arr_reshape = grad_arr.reshape(self.length, -1)
            random_vals = self.get_random_vals(self.length, arr_shape)
            out = grad_arr_reshape[random_vals, np.arange(np.prod(arr_shape))]
            self.random_grad_dict[name] = out.reshape(arr_shape)
    
    def get_random_vals(self, max_val, arr_shape):
        out = np.random.randint(max_val, size=np.prod(arr_shape))
        return out

    def get_keys(self):
        if not self.bgrad:
            assert len(self.gradient_init) != 0, "Running Dictionary is empty"
            return self.gradient_init.keys()
        else:
            return self.bgradient_init.keys()

    def reset(self):
        self.gradient_init = dict()

    def __len__(self):
        return self.length

class GraftCollect(GradientCollect):
    
    def __init__(self, graft_idx, cap, use_graft=True, denoise=False, decorr=False):
        super().__init__()
        self.graft_idx = graft_idx
        self.cap = cap
        self.graft = {}
        self.cossym = None
        if type(self.graft_idx) != int:
            self.cossym = self.graft_idx
            self.cossym_list = {}
            self.graft_idx = np.random.choice(self.graft_idx)
        self.use_graft = use_graft
        self.denoise = denoise
        self.decorr = decorr
        #self.test_decorr = {}
    
    def make_ind_mask(self, net):
        self.indices = {}
        for name, param in net.named_parameters():
            self.indices[name] = torch.randint(high=self.cap, size=param.shape).to(device)



    def accumulate(self, net):
        for name, param in net.named_parameters():
            param_arr = deepcopy(param.grad)
            if self.denoise and self.use_graft:
                if self.graft.get(self.length) is None:
                    self.graft[self.length] = 0
                self.graft[self.length] += param_arr.norm(2).item() ** 2
            elif self.use_graft and self.length == self.graft_idx:
                self.graft[name] = param_arr

            if self.cossym is not None and self.length in self.cossym:
                if self.cossym_list.get(self.length) is None:
                    self.cossym_list[self.length] = {}
                    self.cossym_list[self.length][name] = param_arr
                else:
                    self.cossym_list[self.length][name] = param_arr

            if self.decorr:
                # if self.test_decorr.get(name) is None:
                #     self.test_decorr[name] = [param_arr]
                # else:
                #     self.test_decorr[name].append(param_arr)
                param_arr = torch.where(self.indices[name] == self.length, 1, 0) * param_arr    
                
            if self.gradient_init.get(name) is None:
                self.gradient_init[name] = param_arr
            else:
                self.gradient_init[name] += param_arr
        if self.denoise and self.use_graft:
            self.graft[self.length] = self.graft[self.length] ** (1. / 2)
        self.length += 1
    
    def combine(self):
        if self.decorr:
            # for name in self.test_decorr.keys():
            #     self.test_decorr[name] = torch.stack(self.test_decorr[name], dim=0)
            return
        for name in self.gradient_init.keys():
            self.gradient_init[name] /= self.length
        if self.denoise:
            self.graft_norm = np.mean(list(self.graft.values()))
    
    def avg_cossym(self):
        assert self.cossym is not None
        cos_syms = []
        for gradient in self.cossym_list.keys():
            total_gd = 0
            total_sgd = 0
            sim_sum = 0
            for name, grad, in self.gradient_init.items():
                sgd_norm = self.cossym_list[gradient][name].data.norm(2)
                gd_norm = grad.data.norm(2)
                total_gd += gd_norm.item() ** 2
                total_sgd +=  sgd_norm.item() ** 2
                sim_sum += (self.cossym_list[gradient][name] * grad).sum().item()
            total_gd = total_gd ** (1. / 2)
            total_sgd = total_sgd ** (1. / 2)
            cos_syms.append(sim_sum / (total_gd * total_sgd))
        return np.mean(cos_syms)

def make_network_copy(net, net_clone):
    net_clone.zero_grad()
    net_clone.load_state_dict(net.state_dict())
    return net_clone

def set_gradients_of_network(grad_dict_base, grad_dict_new, net, is_gd=False):
    '''Reassign gradients after sampling from a given noise distribution.
    Current: Gaussian.
    '''
    assert len(grad_dict_base) != 0, "Running Dictionary is empty"
    if not is_gd:
        constant = 1
    else:
        constant = 0
    for name, param in net.named_parameters():
        assert hasattr(param, 'grad')
        noise = torch.multiply(
                torch.sqrt(grad_dict_base.var_dict[name]),
                random_noise(grad_dict_base.var_dict[name].shape))
        param_grad_tensor = Tensor(grad_dict_new.mean_dict[name] + constant * noise)
        param.grad = param_grad_tensor
    return net

def set_gradients_of_network2(var_dict, mean_dict, net, constant=1):
    '''Reassign gradients after sampling from a given noise distribution.
    Current: Gaussian.
    '''
    for name, param in net.named_parameters():
        assert hasattr(param, 'grad')
        noise = torch.multiply(
                torch.sqrt(var_dict[name]),
                random_noise(var_dict[name].shape))
        param_grad_tensor = Tensor(mean_dict[name] + constant * noise)
        param.grad = param_grad_tensor
    return net


def set_gradients(net, grad_dict):
    '''Reassign gradients
    '''
    assert len(grad_dict) != 0, "Dictionary is empty"
    for name, param in net.named_parameters():
        param_grad_tensor = Tensor(grad_dict[name])
        param.grad = param_grad_tensor
    return net

def get_network_weights(net):
    return_dict = dict()
    for name, param in net.named_parameters():
        weight = param.detach().cpu().numpy()
        return_dict[name] = weight
    return return_dict

def get_all_flattened(input_dict):
    '''For each array held within key flatten it up.
    '''
    all_array = np.array([0])
    for key in input_dict.keys():
        all_array = np.concatenate((all_array, input_dict[key].flatten()))
    return_val = all_array[1:]
    return all_array

def get_network_gradients(net):
    return_dict = dict()
    for name, param in net.named_parameters():
        return_dict[name] = param.grad
    return return_dict

def get_labels(classifier, img_tensor):
    pred_logits = classifier(img_tensor)
    pred_labels = pred_logits.argmax(1)
    return pred_labels

def get_accuracy(pred_labels, real_labels):
    num_correct = (pred_labels == real_labels).float().sum()
    acc = (num_correct / len(real_labels)) * 100.
    return acc


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def random_weights(model, n):
    '''
    Return random sample of weights from a given model
    '''
    ne_dict = {n: p.numel() for n, p in model.named_parameters()}
    ne_list = [(n, p.numel()) for n, p in model.named_parameters()]
    n_list = [x[0] for x in ne_list]
    e_list = [x[1] for x in ne_list]
    num_layers = len(ne_list)
    tot_size = sum(e_list)
    layers, counts = np.unique(
        np.random.choice(n_list, n, p=[x/tot_size for x in e_list]),
        return_counts=True
    )
    out_weights = {}
    for layer, count in zip(layers, counts):
        out_weights[layer] = np.random.choice(ne_dict[layer], count, replace=False)
    return out_weights

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)

try:
    _, term_width = os.popen('stty size', 'r').read().split()
except:
    term_width = "80"
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f