import torch
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def get_loss_scalar(loss_var):
    if float(torch.__version__[:3]) >= 0.4:
        return loss_var.detach().cpu().numpy()
    else:
        if type(loss_var) is torch.autograd.Variable:
            return loss_var.data.cpu().numpy()[0]
        else: 
            return loss_var.cpu().numpy()[0]

def convert_tensor_image_to_np(tensor):
    '''This function right now assumes that the tensor is an image'''
    data = tensor.cpu().numpy()
    data = np.squeeze(data)
    data = np.transpose(data, (0, 2, 3, 1))
    return data

def to_np(tensor):
    '''converts torch tensor to numpy
    '''
    data = tensor.data.cpu().numpy()
    return data

def get_tensor(arr, data_type=Tensor):
    ''' arr is a numpy array'''
    return torch.from_numpy(arr).type(Tensor)

def get_rand_spherical(batch_size, ndims):
    normal = torch.randn(batch_size, ndims)
    spherical = torch.nn.functional.normalize(normal, dim=1)
    return spherical

def get_image_np(tensor, img_size=28, reshape=True):
    img_np = tensor.cpu().numpy()
    if reshape:
        img_np = img_np.reshape(28,28)
    return img_np

def str2bool(v):
    '''This is useful for argparse
    '''
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')