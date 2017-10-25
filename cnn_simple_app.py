#coding：utf8
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from utils import load_data_fashion_mnist

ctx =mx.cpu()

#超参数
batch_size = 256
num_inputs = 28*28
num_hiddens = 256
num_outputs = 10
learning_rate = .01

#搭建模型
