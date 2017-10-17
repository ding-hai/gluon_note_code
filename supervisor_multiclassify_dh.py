#coding:utf8
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon
from utils import SGD,load_data_fashion_mnist,evaluate_accuracy,accuracy,Result

import matplotlib.pyplot as plt
# 定义参数
## 超参数
learning_rate = 0.01
num_inputs = 28*28
num_outputs = 10
batch_size = 10
## 模型参数
W = nd.random_normal(shape=(num_inputs,num_outputs))
b = nd.random_normal(shape=(num_outputs))
params = [W,b]

#加载数据
train_data,test_data = load_data_fashion_mnist(batch_size)

#定义模型
def softmax(X):
    exp = nd.exp(X)
    partition = exp.sum(axis=1,keepdims=True)
    return exp/partition

def net(X):
    return softmax(nd.dot(X.reshape((-1,num_inputs)),W)+b)

#损失函数
## 精确度
## 移入到utils.py中
# def accuracy(output,label):
#     return nd.mean(output.argmax(axis=1) == label).asscalar()

## 交叉熵
def cross_entropy(yhat,y):
    return -nd.pick(nd.log(yhat),y)

# # 计算精度
## 移入到utils.py中
# def evaluate_accuracy(data_iterator,net):
#     total_acc =0.0
#     for data,label in data_iterator:
#         output = net(data)
#         total_acc+=accuracy(output,label)
#     return total_acc/len(data_iterator)


#优化
for param in params:
    param.attach_grad()

result = Result()
num_batchs = len(train_data)
for epoch in range(10):
    train_loss = 0.0
    train_acc = 0.0
    for data,label in train_data:
        with ag.record() :
            output = net(data)
            loss = cross_entropy(output,label)
        loss.backward()
        SGD(params,learning_rate/batch_size)
        train_loss+=nd.mean(loss).asscalar()
        train_acc+= accuracy(output,label)

    test_acc = evaluate_accuracy(test_data,net)
    result.add("train_loss",train_loss/num_batchs)
    result.add("test_acc",test_acc)
    result.add("train_acc",train_acc/num_batchs)
    print("Epoch %d: train_loss = %f train_acc = %f test_acc = %f "%(epoch,train_loss/num_batchs,train_acc/num_batchs,test_acc))
result.show()
