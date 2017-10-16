#coding:utf8
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon
from utils import SGD,load_data_fashion_mnist,accuracy,evaluate_accuracy
import random
import matplotlib.pyplot  as plt

learning_rate = 0.04
batch_size = 10
num_inputs= 28*28
num_outputs = 10
W = nd.random_normal(shape=(num_inputs,num_outputs))
b = nd.random_normal(shape=(num_outputs))
params = [W,b]

train_data,test_data = load_data_fashion_mnist(batch_size)

for param in params:
    param.attach_grad()

def softmax(X):
    exp = nd.exp(X)
    partition = exp.sum(axis=1,keepdims=True)
    return exp/partition

def net(X):
    return softmax(nd.dot(X.reshape((-1,num_inputs)),W)+b)


def cross_entropy(yhat,y):
    return -nd.pick(nd.log(yhat),y)

# 移入到utils.py
# def accuracy(output,label):
#     acc = nd.mean(output.argmax(axis=1)==label).asscalar()
#     return acc
#
# def evaluate_accuracy(data_iterator,net):
#     acc = 0
#     for data,label in data_iterator:
#         output = net(data)
#         acc += accuracy(output,label)
#     return acc / len(data_iterator)

#train
for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data,label in train_data:
        with ag.record():
            output = net(data)
            loss = cross_entropy(output,label)
        loss.backward()
        SGD(params,learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output,label)
    test_acc = evaluate_accuracy(test_data, net)

    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))



def show_images(images):
    n = images.shape[0]
    _,figs = plt.subplots(1,n,figsize=(15,15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28,28)).asnumpy())
    plt.show()
def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in label]

# 预测
# data,labels = mnist_test[0:9]
# print("real label")
# print(get_text_labels(labels))
# print("predict label")
# predict_labels = net(data).argmax(axis=1)
# print(get_text_labels(predict_labels.asnumpy()))
# show_images(data)
