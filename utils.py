#conding:utf8
from mxnet import gluon
from mxnet import image
from mxnet import ndarray as nd
try:
    import matplotlib.pyplot as plt
except:
    plt = None
def SGD(params,lr):
    for param in params:
        param[:] = param - lr*param.grad


def data_transform(data,label):
    return data.astype('float32')/255,label.astype('float32')

#加载数据
def load_data_fashion_mnist(batch_size,resize=None):

    def data_transform(data,label):
        if resize:
            data  = image.imresize(data,resize,resize)
        return data.astype('float32')/255,label.astype('float32')
    mnist_train = gluon.data.vision.FashionMNIST(train=True,transform=data_transform)
    mnist_test = gluon.data.vision.FashionMNIST(train=False,transform=data_transform)

    ## 每次迭代产生batch_size个数据
    train_data = gluon.data.DataLoader(mnist_train,batch_size,shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test,batch_size,shuffle=False)
    return train_data,test_data


def accuracy(output,label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()


def evaluate_accuracy(data_iterator,net):
    total_acc =0.0
    for data,label in data_iterator:
        output = net(data)
        total_acc+=accuracy(output,label)
    return total_acc/len(data_iterator)

def softmax(X):
    exp = nd.exp(X)
    partition = exp.sum(axis=1,keepdims=True)
    return exp/partition

class Result(object):
    def __init__(self,result_map=None):
        if not result_map:
            self.result_map = {"train_loss":[],"train_acc":[],"test_acc":[]}
        else:
            self.result_map = result_map

    def add(self,key,value):
        self.result_map[key].append(value)

    def show(self):
        ax = plt.subplot(111)
        train_loss_list = self.result_map.get("train_loss",[])
        train_acc_list = self.result_map.get("train_acc",[])
        test_acc_list = self.result_map.get("test_acc",[])

        ax.plot(list(range(len(train_loss_list))),train_loss_list,color='r',label="train_loss")
        ax.plot(list(range(len(train_loss_list))),train_acc_list,color='g',label="train_acc")
        ax.plot(list(range(len(train_loss_list))),test_acc_list,color='b',label="test_acc")
        ax.legend(loc=1,ncol=3,shadow=True)
        plt.show()
